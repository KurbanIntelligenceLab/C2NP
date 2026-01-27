import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.nn.models import SchNet

# Allowlist needed globals for torch.load
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


class MemoryEfficientAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, chunk_size=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: [N, C] where N is number of nodes and C is hidden_dim
        N, C = x.shape

        # Project queries, keys, values
        q = (
            self.q_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        )  # [H, N, D]
        k = (
            self.k_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        )  # [H, N, D]
        v = (
            self.v_proj(x).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        )  # [H, N, D]

        # Process in chunks to save memory
        output = []
        for i in range(0, N, self.chunk_size):
            chunk_end = min(i + self.chunk_size, N)
            q_chunk = q[:, i:chunk_end]  # [H, chunk_size, D]

            # Compute attention scores for this chunk
            attn = (q_chunk @ k.transpose(-2, -1)) * self.scale  # [H, chunk_size, N]

            if mask is not None:
                attn = attn.masked_fill(mask[i:chunk_end] == 0, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)

            # Apply attention to values
            chunk_output = (attn @ v).transpose(0, 1)  # [chunk_size, H, D]
            output.append(chunk_output)

            # Clear cache after each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Combine chunks
        x = torch.cat(output, dim=0)  # [N, H, D]
        x = x.reshape(N, C)  # [N, C]
        x = self.out_proj(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, chunk_size=128):
        super().__init__()
        self.attention = MemoryEfficientAttention(
            hidden_dim, num_heads, dropout, chunk_size
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        # Self-attention with gradient checkpointing
        def attn_block(x):
            attn_output = self.attention(x, mask)
            return self.norm1(x + attn_output)

        # Feed-forward with gradient checkpointing
        def ff_block(x):
            return self.norm2(x + self.ff(x))

        # Use gradient checkpointing for both blocks
        x = checkpoint(attn_block, x, use_reentrant=False)
        x = checkpoint(ff_block, x, use_reentrant=False)

        return x


class ADiTUnitCell(nn.Module):
    """
    All-atom Diffusion Transformer for crystal structure prediction.
    Based on the ADiT paper while maintaining consistency with DiffCSP parameters.
    """

    def __init__(
        self,
        atom_emb_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 1,
        cutoff_radius: float = 5.0,
        r_emb_dim: int = 16,
        max_atomic_number: int = 100,
        time_emb_dim: int = 16,
        beta_min: float = 0.01,
        beta_max: float = 2.0,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.hidden_dim = hidden_dim

        # Atom embedding
        self.atom_emb = nn.Embedding(max_atomic_number, atom_emb_dim)
        nn.init.xavier_uniform_(self.atom_emb.weight)

        # Unit-cell encoder (SchNet)
        self.unitcell_encoder = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_layers,
            num_gaussians=5,
            readout="add",
            cutoff=cutoff_radius,
        )

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Radius embedding
        self.r_emb = nn.Sequential(
            nn.Linear(1, r_emb_dim),
            nn.SiLU(),
            nn.Linear(r_emb_dim, r_emb_dim),
            nn.SiLU(),
        )

        # Initial feature projection
        self.input_proj = nn.Linear(atom_emb_dim + 3, hidden_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final noise prediction network
        self.noise_net = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim + r_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    @staticmethod
    def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        diffs = ptr[1:] - ptr[:-1]
        return torch.repeat_interleave(
            torch.arange(diffs.size(0), device=ptr.device), diffs
        )

    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise schedule for time t with more stable values"""
        # Clip t to prevent extreme values
        t = torch.clamp(t, min=1e-3, max=0.999)
        # Use a more conservative noise schedule
        beta = self.beta_min + (self.beta_max - self.beta_min) * (t**3)
        return torch.clamp(beta, min=0.1, max=10.0)

    def forward(self, data: Batch, t: torch.Tensor):
        # Prepare cell_ptr for batching
        raw_ptr = data.cell_ptr
        if raw_ptr.numel() > 2:
            assert raw_ptr.numel() % 2 == 0, "Expected even-length cell_ptr"
            cp2d = raw_ptr.view(-1, 2)
            lens = cp2d[:, 1]
            cell_ptr = torch.cat([lens.new_zeros(1), lens.cumsum(dim=0)], dim=0)
        else:
            cell_ptr = raw_ptr

        # Encode unit-cell graph
        cell_batch = self._batch_from_ptr(cell_ptr)
        h_cell = self.unitcell_encoder(
            data.cell_z, data.cell_pos, cell_batch
        )  # [B, hidden_dim]

        # Time and radius embeddings
        t_emb = self.time_emb(t.view(-1, 1))  # [B, time_emb_dim]
        radius = data.radius.view(-1, 1)
        r_emb = self.r_emb(radius)  # [B, r_emb_dim]

        # Prepare nanoparticle graph
        batch_idx = self._batch_from_ptr(data.ptr)

        # Initial feature processing
        atom_feat = self.atom_emb(data.z)  # [N, atom_emb_dim]
        x = torch.cat([atom_feat, data.pos], dim=-1)  # [N, atom_emb_dim + 3]
        x = self.input_proj(x)  # [N, hidden_dim]

        # Broadcast cell, time, and radius embeddings to nodes
        h_cell_node = h_cell[batch_idx]  # [N, hidden_dim]
        t_node = t_emb[batch_idx]  # [N, time_emb_dim]
        r_node = r_emb[batch_idx]  # [N, r_emb_dim]

        # Add cell features to node features
        x = x + h_cell_node

        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Combine features and process through noise prediction network
        x = torch.cat([x, t_node, r_node], dim=-1)
        noise_pred = self.noise_net(x)  # [N, 3]

        # Clip noise predictions to prevent extreme values
        noise_pred = torch.clamp(noise_pred, min=-5.0, max=5.0)

        # Check for NaN values and replace them
        if torch.isnan(noise_pred).any():
            noise_pred = torch.nan_to_num(noise_pred, nan=0.0)

        return noise_pred

    def sample(self, data: Batch, num_steps: int = 1000, chunk_size: int = 100):
        """Generate samples using the reverse diffusion process"""
        device = next(self.parameters()).device
        x = torch.randn_like(data.pos)  # Start from random noise
        x = torch.clamp(x, min=-5.0, max=5.0)  # Initial clipping

        # Get batch indices for proper broadcasting
        batch_idx = self._batch_from_ptr(data.ptr)

        # Process in chunks to manage memory
        for i in range(0, num_steps, chunk_size):
            # Calculate current timestep range
            current_steps = min(chunk_size, num_steps - i)
            t = torch.ones(data.num_graphs, device=device) * (
                1 - (i + current_steps - 1) / num_steps
            )

            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():  # Disable gradient computation during sampling
                noise_pred = self.forward(data, t)

                # Update positions using predicted noise
                alpha = 1 - self.get_noise_schedule(t)
                alpha_nodes = alpha[batch_idx].view(-1, 1)  # [num_nodes, 1]

                # Now dimensions will match: [num_nodes, 3]
                x = (1 / torch.sqrt(alpha_nodes)) * (x - (1 - alpha_nodes) * noise_pred)

                # Clip positions to prevent extreme values
                x = torch.clamp(x, min=-5.0, max=5.0)

                # Check for NaN values and replace them
                if torch.isnan(x).any():
                    x = torch.nan_to_num(x, nan=0.0)

                # Clear cache after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return x
