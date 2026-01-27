import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.models import SchNet


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


class ADiT_Task2(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 16,
        num_layers: int = 1,
        cutoff: float = 5.0,
        num_spacegroups: int = 230,
        time_emb_dim: int = 16,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        num_heads: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 128,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.hidden_dim = hidden_dim

        # Graph encoder (SchNet)
        self.encoder = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_layers,
            num_gaussians=5,
            readout="add",
            cutoff=cutoff,
        )

        # Projection layer to ensure correct output dimension
        self.proj = nn.Linear(1, hidden_dim)

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, dropout, chunk_size)
                for _ in range(num_layers)
            ]
        )

        # Noise prediction network for lattice parameters
        self.lat_net = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

        # Space group classifier
        self.sg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_spacegroups),
        )

        # Initialize weights
        nn.init.zeros_(self.lat_net[-1].weight)
        nn.init.zeros_(self.lat_net[-1].bias)

    @staticmethod
    def batch_from_ptr(ptr):
        return torch.repeat_interleave(
            torch.arange(ptr.size(0) - 1, device=ptr.device), ptr[1:] - ptr[:-1]
        )

    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Get noise schedule for time t with more stable values"""
        # Clip t to prevent extreme values
        t = torch.clamp(t, min=1e-3, max=0.999)
        # Use a more conservative noise schedule
        beta = self.beta_min + (self.beta_max - self.beta_min) * (t**3)
        return torch.clamp(beta, min=0.1, max=10.0)

    def forward(self, data, t):
        # Get batch indices
        batch = self.batch_from_ptr(data.ptr)

        # Encode graph
        g = self.encoder(data.z, data.pos, batch)  # [B, 1]
        g = self.proj(g)  # Project to [B, hidden_dim]
        g = self.norm(g)  # Normalize
        g = torch.nan_to_num(g, nan=0.0)  # Handle NaN values

        # Process through transformer blocks
        for block in self.transformer_blocks:
            g = block(g)

        # Time embedding
        t_emb = self.time_emb(t.view(-1, 1))  # [B, time_emb_dim]

        # Predict noise for lattice parameters
        lat_noise = self.lat_net(torch.cat([g, t_emb], dim=-1))
        lat_noise = torch.clamp(lat_noise, min=-5.0, max=5.0)

        # Predict space group
        sg_logits = self.sg_head(g)

        return lat_noise, sg_logits

    def sample(self, data, num_steps: int = 1000, chunk_size: int = 50):
        """Generate samples using the reverse diffusion process"""
        device = next(self.parameters()).device
        B = int(data.ptr.size(0) - 1)

        # Start from random noise for lattice parameters
        x = torch.randn(B, 6, device=device)  # [B, 6]
        x = torch.clamp(x, min=-5.0, max=5.0)  # Initial clipping

        # Process in chunks to manage memory
        for i in range(0, num_steps, chunk_size):
            # Calculate current timestep range
            current_steps = min(chunk_size, num_steps - i)
            t = torch.ones(B, device=device) * (1 - (i + current_steps - 1) / num_steps)

            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                noise_pred, _ = self.forward(data, t)

                # Update lattice parameters
                alpha = 1 - self.get_noise_schedule(t)
                alpha = alpha.view(-1, 1)  # [B, 1]

                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) * noise_pred)
                x = torch.clamp(x, min=-5.0, max=5.0)

                # Check for NaN values and replace them
                if torch.isnan(x).any():
                    x = torch.nan_to_num(x, nan=0.0)

                # Clear cache after each chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Get final space group prediction
        _, sg_logits = self.forward(data, torch.zeros(B, device=device))

        return x, sg_logits
