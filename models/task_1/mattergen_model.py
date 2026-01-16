import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.nn.models import SchNet

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


class MatterGenUnitCell(nn.Module):
    """
    MatterGen-style model for crystal structure prediction.
    Inputs:
      data.cell_z, data.cell_pos, data.cell_ptr  -> unit-cell graph
      data.radius                              -> scalar R
      data.z, data.pos, data.ptr               -> target nanoparticle graph
    Outputs:
      noise_pred: [Np,3] predicted noise
    """

    def __init__(
        self,
        atom_emb_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 1,
        cutoff_radius: float = 5.0,
        time_emb_dim: int = 16,
        max_atomic_number: int = 100,
    ):
        super().__init__()
        # Store dimensions as instance variables
        self.atom_emb_dim = atom_emb_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.cutoff_radius = cutoff_radius

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
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Initial feature projection
        self.input_proj = nn.Linear(atom_emb_dim + 3, hidden_dim)

        # Noise prediction network
        self.noise_net = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.noise_net[0].weight)
        nn.init.zeros_(self.noise_net[0].bias)
        nn.init.xavier_uniform_(self.noise_net[2].weight)
        nn.init.zeros_(self.noise_net[2].bias)
        nn.init.xavier_uniform_(self.noise_net[4].weight)
        nn.init.zeros_(self.noise_net[4].bias)

    @staticmethod
    def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        diffs = ptr[1:] - ptr[:-1]
        return torch.repeat_interleave(
            torch.arange(diffs.size(0), device=ptr.device), diffs
        )

    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        """Linear noise schedule as in MatterGen"""
        return t

    def forward(self, data: Batch, t: torch.Tensor):
        # Prepare cell_ptr for batching
        raw_ptr = data.cell_ptr
        if raw_ptr.numel() > 2:
            assert raw_ptr.numel() % 2 == 0
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
        r_emb = self.r_emb(data.radius.view(-1, 1))  # [B, hidden_dim]

        # Prepare nanoparticle graph
        batch_idx = self._batch_from_ptr(data.ptr)

        # Initial feature processing
        atom_feat = self.atom_emb(data.z)  # [N, atom_emb_dim]
        x = torch.cat([atom_feat, data.pos], dim=-1)  # [N, atom_emb_dim + 3]
        x = self.input_proj(x)  # [N, hidden_dim]

        # Broadcast embeddings
        h_cell_node = h_cell[batch_idx]  # [N, hidden_dim]
        t_node = t_emb[batch_idx]  # [N, time_emb_dim]
        r_node = r_emb[batch_idx]  # [N, hidden_dim]

        # Combine features
        x = torch.cat(
            [x + h_cell_node, t_node, r_node], dim=-1
        )  # [N, hidden_dim + time_emb_dim + hidden_dim]

        # Predict noise
        noise_pred = self.noise_net(x)  # [N, 3]
        return noise_pred

    def sample(
        self, data: Batch, num_steps: int = 1000, chunk_size: int = 100
    ) -> torch.Tensor:
        """Sample final positions using DDPM with chunked processing"""
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
