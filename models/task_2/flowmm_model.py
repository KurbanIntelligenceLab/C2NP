import math

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import GCN, SchNet


class FlowMMCrystal(nn.Module):
    """
    Flow Matching model for crystal structure generation.
    Inputs:
      data.z, data.pos, data.ptr  -> crystal graph
      data.cell_params           -> unit cell parameters
    Outputs:
      noise_pred: [N,3] predicted noise for positions
    """

    def __init__(
        self,
        atom_emb_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        cutoff_radius: float = 5.0,
        cell_emb_dim: int = 16,
        time_emb_dim: int = 16,
        max_atomic_number: int = 100,
        beta_min: float = 0.01,
        beta_max: float = 2.0,
        num_spacegroups: int = 230,  # Added parameter for number of space groups
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.hidden_dim = hidden_dim

        # Atom embedding
        self.atom_emb = nn.Embedding(max_atomic_number, atom_emb_dim)
        nn.init.xavier_uniform_(self.atom_emb.weight)

        # Lightweight SchNet encoder
        self.crystal_encoder = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=1,  # Minimal interactions
            num_gaussians=3,  # Reduced gaussians
            readout="add",
            cutoff=cutoff_radius,
            max_num_neighbors=8,  # Limit neighbors to save memory
        )

        # Simple projection for SchNet output
        self.crystal_proj = nn.Linear(1, hidden_dim)
        nn.init.xavier_uniform_(self.crystal_proj.weight)
        nn.init.zeros_(self.crystal_proj.bias)

        # Simplified cell parameters embedding
        self.cell_emb = nn.Sequential(nn.Linear(6, cell_emb_dim), nn.SiLU())
        nn.init.xavier_uniform_(self.cell_emb[0].weight)
        nn.init.zeros_(self.cell_emb[0].bias)

        # Simplified time embedding
        self.time_emb = nn.Sequential(nn.Linear(1, time_emb_dim), nn.SiLU())
        nn.init.xavier_uniform_(self.time_emb[0].weight)
        nn.init.zeros_(self.time_emb[0].bias)

        # Simplified projection layer
        self.proj = nn.Sequential(
            nn.Linear(
                atom_emb_dim + 3 + hidden_dim + cell_emb_dim + time_emb_dim, hidden_dim
            ),
            nn.SiLU(),
        )
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.zeros_(self.proj[0].bias)

        # Simplified decoder
        self.decoder = GCN(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            num_layers=1,
            dropout=0.0,
        )

        # Simplified prediction head
        self.pos_head = nn.Linear(hidden_dim, 3)
        nn.init.xavier_uniform_(self.pos_head.weight)
        nn.init.zeros_(self.pos_head.bias)

        # Lattice parameter prediction network
        self.lat_net = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # 6 lattice parameters
        )
        nn.init.xavier_uniform_(self.lat_net[0].weight)
        nn.init.zeros_(self.lat_net[0].bias)
        nn.init.xavier_uniform_(self.lat_net[2].weight)
        nn.init.zeros_(self.lat_net[2].bias)
        nn.init.xavier_uniform_(self.lat_net[4].weight)
        nn.init.zeros_(self.lat_net[4].bias)

        # Space group prediction head
        self.sg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_spacegroups),
        )
        nn.init.xavier_uniform_(self.sg_head[0].weight)
        nn.init.zeros_(self.sg_head[0].bias)
        nn.init.xavier_uniform_(self.sg_head[2].weight)
        nn.init.zeros_(self.sg_head[2].bias)

    @staticmethod
    def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        diffs = ptr[1:] - ptr[:-1]
        return torch.repeat_interleave(
            torch.arange(diffs.size(0), device=ptr.device), diffs
        )

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=1e-6, max=0.999)
        beta = self.beta_min + (self.beta_max - self.beta_min) * (
            1 - torch.cos(math.pi * t)
        )
        return torch.clamp(beta, min=self.beta_min, max=self.beta_max)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=0.0, max=1.0)
        integral = self.beta_min * t + (self.beta_max - self.beta_min) * (
            t - torch.sin(math.pi * t) / math.pi
        )
        alpha_bar = torch.exp(-integral)
        return alpha_bar

    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        return self.get_alpha_bar(t)

    def forward(self, data: Batch, t: torch.Tensor):
        if torch.isnan(data.pos).any():
            data.pos = torch.nan_to_num(data.pos, nan=0.0)

        batch_idx = self._batch_from_ptr(data.ptr)

        # Encode crystal structure with lightweight SchNet
        g = self.crystal_encoder(data.z, data.pos, batch_idx)
        g = torch.nan_to_num(g, nan=0.0)
        g = self.crystal_proj(g.view(-1, 1))

        # Time embedding
        t_emb = self.time_emb(t.view(-1, 1))

        # Cell parameters embedding
        cell_emb = self.cell_emb(data.cell_params)

        # Add noise to positions
        alpha_bar = self.get_noise_schedule(t).view(-1, 1)
        sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-6))
        sqrt_umb = torch.sqrt(torch.clamp(1 - alpha_bar, min=1e-6))

        noise = torch.randn_like(data.pos) * 0.01
        noisy_pos = sqrt_ab[batch_idx] * data.pos + sqrt_umb[batch_idx] * noise
        data.pos = noisy_pos

        # Build graph
        edge_index = radius_graph(
            data.pos, self.cutoff_radius, batch=batch_idx, max_num_neighbors=8
        )

        # Prepare node features
        atom_f = self.atom_emb(data.z)
        pos_f = data.pos
        g_f = g[batch_idx]
        cell_f = cell_emb[batch_idx]
        time_f = t_emb[batch_idx]

        # Concatenate features
        node_features = torch.cat([atom_f, pos_f, g_f, cell_f, time_f], dim=-1)

        # Project to hidden dimension
        node_features = self.proj(node_features)

        # Decode
        h = self.decoder(node_features, edge_index)
        noise_pred = self.pos_head(h)

        # Lattice parameter prediction (using global features)
        lat_noise = self.lat_net(torch.cat([g, t_emb], dim=-1))

        # Space group prediction (using global features)
        sg_logits = self.sg_head(g)

        return torch.clamp(noise_pred, min=-1.0, max=1.0), lat_noise, sg_logits

    def sample(self, data: Batch, num_steps: int = 1000, chunk_size: int = 100):
        device = next(self.parameters()).device
        B = int(data.ptr.size(0) - 1)

        # Add cell parameters if not present
        if not hasattr(data, "cell_params"):
            # Initialize with default values (cubic cell)
            data.cell_params = torch.ones((B, 6), device=device)
            data.cell_params[:, 3:] = torch.tensor(
                [90.0, 90.0, 90.0], device=device
            )  # angles in degrees

        # Start from random noise for lattice parameters
        x = torch.randn(B, 6, device=device)  # [B, 6]
        x = torch.clamp(x, min=-5.0, max=5.0)  # Initial clipping

        # Process in chunks to manage memory
        for i in range(0, num_steps, chunk_size):
            current_steps = min(chunk_size, num_steps - i)
            t = torch.ones(B, device=device) * (1 - (i + current_steps - 1) / num_steps)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                _, lat_noise, _ = self.forward(data, t)

                # Update lattice parameters
                alpha = 1 - self.get_noise_schedule(t)
                alpha = alpha.view(-1, 1)  # [B, 1]

                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) * lat_noise)
                x = torch.clamp(x, min=-5.0, max=5.0)

                if torch.isnan(x).any():
                    x = torch.nan_to_num(x, nan=0.0)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Get final space group prediction
        _, _, sg_logits = self.forward(data, torch.zeros(B, device=device))

        return x, sg_logits
