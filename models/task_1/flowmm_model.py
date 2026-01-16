import math

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import GCN, SchNet


class FlowMMUnitCell(nn.Module):
    """
    Flow Matching model for unit cell to nanoparticle generation.
    Debug-enabled version with detailed print statements.
    """

    def __init__(
        self,
        atom_emb_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 1,
        cutoff_radius: float = 5.0,
        r_emb_dim: int = 16,
        time_emb_dim: int = 16,
        max_atomic_number: int = 100,
        beta_min: float = 0.01,
        beta_max: float = 2.0,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Atom embedding
        self.atom_emb = nn.Embedding(max_atomic_number, atom_emb_dim)
        nn.init.xavier_uniform_(self.atom_emb.weight)

        # Unit-cell encoder
        self.unitcell_encoder = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_layers,
            num_gaussians=5,
            readout="add",
            cutoff=cutoff_radius,
        )

        # Projection and normalization
        self.proj = nn.Linear(1, hidden_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(hidden_dim)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        for layer in self.time_emb:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Radius embedding
        self.r_emb = nn.Sequential(
            nn.Linear(1, r_emb_dim),
            nn.SiLU(),
            nn.Linear(r_emb_dim, r_emb_dim),
            nn.SiLU(),
        )
        for layer in self.r_emb:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Decoder
        in_channels = atom_emb_dim + 3 + hidden_dim + time_emb_dim + r_emb_dim
        self.decoder = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
        )

        # Prediction head
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3)
        )
        for layer in self.pos_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

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

    # Override noise schedule to use cumulative alpha_bar
    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        # return cumulative alpha_bar directly
        ab = self.get_alpha_bar(t)
        return ab

    def forward(self, data: Batch, t: torch.Tensor):
        if torch.isnan(data.pos).any():
            data.pos = torch.nan_to_num(data.pos, nan=0.0)

        cell_ptr = data.cell_ptr
        if cell_ptr.numel() > 2:
            cp2d = cell_ptr.view(-1, 2)
            lens = cp2d[:, 1]
            cell_ptr = torch.cat([lens.new_zeros(1), lens.cumsum(dim=0)], dim=0)

        cell_batch = self._batch_from_ptr(cell_ptr)
        g = self.unitcell_encoder(data.cell_z, data.cell_pos, cell_batch)
        g = self.norm(self.proj(g))
        g = torch.nan_to_num(g, nan=0.0)

        t_emb = self.time_emb(t.view(-1, 1))
        r_emb = self.r_emb(data.radius.view(-1, 1))

        batch_idx = self._batch_from_ptr(data.ptr)
        alpha_bar = self.get_noise_schedule(t).view(-1, 1)
        sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-6))
        sqrt_umb = torch.sqrt(torch.clamp(1 - alpha_bar, min=1e-6))

        noise = torch.randn_like(data.pos)
        noisy_pos = sqrt_ab[batch_idx] * data.pos + sqrt_umb[batch_idx] * noise
        data.pos = noisy_pos

        edge_index = radius_graph(data.pos, self.cutoff_radius, batch=batch_idx)
        atom_f = self.atom_emb(data.z)
        node_features = torch.cat(
            [atom_f, data.pos, g[batch_idx], t_emb[batch_idx], r_emb[batch_idx]], dim=-1
        )
        node_features = torch.nan_to_num(node_features, nan=0.0)
        h = self.decoder(node_features, edge_index)
        noise_pred = self.pos_head(h)
        return torch.clamp(noise_pred, min=-1.0, max=1.0)

    def sample(self, data: Batch, num_steps: int = 1000, chunk_size: int = 100):
        device = next(self.parameters()).device
        B = int(data.ptr.size(0) - 1)
        x = torch.randn_like(data.pos) * 0.01
        x = torch.clamp(x, min=-0.1, max=0.1)
        batch_idx = self._batch_from_ptr(data.ptr)
        for i in range(0, num_steps, chunk_size):
            t = torch.ones(B, device=device) * (
                1 - (i + min(chunk_size, num_steps - i) - 1) / num_steps
            )
            data.pos = x
            noise_pred = self.forward(data, t)
            alpha_bar = self.get_noise_schedule(t).view(-1, 1)
            x = x - 0.01 * (1 - alpha_bar[batch_idx]) * noise_pred
            x = torch.clamp(x, min=-1.0, max=1.0)
            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
        return x
