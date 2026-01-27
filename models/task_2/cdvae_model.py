import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet


class CDVAE_Task2(nn.Module):
    def __init__(
        self,
        hidden_dim=32,
        latent_dim=64,
        num_layers=2,
        cutoff=5.0,
        num_spacegroups=230,
    ):
        super().__init__()

        self.enc = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=num_layers,
            num_gaussians=10,
            cutoff=cutoff,
            readout="add",
        )

        # ── LayerNorm on scalar ── ★
        self.norm = nn.LayerNorm(1)

        self.fc_mu = nn.Linear(1, latent_dim, bias=True)
        self.fc_logvar = nn.Linear(1, latent_dim, bias=True)
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_logvar.weight)
        self.fc_logvar.bias.data.fill_(-2.0)

        # ── Heads ──
        self.lat_head = nn.Linear(latent_dim, 6, bias=True)
        self.sg_head = nn.Linear(latent_dim, num_spacegroups)

        # start regression head at zero → avoids NaN in first forward ★
        nn.init.zeros_(self.lat_head.weight)
        nn.init.zeros_(self.lat_head.bias)

    @staticmethod
    def batch_from_ptr(ptr):
        return torch.repeat_interleave(
            torch.arange(ptr.size(0) - 1, device=ptr.device), ptr[1:] - ptr[:-1]
        )

    @staticmethod
    def reparam(mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        batch = self.batch_from_ptr(data.ptr)
        g = self.enc(data.z, data.pos, batch)  # [B,1]

        g = self.norm(g)  # ★ keep mean≈0, var≈1
        g = torch.nan_to_num(g, nan=0.0)  # ★ clamp any rogue nan/inf

        mu, lv = self.fc_mu(g), self.fc_logvar(g).clamp(-10, 10)

        z = self.reparam(mu, lv)
        lat_pred = torch.nan_to_num(self.lat_head(z), nan=0.0).clamp(-1e4, 1e4)
        sg_logits = self.sg_head(z)

        return lat_pred, sg_logits, mu, lv
