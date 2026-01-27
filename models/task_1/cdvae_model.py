import torch
import torch.nn as nn
from torch_geometric.data import Batch

# 1) Allowlist GlobalStorage so torch.load(..., weights_only=True) works
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import GCN, SchNet

# 1) Allowlist needed globals before any torch.load(...)
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


class CDVAEUnitCell(nn.Module):
    """
    VAE that conditions on a primitive unit cell and a target radius R.
    Inputs:
      data.cell_z, data.cell_pos, data.cell_ptr  -> unit-cell graph
      data.radius                              -> scalar R
      data.z, data.pos, data.ptr               -> target nanoparticle graph
    Outputs:
      recon_pos: [Np,3], mu: [B,latent_dim], logvar: [B,latent_dim]
    """

    def __init__(
        self,
        atom_emb_dim: int = 16,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        num_layers: int = 1,
        cutoff_radius: float = 5.0,
        r_emb_dim: int = 16,
        max_atomic_number: int = 100,
        delta_clip: float = 0.5,
    ):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.delta_clip = delta_clip

        # Atom embedding for decoder
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

        # VAE bottleneck from unit-cell embedding
        self.fc_mu = nn.Linear(1, latent_dim)
        self.fc_logvar = nn.Linear(1, latent_dim)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight)
        self.fc_logvar.bias.data.fill_(-4.0)

        # Radius embedding MLP
        self.r_emb = nn.Sequential(
            nn.Linear(1, r_emb_dim),
            nn.SiLU(),
            nn.Linear(r_emb_dim, r_emb_dim),
            nn.SiLU(),
        )

        # Decoder GCN: input = atom_emb + pos + z_lat + r_emb
        in_channels = atom_emb_dim + 3 + latent_dim + r_emb_dim
        self.decoder_net = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=0.1,
        )
        self.coord_out = nn.Linear(hidden_dim, 3)

    @staticmethod
    def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        # Convert ptr=[0, n0, n0+n1, ...] to per-node batch indices
        diffs = ptr[1:] - ptr[:-1]
        return torch.repeat_interleave(
            torch.arange(diffs.size(0), device=ptr.device), diffs
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data: Batch):
        # --- Prepare a correct cell_ptr for batching ---
        raw_ptr = data.cell_ptr
        if raw_ptr.numel() > 2:
            # tensor([0, n0, 0, n1, 0, n2, ...]) -> reshape into rows [0,n0], [0,n1], ...
            assert raw_ptr.numel() % 2 == 0, "Expected even-length cell_ptr"
            cp2d = raw_ptr.view(-1, 2)
            lens = cp2d[:, 1]
            # build cumulative ptr: [0, n0, n0+n1, n0+n1+n2, ...]
            cell_ptr = torch.cat([lens.new_zeros(1), lens.cumsum(dim=0)], dim=0)
        else:
            cell_ptr = raw_ptr

        # Encode unit-cell graph
        cell_batch = self._batch_from_ptr(cell_ptr)
        h_cell = self.unitcell_encoder(data.cell_z, data.cell_pos, cell_batch)  # [B,1]

        # VAE bottleneck
        mu = self.fc_mu(h_cell)
        logvar = self.fc_logvar(h_cell)
        z_lat = self.reparameterize(mu, logvar)

        # Radius embedding
        radius = data.radius.view(-1, 1)  # [B,1]
        r_emb = self.r_emb(radius)  # [B, r_emb_dim]

        # Decode nanoparticle graph
        batch_idx = self._batch_from_ptr(data.ptr)
        edge_index = radius_graph(data.pos, self.cutoff_radius, batch=batch_idx)

        # Broadcast latent and radius embeddings
        z_node = z_lat[batch_idx]
        r_node = r_emb[batch_idx]

        # Atom features + positions + conditioning
        atom_feat = self.atom_emb(data.z)
        dec_in = torch.cat([atom_feat, data.pos, z_node, r_node], dim=-1)

        # GCN decode to coordinate shifts
        h_dec = self.decoder_net(dec_in, edge_index)
        raw_delta = self.coord_out(h_dec)
        delta = self.delta_clip * torch.tanh(raw_delta)

        recon_pos = data.pos + delta
        return recon_pos, mu, logvar
