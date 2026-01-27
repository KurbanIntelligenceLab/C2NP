import torch
import torch.nn as nn
from torch_geometric.nn.models import SchNet


class MatterGen_Task2(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 16,
        num_layers: int = 1,
        cutoff: float = 5.0,
        num_spacegroups: int = 230,
        time_emb_dim: int = 16,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
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

        # Projection layer
        self.proj = nn.Linear(1, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
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
        """Get noise schedule for time t"""
        t = torch.clamp(t, min=1e-3, max=0.999)
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

        # Time embedding
        t_emb = self.time_emb(t.view(-1, 1))  # [B, time_emb_dim]

        # Predict noise for lattice parameters
        lat_noise = self.lat_net(torch.cat([g, t_emb], dim=-1))
        lat_noise = torch.clamp(lat_noise, min=-5.0, max=5.0)

        # Predict space group
        sg_logits = self.sg_head(g)

        return lat_noise, sg_logits

    def sample(self, data, num_steps: int = 1000, chunk_size: int = 100):
        """Generate samples using the reverse diffusion process"""
        device = next(self.parameters()).device
        B = int(data.ptr.size(0) - 1)

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
                noise_pred, _ = self.forward(data, t)

                # Update lattice parameters
                alpha = 1 - self.get_noise_schedule(t)
                alpha = alpha.view(-1, 1)  # [B, 1]

                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) * noise_pred)
                x = torch.clamp(x, min=-5.0, max=5.0)

                if torch.isnan(x).any():
                    x = torch.nan_to_num(x, nan=0.0)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Get final space group prediction
        _, sg_logits = self.forward(data, torch.zeros(B, device=device))

        return x, sg_logits
