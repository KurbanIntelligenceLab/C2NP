import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import GCN, SchNet
from transformers import AutoModel, AutoTokenizer


class FlowLLM_Task2(nn.Module):
    """Flow Matching model with LLM base distribution for structure generation."""

    def __init__(
        self,
        atom_emb_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 2,
        cutoff_radius: float = 5.0,
        max_atomic_number: int = 100,
        beta_min: float = 0.01,
        beta_max: float = 2.0,
        llm_model_name: str = "prajjwal1/bert-tiny",  # Use TinyBERT for efficiency
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
        self.encoder = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=1,
            num_gaussians=3,
            readout="add",
            cutoff=cutoff_radius,
            max_num_neighbors=8,
        )

        # Simple projection for SchNet output
        self.encoder_proj = nn.Linear(1, hidden_dim)
        nn.init.xavier_uniform_(self.encoder_proj.weight)
        nn.init.zeros_(self.encoder_proj.bias)

        # TinyBERT for base distribution
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModel.from_pretrained(llm_model_name)

        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Projection from LLM hidden states to node features
        self.llm_proj = nn.Linear(self.llm.config.hidden_size, hidden_dim)
        nn.init.xavier_uniform_(self.llm_proj.weight)
        nn.init.zeros_(self.llm_proj.bias)

        # Decoder for full graph
        self.decoder = GCN(
            in_channels=atom_emb_dim + 3 + hidden_dim,  # atom_emb + pos + encoder_out
            hidden_channels=hidden_dim,
            num_layers=1,
            dropout=0.0,
        )

        # Prediction head for positions
        self.pos_head = nn.Linear(hidden_dim, 3)
        nn.init.xavier_uniform_(self.pos_head.weight)
        nn.init.zeros_(self.pos_head.bias)

        # Lattice parameter prediction network
        self.lat_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
            nn.Linear(hidden_dim, 230),  # 230 space groups
        )
        nn.init.xavier_uniform_(self.sg_head[0].weight)
        nn.init.zeros_(self.sg_head[0].bias)
        nn.init.xavier_uniform_(self.sg_head[2].weight)
        nn.init.zeros_(self.sg_head[2].bias)

    @staticmethod
    def _batch_from_ptr(ptr: torch.Tensor) -> torch.Tensor:
        return torch.repeat_interleave(
            torch.arange(ptr.size(0) - 1, device=ptr.device), ptr[1:] - ptr[:-1]
        )

    def get_beta(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=1e-6, max=0.999)
        beta = self.beta_min + (self.beta_max - self.beta_min) * (
            1 - torch.cos(torch.pi * t)
        )
        return torch.clamp(beta, min=self.beta_min, max=self.beta_max)

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=0.0, max=1.0)
        integral = self.beta_min * t + (self.beta_max - self.beta_min) * (
            t - torch.sin(torch.pi * t) / torch.pi
        )
        alpha_bar = torch.exp(-integral)
        return alpha_bar

    def get_noise_schedule(self, t: torch.Tensor) -> torch.Tensor:
        return self.get_alpha_bar(t)

    def _get_llm_features(self, data: Batch) -> torch.Tensor:
        """Get features from TinyBERT for all nodes."""
        device = next(self.parameters()).device

        # Convert atomic numbers to text descriptions
        atom_descriptions = []
        for z in data.z:
            atom_descriptions.append(f"atom with atomic number {z.item()}")

        # Process in smaller batches to avoid CUDA memory issues
        batch_size = 8  # Process 8 atoms at a time
        all_features = []

        for i in range(0, len(atom_descriptions), batch_size):
            batch_descriptions = atom_descriptions[i : i + batch_size]

            # Tokenize descriptions
            inputs = self.tokenizer(
                batch_descriptions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get TinyBERT hidden states
            with torch.no_grad():
                outputs = self.llm(**inputs)
                hidden_states = outputs.last_hidden_state

            # Project to node feature dimension
            batch_features = self.llm_proj(hidden_states[:, 0, :])  # Use [CLS] token
            all_features.append(batch_features)

            # Clear CUDA cache after each batch
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Concatenate all features
        return torch.cat(all_features, dim=0)

    def forward(self, data: Batch, t: torch.Tensor):
        if torch.isnan(data.pos).any():
            data.pos = torch.nan_to_num(data.pos, nan=0.0)

        batch_idx = self._batch_from_ptr(data.ptr)

        # Encode graph
        g = self.encoder(data.z, data.pos, batch_idx)
        g = torch.nan_to_num(g, nan=0.0)
        g = self.encoder_proj(g.view(-1, 1))

        # Get LLM features for all nodes
        llm_features = self._get_llm_features(data)

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

        # Concatenate features
        node_features = torch.cat([atom_f, pos_f, g_f], dim=-1)

        # Decode
        h = self.decoder(node_features, edge_index)

        # Add LLM features
        h = h + llm_features

        # Get predictions
        noise_pred = self.pos_head(h)
        lat_pred = self.lat_net(g)  # Use global features for lattice prediction
        sg_logits = self.sg_head(g)  # Use global features for space group prediction

        return torch.clamp(noise_pred, min=-1.0, max=1.0), lat_pred, sg_logits

    def sample(self, data: Batch, num_steps: int = 1000, chunk_size: int = 100):
        device = next(self.parameters()).device
        B = int(data.ptr.size(0) - 1)

        # Initialize positions with small random noise
        x = torch.randn_like(data.pos) * 0.01
        x = torch.clamp(x, min=-0.1, max=0.1)

        # Initialize lattice parameters with small random noise
        lat = torch.randn(B, 6, device=device) * 0.01
        lat = torch.clamp(lat, min=-0.1, max=0.1)

        batch_idx = self._batch_from_ptr(data.ptr)

        for i in range(0, num_steps, chunk_size):
            t = torch.ones(B, device=device) * (
                1 - (i + min(chunk_size, num_steps - i) - 1) / num_steps
            )
            data.pos = x
            noise_pred, lat_pred, sg_logits = self.forward(data, t)

            alpha_bar = self.get_noise_schedule(t).view(-1, 1)
            x = x - 0.01 * (1 - alpha_bar[batch_idx]) * noise_pred
            x = torch.clamp(x, min=-1.0, max=1.0)

            # Update lattice parameters
            lat = lat - 0.01 * (1 - alpha_bar) * lat_pred
            lat = torch.clamp(lat, min=-1.0, max=1.0)

            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)
            if torch.isnan(lat).any():
                lat = torch.nan_to_num(lat, nan=0.0)

        return lat, sg_logits
