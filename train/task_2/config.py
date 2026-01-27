"""
Central configuration for Task 2 training.

All training hyperparameters, model architectures, and data settings are defined here.
All training scripts in task_2/ should import and use these configurations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# DATA_ROOT is resolved relative to the current working directory
# This makes it dynamic - it will point to C2NP in whatever directory the script is run from
# Using a function to resolve at runtime, not at import time
def get_data_root() -> str:
    """Get DATA_ROOT resolved relative to current working directory."""
    return str(Path.cwd() / "C2NP")

DATA_ROOT = "C2NP"  # Default relative path (will be resolved dynamically)
SUBSET_RATIO = 1.0  # Fraction of dataset to use (1.0 = 100%)
TRAIN_VAL_SPLIT = (0.8, 0.2)  # (train_ratio, val_ratio)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5

# Loss weights (Task 2 specific)
CLS_W = 0.5  # Classification loss weight
KL_W = 1e-4  # KL divergence weight (for VAE models)

# Random seeds for different runs (can be model-specific)
DEFAULT_SEEDS = [50, 60]

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER & SCHEDULER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
OPTIMIZER_TYPE = "adam"
SCHEDULER_TYPE = "reduce_lr_on_plateau"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MODE = "min"

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════
# Common architecture parameters used across most models
HIDDEN_DIM = 4
NUM_LAYERS = 1
CUTOFF_RADIUS = 5.0

# ADiT-specific parameters
ADIT_NUM_HEADS = 2
ADIT_DROPOUT = 0.1
ADIT_CHUNK_SIZE = 128

# CDVAE-specific parameters
CDVAE_LATENT_DIM = 4

# FlowLLM-specific parameters
FLOWLLM_ATOM_EMB_DIM = 4
FLOWLLM_LLM_MODEL_NAME = "prajjwal1/bert-tiny"

# FlowMM-specific parameters
FLOWMM_ATOM_EMB_DIM = 4
FLOWMM_R_EMB_DIM = 4
FLOWMM_TIME_EMB_DIM = 4

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_BASE_DIR = "results/task_2"

# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# Device will be set automatically: "cuda" if available, else "cpu"
# This is handled in training scripts

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL-SPECIFIC CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Model-specific seed lists (override DEFAULT_SEEDS if needed)
MODEL_SEEDS = {
    "adit": [50, 60],
    "cdvae": [50, 60],
    "diffcsp": [50, 60],
    "flowllm": [42, 50, 60],
    "flowmm": [42, 50, 60],
    "mattergen": [42, 50, 60],
}

# Model-specific batch sizes (override BATCH_SIZE if needed)
MODEL_BATCH_SIZES = {
    "adit": 1,
    "cdvae": 1,
    "diffcsp": 1,
    "flowllm": 1,
    "flowmm": 1,
    "mattergen": 1,
}

# Model-specific gradient clipping (Task 2 specific)
MODEL_GRAD_CLIP = {
    "adit": 1.0,
    "cdvae": 10.0,
    "diffcsp": 1.0,
    "flowllm": 1.0,
    "flowmm": 0.1,
    "mattergen": 1.0,
}

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Task2TrainingConfig:
    """Configuration for Task 2 training."""

    # Data settings
    # data_root is resolved dynamically relative to current working directory
    data_root: str = field(default_factory=get_data_root)
    subset_ratio: float = SUBSET_RATIO
    train_val_split: tuple = field(default_factory=lambda: TRAIN_VAL_SPLIT)

    # Training hyperparameters
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    num_epochs: int = NUM_EPOCHS
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))

    # Loss weights (Task 2 specific)
    cls_weight: float = CLS_W
    kl_weight: float = KL_W

    # Optimizer settings
    optimizer_type: str = OPTIMIZER_TYPE
    scheduler_type: str = SCHEDULER_TYPE
    scheduler_factor: float = SCHEDULER_FACTOR
    scheduler_patience: int = SCHEDULER_PATIENCE
    scheduler_mode: str = SCHEDULER_MODE

    # Model architecture
    hidden_dim: int = HIDDEN_DIM
    num_layers: int = NUM_LAYERS
    cutoff_radius: float = CUTOFF_RADIUS

    # ADiT-specific
    adit_num_heads: int = ADIT_NUM_HEADS
    adit_dropout: float = ADIT_DROPOUT
    adit_chunk_size: int = ADIT_CHUNK_SIZE

    # CDVAE-specific
    cdvae_latent_dim: int = CDVAE_LATENT_DIM

    # FlowLLM-specific
    flowllm_atom_emb_dim: int = FLOWLLM_ATOM_EMB_DIM
    flowllm_llm_model_name: str = FLOWLLM_LLM_MODEL_NAME

    # FlowMM-specific
    flowmm_atom_emb_dim: int = FLOWMM_ATOM_EMB_DIM
    flowmm_r_emb_dim: int = FLOWMM_R_EMB_DIM
    flowmm_time_emb_dim: int = FLOWMM_TIME_EMB_DIM

    # Output paths
    # results_base_dir is also relative to current working directory
    results_base_dir: str = field(
        default_factory=lambda: str(Path.cwd() / RESULTS_BASE_DIR)
    )

    def get_seeds_for_model(self, model_name: str) -> List[int]:
        """Get seeds for a specific model."""
        return MODEL_SEEDS.get(model_name.lower(), self.seeds)

    def get_batch_size_for_model(self, model_name: str) -> int:
        """Get batch size for a specific model."""
        return MODEL_BATCH_SIZES.get(model_name.lower(), self.batch_size)

    def get_grad_clip_for_model(self, model_name: str) -> float:
        """Get gradient clipping value for a specific model."""
        return MODEL_GRAD_CLIP.get(model_name.lower(), 1.0)

    def get_output_dir(self, model_name: str, seed: int) -> str:
        """Get output directory for a model and seed."""
        return str(Path(self.results_base_dir) / model_name.lower() / str(seed))

    @classmethod
    def default(cls) -> "Task2TrainingConfig":
        """Create config with default values."""
        return cls()
