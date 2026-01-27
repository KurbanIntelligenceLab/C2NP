"""
Central configuration for C2NP dataset generation.

All settings for quaternion generation, splits, and paths are defined here.
"""

from dataclasses import dataclass, field
from pathlib import Path

from scipy.spatial.transform import Rotation as R

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════════════
RAW_DATA_DIR = Path("c2np_raw")
OUTPUT_DIR = Path("c2np")

# These will be set relative to OUTPUT_DIR
QUATERNIONS_SUBDIR = "quaternions"
UNIT_CELLS_SUBDIR = "cifs"

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION DICTIONARY
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_QUATERNION_CONFIG = {
    # R values and splits
    "r_values": list(range(6, 31)),  # R6 to R30
    "r_splits": {
        "ID": [10, 11, 17, 21, 24, 26],    # In-distribution (validation/test)
        "OOD": [6, 7, 29, 30],              # Out-of-distribution (test)
    },
    # Train = all R values not in ID or OOD
    
    # Rotational sampling density (minimum geodesic separation in degrees)
    "theta_train": 15.0,  # Training set rotation spacing
    "theta_id": 12.0,     # ID set rotation spacing (finer)
    "theta_ood": 9.0,     # OOD set rotation spacing (finest)
    
    # Fixed rotation offsets for visual separation (Euler angles in degrees)
    "id_offset_euler": [6, 8, 12],      # xyz Euler angles for ID offset
    "ood_offset_euler": [15, 25, 35],   # xyz Euler angles for OOD offset
    
    # Enforced angular margins (geodesic angle on SO(3)) from Train set
    "margins": {
        "ID": 6.0,      # degrees
        "OOD": 4.5,     # degrees
        "train": 0.0,
    },
    
    # Processing parameters
    "train_seed": 1337,           # Deterministic seed for train grid
    "max_workers": 4,             # Number of parallel workers for quaternion generation
    "coord_tolerance": 1e-6,      # Tolerance for coordinate deduplication
}

# ═══════════════════════════════════════════════════════════════════════════════
# QUATERNION CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class QuaternionConfig:
    """Configuration for quaternion generation."""

    r_values: list = field(default_factory=lambda: list(DEFAULT_QUATERNION_CONFIG["r_values"]))
    r_splits: dict = field(default_factory=lambda: dict(DEFAULT_QUATERNION_CONFIG["r_splits"]))
    theta_train: float = DEFAULT_QUATERNION_CONFIG["theta_train"]
    theta_id: float = DEFAULT_QUATERNION_CONFIG["theta_id"]
    theta_ood: float = DEFAULT_QUATERNION_CONFIG["theta_ood"]
    id_offset_euler: list = field(default_factory=lambda: list(DEFAULT_QUATERNION_CONFIG["id_offset_euler"]))
    ood_offset_euler: list = field(default_factory=lambda: list(DEFAULT_QUATERNION_CONFIG["ood_offset_euler"]))
    margins: dict = field(default_factory=lambda: dict(DEFAULT_QUATERNION_CONFIG["margins"]))
    train_seed: int = DEFAULT_QUATERNION_CONFIG["train_seed"]
    max_workers: int = DEFAULT_QUATERNION_CONFIG["max_workers"]
    coord_tolerance: float = DEFAULT_QUATERNION_CONFIG["coord_tolerance"]

    def __post_init__(self):
        """Build derived attributes after initialization."""
        self._build_angle_map()
        self._build_offsets()

    def _build_angle_map(self):
        """Build the angle map from configuration."""
        self.angle_map = {r: [self.theta_train] for r in self.r_values}
        for r in self.r_splits.get("ID", []):
            self.angle_map[r] = [self.theta_id]
        for r in self.r_splits.get("OOD", []):
            self.angle_map[r] = [self.theta_ood]

    def _build_offsets(self):
        """Build rotation offsets from Euler angles."""
        self.id_offset = R.from_euler("xyz", self.id_offset_euler, degrees=True)
        self.ood_offset = R.from_euler("xyz", self.ood_offset_euler, degrees=True)

    def get_split(self, r: int) -> str:
        """Get the split name for a given R value."""
        if r in self.r_splits.get("ID", []):
            return "ID"
        elif r in self.r_splits.get("OOD", []):
            return "OOD"
        return "train"

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "r_values": self.r_values,
            "r_splits": self.r_splits,
            "theta_train": self.theta_train,
            "theta_id": self.theta_id,
            "theta_ood": self.theta_ood,
            "id_offset_euler": self.id_offset_euler,
            "ood_offset_euler": self.ood_offset_euler,
            "margins": self.margins,
            "train_seed": self.train_seed,
            "max_workers": self.max_workers,
            "coord_tolerance": self.coord_tolerance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QuaternionConfig":
        """Create config from dictionary."""
        return cls(**d)

    @classmethod
    def default(cls) -> "QuaternionConfig":
        """Create config with default values."""
        return cls.from_dict(DEFAULT_QUATERNION_CONFIG)
