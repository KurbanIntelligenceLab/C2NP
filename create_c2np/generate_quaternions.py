"""
Quaternion rotation generation for C2NP dataset.

All configuration is imported from create_c2np.config.
"""

import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from create_c2np.config import (
    QuaternionConfig,
    DEFAULT_QUATERNION_CONFIG,
    OUTPUT_DIR as DEFAULT_OUTPUT_DIR,
    RAW_DATA_DIR,
    QUATERNIONS_SUBDIR,
)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE FOR WORKER PROCESSES
# ═══════════════════════════════════════════════════════════════════════════════
# These globals are necessary for ProcessPoolExecutor workers
_WORKER_TRAIN_QS: Optional[np.ndarray] = None
_WORKER_OUTPUT_DIR: Optional[Path] = None
_WORKER_CONFIG: Optional["QuaternionConfig"] = None


def _init_worker(train_qs: np.ndarray, output_dir: Path, config_dict: dict):
    """Initialize worker process with shared state."""
    global _WORKER_TRAIN_QS, _WORKER_OUTPUT_DIR, _WORKER_CONFIG
    _WORKER_TRAIN_QS = train_qs
    _WORKER_OUTPUT_DIR = output_dir
    _WORKER_CONFIG = QuaternionConfig.from_dict(config_dict)


# ═══════════════════════════════════════════════════════════════════════════════
# QUATERNION GENERATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class QuaternionGenerator:
    """
    Generator for quaternion rotations of molecular structures.

    This class handles the generation of rotated structures for the C2NP dataset,
    with separate rotation grids for Train, ID, and OOD splits.

    Usage:
        generator = QuaternionGenerator(
            xyz_dir=Path("materials"),
            output_dir=Path(f"c2np/{QUATERNIONS_SUBDIR}")
        )
        generator.generate()
    """

    def __init__(
        self,
        xyz_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        config: Optional[QuaternionConfig] = None,
    ):
        """
        Initialize the quaternion generator.

        Args:
            xyz_dir: Directory containing input XYZ files
            output_dir: Directory for output rotated structures
            config: Configuration object (uses defaults if not provided)
        """
        self.xyz_dir = xyz_dir or (RAW_DATA_DIR / "materials")
        self.output_dir = output_dir or (DEFAULT_OUTPUT_DIR / QUATERNIONS_SUBDIR)
        self.config = config or QuaternionConfig.default()

        # Will be populated during generation
        self.train_ref_qs: Optional[np.ndarray] = None
        self.counts = {"train": 0, "ID": 0, "OOD": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # STATIC METHODS (Pure functions, no state)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def num_for_spacing(angle_deg: float) -> int:
        """
        Compute number of quaternions for a given angular separation.

        Uses spherical cap coverage estimate: N(θ) ≈ 2 / (1 - cos(θ))
        For 15°, 12°, 9° this gives approximately 59, 92, 163 quaternions.
        """
        theta = np.radians(angle_deg)
        solid_angle = 2 * np.pi * (1 - np.cos(theta))
        return max(1, int(np.ceil(4 * np.pi / solid_angle)))

    @staticmethod
    def generate_uniform_rotations(
        angle_sep_deg: float, seed: int, n_quats: int
    ) -> np.ndarray:
        """
        Greedy sampler on SO(3) with minimum angular separation.

        Args:
            angle_sep_deg: Minimum angular separation in degrees
            seed: Random seed for reproducibility
            n_quats: Number of quaternions to generate

        Returns:
            Array of quaternions in xyzw format
        """
        rng = np.random.default_rng(seed)
        cos_cap = np.cos(np.radians(angle_sep_deg) / 2)
        quats = []
        trials = 200_000

        while len(quats) < n_quats and trials:
            q = R.random(random_state=rng).as_quat()
            # Canonical form: w >= 0
            if q[3] < 0:
                q = -q

            if not quats:
                quats.append(q)
            else:
                Q = np.vstack(quats)
                if np.max(np.abs(Q @ q)) <= cos_cap:
                    quats.append(q)
            trials -= 1

        if len(quats) < n_quats:
            raise RuntimeError(
                f"Only {len(quats)} of {n_quats} rotations placed for {angle_sep_deg}°."
            )
        return np.asarray(quats, dtype=np.float64)

    @staticmethod
    def sample_with_exclusion(
        angle_sep_deg: float,
        seed: int,
        n_quats: int,
        exclude_Q: Optional[np.ndarray] = None,
        margin_deg: float = 0.0,
        left_mul: Optional[R] = None,
    ) -> np.ndarray:
        """
        Greedy sampler with exclusion margin from reference set.

        Args:
            angle_sep_deg: Internal minimum separation between samples
            seed: Random seed
            n_quats: Number of quaternions to generate
            exclude_Q: Reference quaternions to exclude from
            margin_deg: Minimum geodesic distance from exclude_Q
            left_mul: Optional rotation to apply (offset)

        Returns:
            Array of effective quaternions (post left_mul) in xyzw format
        """
        rng = np.random.default_rng(seed)
        cos_internal = np.cos(np.radians(angle_sep_deg) / 2)
        cos_excl = (
            np.cos(np.radians(margin_deg) / 2)
            if (exclude_Q is not None and margin_deg > 0)
            else None
        )

        eff_quats = []
        trials = 300_000

        while len(eff_quats) < n_quats and trials:
            q = R.random(random_state=rng).as_quat()
            if q[3] < 0:
                q = -q

            # Apply left multiplication if requested
            if left_mul is not None:
                q_eff = (left_mul * R.from_quat(q)).as_quat()
                if q_eff[3] < 0:
                    q_eff = -q_eff
            else:
                q_eff = q

            # Check internal spacing
            if eff_quats:
                Qeff = np.vstack(eff_quats)
                if np.max(np.abs(Qeff @ q_eff)) > cos_internal:
                    trials -= 1
                    continue

            # Check exclusion margin
            if cos_excl is not None:
                if np.max(np.abs(exclude_Q @ q_eff)) > cos_excl:
                    trials -= 1
                    continue

            eff_quats.append(q_eff)
            trials -= 1

        if len(eff_quats) < n_quats:
            raise RuntimeError(
                f"Only {len(eff_quats)} of {n_quats} placed "
                f"(sep {angle_sep_deg}°, margin {margin_deg}°)."
            )
        return np.asarray(eff_quats, dtype=np.float64)

    @staticmethod
    def unique_rotations(
        coords: np.ndarray, quats: np.ndarray, tol: float = DEFAULT_QUATERNION_CONFIG["coord_tolerance"]
    ) -> np.ndarray:
        """
        Deduplicate rotations that produce identical coordinates.

        Args:
            coords: Original coordinates
            quats: Quaternions to deduplicate
            tol: Tolerance for coordinate comparison

        Returns:
            Array of unique quaternions
        """
        seen = set()
        unique = []
        for q in quats:
            rotated = R.from_quat(q).apply(coords)
            key = tuple(map(tuple, np.round(rotated / tol).astype(int)))
            if key not in seen:
                seen.add(key)
                unique.append(q)
        return np.array(unique, dtype=np.float64)

    @staticmethod
    def save_xyz_file(
        coords: np.ndarray,
        elements: tuple,
        num_atoms: int,
        r_val: int,
        path: Path,
        rotation_angles: Optional[tuple] = None,
    ):
        """
        Write an XYZ file.

        Args:
            coords: Atomic coordinates
            elements: Element symbols
            num_atoms: Number of atoms
            r_val: R value for this structure
            path: Output file path
            rotation_angles: Optional (x, y, z) Euler angles in degrees
        """
        with open(path, "w") as f:
            f.write(f"{num_atoms}\n")
            if rotation_angles is None:
                f.write(f"Original structure at R{r_val}\n")
            else:
                x, y, z = rotation_angles
                f.write(f"Rotated structure: x={x:.1f}°, y={y:.1f}°, z={z:.1f}° at R{r_val}\n")
            for el, (X, Y, Z) in zip(elements, coords):
                f.write(f"{el:2s} {X:15.8f} {Y:15.8f} {Z:15.8f}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # INSTANCE METHODS
    # ─────────────────────────────────────────────────────────────────────────
    def calculate_split_sizes(self) -> dict:
        """Compute expected file counts per split."""
        counts = {"train": 0, "ID": 0, "OOD": 0}
        for r in self.config.r_values:
            total = 1  # original
            for spacing in self.config.angle_map[r]:
                total += self.num_for_spacing(spacing)
            split = self.config.get_split(r)
            counts[split] += total
        return counts

    def build_train_grid(self) -> np.ndarray:
        """Build the reference training quaternion grid."""
        n_train = self.num_for_spacing(self.config.theta_train)
        return self.generate_uniform_rotations(
            self.config.theta_train, self.config.train_seed, n_train
        )

    def generate(self) -> dict:
        """
        Generate all quaternion rotations.

        Returns:
            Dictionary with counts per split
        """
        # Build training reference grid
        self.train_ref_qs = self.build_train_grid()

        # Report expected sizes
        expected = self.calculate_split_sizes()
        total_expected = sum(expected.values())
        print("=== EXPECTED SPLIT SIZES ===")
        print(f"  Train : {expected['train']} files")
        print(f"  ID    : {expected['ID']} files")
        print(f"  OOD   : {expected['OOD']} files")
        print(f"  Total : {total_expected} files\n")

        # Get XYZ files
        xyz_files = list(self.xyz_dir.glob("*.xyz"))
        print(f"=== GENERATING ROTATIONS ({len(xyz_files)} files) ===")

        # Reset counts
        self.counts = {"train": 0, "ID": 0, "OOD": 0}

        # Process files in parallel with progress bar
        with ProcessPoolExecutor(
            max_workers=self.config.max_workers,
            initializer=_init_worker,
            initargs=(self.train_ref_qs, self.output_dir, self.config.to_dict()),
        ) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_xyz_file_worker, xyz_file): xyz_file
                for xyz_file in xyz_files
            }
            
            # Process with progress bar
            with tqdm(total=len(xyz_files), desc="Generating rotations", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    for k in self.counts:
                        self.counts[k] += result[k]
                    pbar.update(1)

        # Report actual counts
        total_actual = sum(self.counts.values())
        print("\n=== ACTUAL SPLIT COUNTS ===")
        print(f"  Train : {self.counts['train']} files")
        print(f"  ID    : {self.counts['ID']} files")
        print(f"  OOD   : {self.counts['OOD']} files")
        print(f"  Total : {total_actual} files")

        return self.counts


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER FUNCTION (Must be at module level for pickling)
# ═══════════════════════════════════════════════════════════════════════════════
def _process_xyz_file_worker(xyz_path: Path) -> dict:
    """
    Worker function to process a single XYZ file.

    This function runs in worker processes and uses global state
    set by _init_worker.
    """
    global _WORKER_TRAIN_QS, _WORKER_OUTPUT_DIR, _WORKER_CONFIG

    counts = {"train": 0, "ID": 0, "OOD": 0}
    config = _WORKER_CONFIG

    # Parse filename
    match = re.match(r"(.+)_R(\d+)\.xyz", xyz_path.name)
    if not match:
        print(f"[WARN] Skipping file with unexpected name: {xyz_path.name}")
        return counts

    material = match.group(1)
    r = int(match.group(2))

    if r not in config.r_values:
        print(f"[WARN] R value {r} not in allowed R_VALUES for {xyz_path.name}")
        return counts

    # Load atoms - handle encoding issues gracefully
    try:
        # Try UTF-8 first
        with xyz_path.open(encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except UnicodeDecodeError:
        # Fall back to latin-1 which can decode any byte
        with xyz_path.open(encoding='latin-1', errors='replace') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    
    if not lines:
        print(f"[WARN] Empty or invalid XYZ file: {xyz_path.name}")
        return counts
    
    num_atoms = int(lines[0])
    atom_lines = lines[2 : 2 + num_atoms]
    elements, coords = zip(
        *[(ln.split()[0], list(map(float, ln.split()[1:4]))) for ln in atom_lines]
    )
    coords = np.array(coords)

    # Prepare output directory
    out_dir = _WORKER_OUTPUT_DIR / material / f"R{r}" / "xyz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    QuaternionGenerator.save_xyz_file(
        coords, elements, num_atoms, r, out_dir / "rot_0.xyz", None
    )

    split = config.get_split(r)
    counts[split] += 1

    # Generate rotations
    all_quats = []
    for spacing in config.angle_map[r]:
        n_q = QuaternionGenerator.num_for_spacing(spacing)

        if split == "train":
            qs_eff = _WORKER_TRAIN_QS
            if qs_eff.shape[0] != n_q:
                qs_eff = QuaternionGenerator.generate_uniform_rotations(
                    spacing, config.train_seed, n_q
                )
        else:
            seed = int(1000 + r + spacing)
            margin = config.margins[split]
            left_mul = config.id_offset if split == "ID" else config.ood_offset
            qs_eff = QuaternionGenerator.sample_with_exclusion(
                spacing,
                seed,
                n_q,
                exclude_Q=_WORKER_TRAIN_QS,
                margin_deg=margin,
                left_mul=left_mul,
            )
        all_quats.append(qs_eff)

    if not all_quats:
        return counts

    all_quats = np.vstack(all_quats)

    # Deduplicate and save
    unique_qs = QuaternionGenerator.unique_rotations(
        coords, all_quats, config.coord_tolerance
    )

    for idx, q in enumerate(unique_qs, start=1):
        rot_coords = R.from_quat(q).apply(coords)
        angles = R.from_quat(q).as_euler("xyz", degrees=True)
        QuaternionGenerator.save_xyz_file(
            rot_coords, elements, num_atoms, r, out_dir / f"rot_{idx}.xyz", tuple(angles)
        )
        counts[split] += 1

    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
# These module-level variables maintain backward compatibility with create_c2np.py
xyz_dir = RAW_DATA_DIR / "materials"
base_output_dir = DEFAULT_OUTPUT_DIR / QUATERNIONS_SUBDIR


def main():
    """Main entry point for standalone execution."""
    generator = QuaternionGenerator(xyz_dir=xyz_dir, output_dir=base_output_dir)
    generator.generate()


if __name__ == "__main__":
    main()
