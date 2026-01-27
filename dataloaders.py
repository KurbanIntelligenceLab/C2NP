import logging
import os
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import spglib
import torch
from ase.data import atomic_numbers
from ase.io import read as ase_read
from torch.utils.data import random_split
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.nn import radius_graph

from create_c2np.config import DEFAULT_QUATERNION_CONFIG, UNIT_CELLS_SUBDIR, QUATERNIONS_SUBDIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add safe globals for torch serialization
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


class C2NPDataloader(InMemoryDataset):
    """
    A PyTorch Geometric dataset for crystal structure prediction tasks.

    This dataset loads crystal structures from the unit_cells and quaternions directories
    and provides train/ID test/OOD test splits based on R values (cutoff radii).

    Directory structure expected:
    - root/
      - unit_cells/    # CIF files for unit cells
      - quaternions/   # XYZ files organized by material and R values
      - all_splits.pt  # Processed data (created automatically)
      - metadata.pkl   # Dataset metadata (created automatically)

    The dataset returns Data objects with:
    - Node features: atomic numbers and positions
    - Graph structure: edges based on cutoff radius
    - Targets: lattice parameters and space group numbers
    - Unit cell information for reference
    """

    SPLITS = ["train", "id_test", "ood_test"]

    # R value splits from config
    R_SPLITS = DEFAULT_QUATERNION_CONFIG["r_splits"]

    @property
    def raw_file_names(self):
        return []  # we scan quaternions/ ourselves

    @property
    def processed_file_names(self):
        return ["all_splits.pt", "metadata.pkl"]

    @property
    def raw_dir(self):
        # Override to use our custom directory structure
        return str(self.root_path)

    @property
    def processed_dir(self):
        # Override to use our custom directory structure
        return str(self.root_path)

    def download(self):
        pass  # no download needed

    def _process(self):
        """Override the parent's _process method to prevent raw/processed dir creation."""
        # Skip the parent's _process which creates raw/processed directories
        # We handle our own processing in the __init__ method
        pass

    def __init__(
        self, root: str, transform=None, pre_transform=None, num_workers: int = None
    ):
        self.num_workers = num_workers or cpu_count()
        self.root_path = Path(root)
        super().__init__(root, transform, pre_transform)

        # Check if processed data exists and is valid
        if not self._is_processed_data_valid():
            logger.info("Processed data not found or invalid, processing...")
            self.process()
        else:
            logger.info("Loading existing processed data...")

        # Load data with error handling
        try:
            self._all = self._load_processed_data()
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            logger.info("Reprocessing data...")
            self.process()
            self._all = self._load_processed_data()

    def _is_processed_data_valid(self):
        """Check if processed data files exist and are valid."""
        try:
            # Check if main data file exists
            if not os.path.isfile(self.processed_paths[0]):
                return False

            # Check if metadata file exists
            if not os.path.isfile(self.processed_paths[1]):
                return False

            # Try to load metadata to check integrity
            with open(self.processed_paths[1], "rb") as f:
                metadata = pickle.load(f)

            # Validate metadata structure
            required_keys = ["version", "num_samples", "splits_info"]
            if not all(key in metadata for key in required_keys):
                return False

            return True
        except Exception as e:
            logger.warning(f"Data validation failed: {e}")
            return False

    def _load_processed_data(self):
        """Load processed data with error handling."""
        try:
            # Try loading with weights_only first (safer)
            data = torch.load(self.processed_paths[0], weights_only=True)
            logger.info("Successfully loaded data with weights_only=True")
            return data
        except Exception as e1:
            logger.warning(f"Failed to load with weights_only=True: {e1}")
            try:
                # Fallback to regular loading
                data = torch.load(self.processed_paths[0], map_location="cpu")
                logger.info("Successfully loaded data with regular torch.load")
                return data
            except Exception as e2:
                logger.error(f"Failed to load data with regular torch.load: {e2}")
                raise RuntimeError(f"Could not load processed data: {e1}, {e2}")

    def _save_metadata(self, splits_data):
        """Save metadata about the processed dataset."""
        metadata = {
            "version": "1.0",
            "num_samples": sum(
                len(split[0]) if split is not None else 0
                for split in splits_data.values()
            ),
            "splits_info": {
                split: len(data[0]) if data is not None else 0
                for split, data in splits_data.items()
            },
            "r_splits": self.R_SPLITS,
            "processed_timestamp": str(Path(self.processed_paths[0]).stat().st_mtime),
        }

        with open(self.processed_paths[1], "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata: {metadata}")

    def process_single_file(self, args):
        """Process a single XYZ file with comprehensive error handling."""
        xyz_path, unit_cell_path, cutoff = args

        try:
            # --- read unit cell ---
            if not os.path.exists(unit_cell_path):
                logger.warning(f"Unit cell file not found: {unit_cell_path}")
                return None

            cell = ase_read(unit_cell_path)
            if cell is None or len(cell) == 0:
                logger.warning(f"Empty or invalid unit cell: {unit_cell_path}")
                return None

            cell_pos = torch.from_numpy(
                np.array(cell.get_positions(), dtype=np.float32)
            )
            cell_cell = torch.from_numpy(np.array(cell.get_cell(), dtype=np.float32))
            cell_z = torch.from_numpy(
                np.array(cell.get_atomic_numbers(), dtype=np.int64)
            )

            # --- read nanoparticle XYZ ---
            if not os.path.exists(xyz_path):
                logger.warning(f"XYZ file not found: {xyz_path}")
                return None

            symbols, coords = [], []
            try:
                with open(xyz_path, "r") as f:
                    lines = f.readlines()
                    if len(lines) < 3:
                        logger.warning(f"Invalid XYZ file (too few lines): {xyz_path}")
                        return None

                    for line in lines[
                        2:
                    ]:  # Skip first two lines (atom count and comment)
                        parts = line.strip().split()
                        if len(parts) < 4:
                            continue
                        try:
                            symbols.append(parts[0])
                            coords.append([float(x) for x in parts[1:4]])
                        except (ValueError, IndexError) as e:
                            logger.warning(
                                f"Error parsing XYZ line: {line.strip()}, error: {e}"
                            )
                            continue

            except Exception as e:
                logger.warning(f"Error reading XYZ file {xyz_path}: {e}")
                return None

            if len(symbols) == 0:
                logger.warning(f"No valid atoms found in XYZ file: {xyz_path}")
                return None

            pos = torch.tensor(coords, dtype=torch.float32)
            z = torch.tensor(
                [atomic_numbers.get(s, 0) for s in symbols], dtype=torch.long
            )

            # Check for unknown elements
            if (z == 0).any():
                unknown_elements = [s for s in symbols if atomic_numbers.get(s, 0) == 0]
                logger.warning(f"Unknown elements in {xyz_path}: {unknown_elements}")
                return None

            # --- build graph and lattice info ---
            try:
                edge_index = radius_graph(pos, r=cutoff)
            except Exception as e:
                logger.warning(f"Error building radius graph for {xyz_path}: {e}")
                return None

            try:
                a, b, c, alpha, beta, gamma = cell.get_cell_lengths_and_angles()
                lattice_vec = torch.tensor(
                    [a, b, c, alpha, beta, gamma], dtype=torch.float32
                )
            except Exception as e:
                logger.warning(
                    f"Error getting cell parameters for {unit_cell_path}: {e}"
                )
                return None

            # --- spacegroup via spglib ---
            try:
                dataset = spglib.get_symmetry_dataset(
                    (
                        cell.get_cell().array,
                        cell.get_scaled_positions(),
                        cell.get_atomic_numbers(),
                    )
                )
                if dataset is None:
                    logger.warning(
                        f"Could not determine space group for {unit_cell_path}"
                    )
                    sg_number = 1  # Default to P1
                else:
                    sg_number = int(dataset.number)
            except Exception as e:
                logger.warning(
                    f"Error determining space group for {unit_cell_path}: {e}"
                )
                sg_number = 1  # Default to P1

            data = Data(
                z=z,
                x=z.view(-1, 1).float(),
                pos=pos,
                edge_index=edge_index,
                lattice=lattice_vec,  # target ℓ
                spacegroup=torch.tensor([sg_number]),  # target g
                cell_pos=cell_pos,
                cell_cell=cell_cell,
                cell_z=cell_z,
                radius=torch.tensor([cutoff], dtype=torch.float32),
            )

            if self.pre_transform:
                try:
                    data = self.pre_transform(data)
                except Exception as e:
                    logger.warning(f"Error applying pre_transform to {xyz_path}: {e}")
                    return None

            return data

        except Exception as e:
            logger.warning(f"Unexpected error processing {xyz_path}: {e}")
            return None

    def process(self):
        """Process the dataset with improved error handling and validation."""
        logger.info("Starting dataset processing...")

        unit_cells_dir = self.root_path / UNIT_CELLS_SUBDIR
        quat_dir = self.root_path / QUATERNIONS_SUBDIR

        # Validate input directories
        if not unit_cells_dir.exists():
            raise FileNotFoundError(f"Unit cells directory not found: {unit_cells_dir}")
        if not quat_dir.exists():
            raise FileNotFoundError(f"Quaternions directory not found: {quat_dir}")

        # map material name → its CIF path
        unit_cells = {}
        for f in unit_cells_dir.glob("*.cif"):
            material_name = f.stem.lower()
            unit_cells[material_name] = str(f)

        logger.info(f"Found {len(unit_cells)} CIF files")

        # gather args for each split
        train_args, id_args, ood_args = [], [], []
        processed_materials = 0

        for mat_dir in quat_dir.iterdir():
            if not mat_dir.is_dir():
                continue

            mat_low = mat_dir.name.lower()
            unit_cell = unit_cells.get(mat_low)
            if not unit_cell:
                logger.warning(f"No CIF found for material: {mat_low}")
                continue

            # each R-folder: R6, R7, ..., R30
            for rad_folder in sorted(mat_dir.iterdir()):
                if not rad_folder.is_dir() or not rad_folder.name.startswith("R"):
                    continue

                try:
                    cutoff = float(rad_folder.name[1:])
                except ValueError:
                    logger.warning(f"Invalid R folder name: {rad_folder.name}")
                    continue

                xyz_dir = rad_folder / "xyz"
                if not xyz_dir.exists():
                    continue

                xyz_files = sorted(list(xyz_dir.glob("*.xyz")))
                for xyz_file in xyz_files:
                    args = (str(xyz_file), unit_cell, cutoff)

                    # Determine split based on R value
                    if cutoff in self.R_SPLITS["ID"]:
                        id_args.append(args)
                    elif cutoff in self.R_SPLITS["OOD"]:
                        ood_args.append(args)
                    else:
                        train_args.append(args)

            processed_materials += 1
            if processed_materials % 10 == 0:
                logger.info(f"Processed {processed_materials} materials...")

        logger.info(
            f"Found {len(train_args)} train, {len(id_args)} ID test, {len(ood_args)} OOD test samples"
        )

        # Process all splits using a single pool
        splits_data = {}
        with Pool(self.num_workers) as pool:
            for name, args_list in [
                ("train", train_args),
                ("id_test", id_args),
                ("ood_test", ood_args),
            ]:
                if len(args_list) == 0:
                    logger.warning(f"No data found for split: {name}")
                    splits_data[name] = None
                    continue

                logger.info(f"Processing {len(args_list)} samples for {name} split...")
                data_list = []

                try:
                    for i, data in enumerate(
                        pool.imap(self.process_single_file, args_list)
                    ):
                        if data is not None:
                            data_list.append(data)
                        if (i + 1) % 1000 == 0:
                            logger.info(
                                f"Processed {i + 1}/{len(args_list)} samples for {name}"
                            )

                    if data_list:
                        splits_data[name] = self.collate(data_list)
                        logger.info(
                            f"Successfully processed {len(data_list)} samples for {name}"
                        )
                    else:
                        logger.warning(f"No valid data processed for {name}")
                        splits_data[name] = None

                except Exception as e:
                    logger.error(f"Error processing {name} split: {e}")
                    splits_data[name] = None

        # Save data with error handling - save directly in root directory
        try:
            # Save main data file
            torch.save(splits_data, self.processed_paths[0])
            logger.info(f"Saved processed data to {self.processed_paths[0]}")

            # Save metadata
            self._save_metadata(splits_data)

        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise

    def get_split(self, split: str, subset_ratio: float = 1.0) -> InMemoryDataset:
        split = split.lower()
        assert split in self.SPLITS, f"Unknown split '{split}'"
        if self._all[split] is None:
            raise ValueError(
                f"Split '{split}' has no data. This split was empty during processing."
            )
        data, slices = self._all[split]

        # If subset_ratio < 1.0, take only a subset of the data
        if subset_ratio < 1.0:
            total_samples = len(data)
            subset_size = max(1, int(total_samples * subset_ratio))

            # Create a subset by taking the first subset_size samples
            subset_data = data[:subset_size]

            # Recalculate slices for the subset
            subset_slices = {}
            for key, value in slices.items():
                if isinstance(value, torch.Tensor):
                    # For tensor slices, take the first subset_size+1 elements
                    subset_slices[key] = value[: subset_size + 1]
                else:
                    subset_slices[key] = value

            ds = InMemoryDataset(self.root, transform=self.transform)
            ds.data, ds.slices = subset_data, subset_slices
        else:
            ds = InMemoryDataset(self.root, transform=self.transform)
            ds.data, ds.slices = data, slices

        return ds

    def random_train_splits(
        self, train_ratio=0.8, val_ratio=0.2, seed=42, subset_ratio=1.0
    ):
        """
        Further split the 'train' split into train/val for training purposes.
        Returns: (train_ds, val_ds)
        Note: id_test and ood_test are already separate splits.
        """
        train_split = self.get_split("train", subset_ratio=subset_ratio)
        total = len(train_split)
        n_train = int(train_ratio * total)
        n_val = total - n_train
        generator = torch.Generator().manual_seed(seed)
        return random_split(train_split, [n_train, n_val], generator=generator)

    def get_dataset_info(self):
        """Get information about the dataset."""
        info = {
            "total_samples": 0,
            "splits": {},
            "r_splits": self.R_SPLITS,
            "processed_files": {
                "data_file": str(self.processed_paths[0]),
                "metadata_file": str(self.processed_paths[1]),
            },
        }

        for split in self.SPLITS:
            if self._all[split] is not None:
                split_data, _ = self._all[split]
                num_samples = len(split_data)
                info["splits"][split] = num_samples
                info["total_samples"] += num_samples
            else:
                info["splits"][split] = 0

        return info

    def print_dataset_info(self):
        """Print detailed information about the dataset."""
        info = self.get_dataset_info()
        print("=" * 10)
        print("C2NP Dataset Information")
        print("=" * 10)
        print(f"Total samples: {info['total_samples']}")
        print("\nSplit breakdown:")
        for split, count in info["splits"].items():
            print(f"  {split}: {count} samples")
        print("\nR value splits:")
        print(f"  ID (In-distribution): {info['r_splits']['ID']}")
        print(f"  OOD (Out-of-distribution): {info['r_splits']['OOD']}")
        print("\nProcessed files:")
        print(f"  Data: {info['processed_files']['data_file']}")
        print(f"  Metadata: {info['processed_files']['metadata_file']}")
        print("=" * 10)