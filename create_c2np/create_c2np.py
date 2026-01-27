#!/usr/bin/env python3
"""
Create c2np dataset from raw data.

Takes raw_data/ (directory or zip file) as input and outputs c2np/ with:
- Directory names are configurable via create_c2np.config

Usage:
    python create_c2np.py --raw-data path/to/raw_data --output c2np
    python create_c2np.py --raw-data c2np_raw.zip --output c2np

Final output structure:
    c2np/
    ├── <QUATERNIONS_SUBDIR>/
    │   └── {Material}/
    │       └── R{x}/
    │           └── xyz/
    │               └── rot_*.xyz
    └── <UNIT_CELLS_SUBDIR>/
        └── {Material}.cif

Note: Directory names are defined in create_c2np.config (QUATERNIONS_SUBDIR, UNIT_CELLS_SUBDIR)
"""

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from glob import glob
from pathlib import Path

from create_c2np.generate_quaternions import QuaternionGenerator
from create_c2np.config import QUATERNIONS_SUBDIR, UNIT_CELLS_SUBDIR

# Project root for reference
PROJECT_ROOT = Path(__file__).parent.parent


def extract_cifs(raw_data_dir: Path, output_dir: Path) -> int:
    """Extract all CIF files from raw_data to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    
    # Check if there's a cifs/ subdirectory
    cifs_subdir = raw_data_dir / "cifs"
    search_dir = cifs_subdir if cifs_subdir.exists() else raw_data_dir
    
    for root, _, files in os.walk(search_dir):
        for f in files:
            if f.lower().endswith(".cif"):
                shutil.copy2(os.path.join(root, f), output_dir / f)
                count += 1
    return count


def extract_xyz_files(raw_data_dir: Path, output_dir: Path) -> int:
    """Extract XYZ files from raw_data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    
    # Check if there's a materials/ subdirectory
    materials_subdir = raw_data_dir / "materials"
    search_dir = materials_subdir if materials_subdir.exists() else raw_data_dir
    
    # Debug: print what we're searching
    if not search_dir.exists():
        print(f"  [DEBUG] Search directory does not exist: {search_dir}")
        return 0
    
    # Use os.walk to recursively find all XYZ files
    xyz_files_found = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file.lower().endswith(".xyz"):
                xyz_path = Path(root) / file
                xyz_files_found.append(xyz_path)
    
    # Copy all found XYZ files
    for xyz_path in xyz_files_found:
        # Use the original filename, but ensure uniqueness if needed
        dest_path = output_dir / xyz_path.name
        # If file already exists, use full relative path to preserve structure
        if dest_path.exists() and xyz_path != dest_path:
            # Create a unique name using relative path
            rel_path = xyz_path.relative_to(search_dir)
            dest_path = output_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(xyz_path, dest_path)
        count += 1
    
    if count == 0:
        print(f"  [DEBUG] No XYZ files found. Searched in: {search_dir}")
        print(f"  [DEBUG] Directory contents: {list(search_dir.iterdir())[:10] if search_dir.exists() else 'N/A'}")
    
    return count


def extract_zip_if_needed(raw_data_path: Path) -> tuple[Path, bool]:
    """
    Extract zip file if needed, return path to data directory and cleanup flag.
    
    Returns:
        Tuple of (data_path, needs_cleanup)
    """
    if raw_data_path.suffix.lower() in ['.zip', '.zipx']:
        print(f"  Detected zip file: {raw_data_path.name}")
        print("  Extracting to temporary directory...")
        temp_extract = tempfile.mkdtemp(prefix="c2np_raw_")
        with zipfile.ZipFile(raw_data_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        return Path(temp_extract), True
    return raw_data_path, False


def create_c2np(raw_data_dir: str = "c2np_raw", output_dir: str = "c2np"):
    """
    Create c2np dataset from raw data (directory or zip file).

    Args:
        raw_data_dir: Path to raw data directory or zip file
        output_dir: Path to output c2np directory
    """
    print("\n" + "=" * 60)
    print("Creating c2np dataset from raw data")
    print("=" * 60)
    print(f"  Working directory: {os.getcwd()}")
    print(f"  Raw data: {raw_data_dir}")
    print(f"  Output: {output_dir}")

    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    unit_cells_dir = output_path / UNIT_CELLS_SUBDIR
    quaternions_dir = output_path / QUATERNIONS_SUBDIR

    # If path doesn't exist, check for zip version
    if not raw_path.exists():
        zip_path = Path(f"{raw_data_dir}.zip")
        if zip_path.exists():
            print(f"  Path '{raw_data_dir}' not found, using '{zip_path.name}' instead")
            raw_path = zip_path
        else:
            print(f"\n  [ERROR] Raw data path not found: {raw_data_dir}")
            print(f"  Also checked: {zip_path}")
            return False

    # Extract zip if needed
    data_path, needs_cleanup = extract_zip_if_needed(raw_path)
    
    try:
        # Clear output directory if it exists
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Extract CIFs to unit_cells directory (from config)
        print("\n" + "-" * 60)
        print("Step 1: Extracting CIF files")
        print("-" * 60)
        cif_count = extract_cifs(data_path, unit_cells_dir)
        print(f"  Extracted {cif_count} CIF files to {unit_cells_dir}")

        # Step 2: Extract XYZ files to temp directory
        print("\n" + "-" * 60)
        print("Step 2: Extracting XYZ files")
        print("-" * 60)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_materials = Path(temp_dir) / "materials"
            xyz_count = extract_xyz_files(data_path, temp_materials)
            print(f"  Extracted {xyz_count} XYZ files")

            # Step 3: Generate quaternions
            print("\n" + "-" * 60)
            print("Step 3: Generating quaternion rotations")
            print("-" * 60)
            
            if xyz_count == 0:
                print("  [WARN] No XYZ files found, skipping quaternion generation")
            else:
                # Use QuaternionGenerator class
                generator = QuaternionGenerator(
                    xyz_dir=temp_materials,
                    output_dir=quaternions_dir,
                )
                generator.generate()

        # Count generated materials
        mat_count = len([d for d in quaternions_dir.iterdir() if d.is_dir()]) if quaternions_dir.exists() else 0

        print("\n" + "=" * 60)
        print("c2np dataset created successfully!")
        print("=" * 60)
        print(f"\n  Output: {output_path.resolve()}")
        print(f"\n  {output_path}/")
        print(f"  ├── {QUATERNIONS_SUBDIR}/  ({mat_count} materials)")
        print(f"  └── {UNIT_CELLS_SUBDIR}/             ({cif_count} CIF files)")

        return True
    finally:
        # Clean up extracted zip if needed
        if needs_cleanup and data_path.exists():
            print("\n  Cleaning up temporary extraction directory...")
            shutil.rmtree(data_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create c2np dataset from raw data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m create_c2np.create_c2np
    python -m create_c2np.create_c2np --raw-data c2np_raw --output c2np
        """,
    )
    parser.add_argument(
        "--raw-data",
        "-r",
        default="c2np_raw",
        help="Path to raw data directory or zip file (default: c2np_raw)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="c2np",
        help="Path to output directory (default: c2np)",
    )

    args = parser.parse_args()
    success = create_c2np(
        raw_data_dir=args.raw_data,
        output_dir=args.output,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
