# C2NP: A Benchmark for Learning Scale-Dependent Geometric Invariances in 3D Materials Generation

This repository contains the official code and data generation pipeline for the C2NP benchmark.

## Abstract

Generative models for materials have achieved strong performance on periodic bulk crystals, yet their ability to generalize across scale transitions to finite nanostructures remains largely untested. We introduce **Crystal-to-Nanoparticle (C2NP)**, a systematic benchmark for evaluating generative models when moving between infinite crystalline unit cells and finite nanoparticles, where surface effects and size-dependent distortions dominate. 

C2NP defines two complementary tasks:
1. **Task 1**: Generating nanoparticles of specified radii from periodic unit cells, testing whether models capture surface truncation and geometric constraints
2. **Task 2**: Recovering bulk lattice parameters and space-group symmetry from finite particle configurations, assessing whether models can infer underlying crystallographic order despite surface perturbations

Using diverse materials as a structurally consistent testbed, we construct over **170,000 nanoparticle configurations** by carving particles from supercells derived from DFT-relaxed crystal unit cells, and introduce size-based splits that separate interpolation from extrapolation regimes. Experiments with state-of-the-art approaches, including diffusion, flow-matching, and variational models, show that even when losses are low, models often fail geometrically under distribution shift, yielding large lattice-recovery errors and near-zero joint accuracy on structure and symmetry.

Overall, our results suggest that current methods rely on template memorization rather than scalable physical generalization. C2NP offers a controlled, reproducible framework for diagnosing these failures, with immediate applications to **nanoparticle catalyst design**, **nanostructured hydrides for hydrogen storage**, and **materials discovery**.

## Overview

The benchmark provides:

1. **Dataset generation tools** - Generate rotated nanoparticle structures with controlled quaternion rotations from crystal unit cells
2. **Train/ID/OOD splits** - Systematic evaluation of in-distribution and out-of-distribution generalization
3. **Model implementations** - State-of-the-art 3D materials generation models for benchmarking

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Generate C2NP Dataset

Starting from raw data (directory or zip file), generate the complete dataset with quaternion rotations:

```bash
# Using a directory
python -m create_c2np.create_c2np --raw-data c2np_raw --output c2np

# Using a zip file (automatically extracted)
python -m create_c2np.create_c2np --raw-data c2np_raw.zip --output c2np
```

**Input structure** (`c2np_raw/` or `c2np_raw.zip`):
```
c2np_raw/
├── cifs/
│   └── *.cif          # Crystal unit cell files
└── materials/
    └── *.xyz          # Material XYZ files (named {Material}_R{6-30}.xyz)
```

**Note:** 
- The raw data can be provided as either a directory or a zip file. If a zip file is provided, it will be automatically extracted to a temporary directory during processing.
- If you specify a directory path that doesn't exist, the script will automatically check for a `.zip` version (e.g., `c2np_raw` → `c2np_raw.zip`).
- The script handles encoding issues in XYZ files automatically (UTF-8 with fallback to latin-1).

**Output structure** (`c2np/`):
```
c2np/
├── quaternions/                 # Configurable via QUATERNIONS_SUBDIR in config.py
│   └── {Material}/
│       └── R{x}/
│           └── xyz/
│               └── rot_*.xyz    # Original + rotated structures
└── unit_cells/                  # Configurable via UNIT_CELLS_SUBDIR in config.py
    └── {Material}.cif           # Crystal unit cell files
```

**Note:** Directory names are configurable via `create_c2np/config.py`. The default uses `quaternions/` and `unit_cells/` subdirectories.

### Command Line Options

```bash
python -m create_c2np.create_c2np --help

Options:
  -r, --raw-data PATH   Path to raw data directory or zip file (default: c2np_raw)
  -o, --output PATH     Path to output directory (default: c2np)
```

**Note:** The `--raw-data` argument accepts both directories and zip files (`.zip` or `.zipx`). Zip files are automatically extracted to a temporary directory during processing and cleaned up afterward.

## Dataset Splits

The quaternion generation automatically creates three splits based on R values:

| Split | R Values | Rotation Density | Purpose |
|-------|----------|------------------|---------|
| **Train** | R8, R9, R12-R16, R18-R20, R22, R23, R25, R27, R28 | 15° spacing | Training data |
| **ID** (In-Distribution) | R10, R11, R17, R21, R24, R26 | 12° spacing | Validation/test on seen R values |
| **OOD** (Out-of-Distribution) | R6, R7, R29, R30 | 9° spacing | Test generalization to unseen R values |

The ID and OOD splits have enforced angular margins from the training set to ensure no overlap.

**Customizing Splits:** All split definitions and rotation parameters can be customized in `create_c2np/config.py`. This includes:
- R value assignments for Train/ID/OOD splits
- Rotation spacing angles (theta_train, theta_id, theta_ood)
- Angular margins to ensure separation between splits
- Output directory names (QUATERNIONS_SUBDIR, UNIT_CELLS_SUBDIR)

## Project Structure

```
NanoScale/
├── create_c2np/
│   ├── create_c2np.py           # Main dataset generation script
│   ├── generate_quaternions.py   # Quaternion rotation generation (OOP)
│   └── config.py                # Central configuration (splits, parameters, directory names)
├── models/
│   ├── task_1/                  # Task 1 model architectures
│   └── task_2/                  # Task 2 model architectures
├── train/
│   ├── task_1/
│   │   ├── config.py            # Centralized training config for Task 1
│   │   ├── adit_train.py        # ADiT training script
│   │   ├── cdvae_train.py       # CDVAE training script
│   │   ├── diffcsp_train.py     # DiffCSP training script
│   │   ├── flowllm_train.py     # FlowLLM training script
│   │   ├── flowmm_train.py      # FlowMM training script
│   │   └── mattergen_train.py   # MatterGen training script
│   └── task_2/
│       ├── config.py            # Centralized training config for Task 2
│       ├── adit_train.py        # ADiT training script
│       ├── cdvae_train.py       # CDVAE training script
│       ├── diffcsp_train.py     # DiffCSP training script
│       ├── flowllm_train.py     # FlowLLM training script
│       ├── flowmm_train.py      # FlowMM training script
│       └── mattergen_train.py   # MatterGen training script
├── utils/
│   ├── process_raw_data.py      # Raw data processing utilities
│   └── rename_materials_auto.py # Material renaming utilities
├── dataloaders.py               # PyTorch dataloaders
└── requirements.txt
```

## Models

Available model architectures for both tasks:

- **ADIT** - Attention-based diffusion transformer
- **CDVAE** - Crystal diffusion variational autoencoder
- **DiffCSP** - Diffusion for crystal structure prediction
- **FlowLLM** - Flow-based language model
- **FlowMM** - Flow matching for materials
- **MatterGen** - Materials generation model

## Training

All training scripts use centralized configuration files for easy customization. Training hyperparameters, model architectures, and data settings can be modified in the respective config files.

### Configuration System

Each task has a centralized config file:
- **Task 1**: `train/task_1/config.py` - Contains all training hyperparameters, model settings, and data configurations
- **Task 2**: `train/task_2/config.py` - Contains Task 2-specific settings including loss weights (CLS_W, KL_W)

Key features:
- **Dynamic DATA_ROOT**: Automatically resolves relative to the current working directory
- **Model-specific settings**: Seeds, batch sizes, and gradient clipping values per model
- **Centralized hyperparameters**: Learning rate, epochs, optimizer settings, etc.

### Running Training

Simply run the training script from any directory:

```bash
# Task 1: Generate nanoparticles from unit cells
python train/task_1/adit_train.py
python train/task_1/cdvae_train.py
# ... etc

# Task 2: Recover lattice parameters from nanoparticles
python train/task_2/adit_train.py
python train/task_2/cdvae_train.py
# ... etc
```

**Note:** The `DATA_ROOT` path is automatically resolved relative to where you run the script. If you run from `/path/to/project/`, it will use `/path/to/project/C2NP` as the data directory.

### Customizing Training

Edit the config files to customize training:

**Task 1** (`train/task_1/config.py`):
- Modify `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS` for global settings
- Adjust `MODEL_SEEDS` for model-specific random seeds
- Change `MODEL_BATCH_SIZES` for model-specific batch sizes
- Update architecture parameters (`HIDDEN_DIM`, `NUM_LAYERS`, etc.)

**Task 2** (`train/task_2/config.py`):
- Modify loss weights: `CLS_W` (classification), `KL_W` (KL divergence for VAE)
- Adjust model-specific gradient clipping via `MODEL_GRAD_CLIP`
- Same hyperparameter controls as Task 1

All changes in the config files automatically apply to all training scripts.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{c2np2026,
  title={C2NP: A Benchmark for Learning Scale-Dependent Geometric Invariances in 3D Materials Generation},
  author={},
  journal={},
  year={2026}
}
```

## License

See [LICENSE](LICENSE) for details.
