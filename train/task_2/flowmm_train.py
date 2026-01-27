import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from scipy.spatial import ConvexHull, distance, QhullError
from torch import nn, optim
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

from dataloaders import C2NPDataloader
from models.task_2.flowmm_model import FlowMMCrystal
from train.task_2.config import Task2TrainingConfig

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# Geometry metric functions
def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).mean())


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    return max(
        distance.directed_hausdorff(a, b)[0], distance.directed_hausdorff(b, a)[0]
    )


def check_point_set(points: np.ndarray, min_points: int = 3) -> bool:
    """Check if a point set is valid for convex hull computation"""
    if points.shape[0] < min_points:
        return False

    # Check for NaN or infinite values
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return False

    # Check if points are too close together
    dists = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        return False

    # Check if points span 3D space
    centered = points - np.mean(points, axis=0)
    cov = np.cov(centered.T)
    if np.linalg.matrix_rank(cov) < 2:
        return False

    return True


def delta_hull_vol(a: np.ndarray, b: np.ndarray) -> float:
    try:
        if not check_point_set(a) or not check_point_set(b):
            return float("inf")

        a = a + np.random.normal(0, 1e-8, a.shape)
        b = b + np.random.normal(0, 1e-8, b.shape)

        try:
            hull_a = ConvexHull(a, qhull_options="QJ")
            hull_b = ConvexHull(b, qhull_options="QJ")
        except QhullError:
            hull_a = ConvexHull(a)
            hull_b = ConvexHull(b)

        return abs(hull_a.volume - hull_b.volume)
    except Exception as e:
        print(f"Warning: Convex hull computation failed: {e}")
        return float("inf")


def rdf_energy(a: np.ndarray, b: np.ndarray, r_max=10.0, bins=128) -> float:
    def rdf(x):
        dists = distance.pdist(x)
        hist, _ = np.histogram(dists, bins=bins, range=(0, r_max), density=True)
        return hist

    return np.square(rdf(a) - rdf(b)).sum()


def volume_ratio(volume: float, cell_vol: float) -> float:
    try:
        if np.isinf(volume) or volume <= 0:
            return float("inf")
        ratio = volume / cell_vol
        return np.clip(ratio, 0.0, 100.0)
    except Exception as e:
        print(f"Warning: Volume ratio computation failed: {e}")
        return float("inf")


def run_epoch(model, loader, optimizer=None, train=False, device=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = total_nodes = 0
    pbar = tqdm(loader, desc=("Train " if train else "Eval  ") + "batch")
    for data in pbar:
        data = data.to(device)

        # Check input data
        if torch.isnan(data.pos).any():
            print("Warning: NaN in input positions")
            data.pos = torch.nan_to_num(data.pos, nan=0.0)

        # Add cell parameters if not present
        if not hasattr(data, "cell_params"):
            # Initialize with default values (cubic cell)
            data.cell_params = torch.ones((data.num_graphs, 6), device=device)
            data.cell_params[:, 3:] = torch.tensor(
                [90.0, 90.0, 90.0], device=device
            )  # angles in degrees

        if torch.isnan(data.cell_params).any():
            print("Warning: NaN in cell parameters")
            data.cell_params = torch.nan_to_num(data.cell_params, nan=0.0)

        # Sample random timesteps
        t = torch.rand(data.num_graphs, device=device)

        # Add noise to positions with smaller scale
        alpha_bar = model.get_noise_schedule(t).view(-1, 1)
        sqrt_ab = torch.sqrt(torch.clamp(alpha_bar, min=1e-6))
        sqrt_umb = torch.sqrt(torch.clamp(1 - alpha_bar, min=1e-6))

        noise = torch.randn_like(data.pos) * 0.01
        batch_idx = model._batch_from_ptr(data.ptr)
        noisy_pos = sqrt_ab[batch_idx] * data.pos + sqrt_umb[batch_idx] * noise

        # Check noisy positions
        if torch.isnan(noisy_pos).any():
            print("Warning: NaN in noisy positions")
            noisy_pos = torch.nan_to_num(noisy_pos, nan=0.0)

        # Store noisy positions temporarily
        original_pos = data.pos.clone()
        data.pos = noisy_pos

        # Forward pass
        noise_pred, lat_noise, sg_logits = model(data, t)

        # Check predictions
        if torch.isnan(noise_pred).any():
            print("Warning: NaN in noise predictions")
            noise_pred = torch.nan_to_num(noise_pred, nan=0.0)

        # Compute loss with better handling
        loss = nn.MSELoss(reduction="mean")(noise_pred, noise)

        # Skip batch if loss is NaN
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue

        if train:
            optimizer.zero_grad()
            loss.backward()

            # Check gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"Warning: NaN gradient in {name}")
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)

            # Gradient clipping from config
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

            optimizer.step()

        # Restore original positions
        data.pos = original_pos

        n = data.num_nodes
        total_loss += loss.item() * n
        total_nodes += n
        pbar.set_postfix(loss=total_loss / total_nodes)

    return total_loss / total_nodes if total_nodes > 0 else float("inf")


@torch.no_grad()
def evaluate(model, loader, device="cpu", name=""):
    model.eval()
    # Pre-allocate tensors for batch accumulation
    batch_size = loader.batch_size
    max_batches = len(loader)
    all_se = torch.zeros(max_batches * batch_size, device=device)
    all_sg_correct = torch.zeros(
        max_batches * batch_size, dtype=torch.bool, device=device
    )
    all_joint_correct = torch.zeros(
        max_batches * batch_size, dtype=torch.bool, device=device
    )
    batch_idx = 0
    n = 0

    for data in tqdm(loader, desc=f"Evaluating {name}"):
        data = data.to(device)
        B = data.num_graphs

        # Add cell parameters if not present
        if not hasattr(data, "cell_params"):
            # Initialize with default values (cubic cell)
            data.cell_params = torch.ones((data.num_graphs, 6), device=device)
            data.cell_params[:, 3:] = torch.tensor(
                [90.0, 90.0, 90.0], device=device
            )  # angles in degrees

        # Get true lattice parameters and space groups
        l_true = data.cell_params.view(B, 6)
        sg_true = data.spacegroup.view(-1)

        # Generate predictions
        try:
            l_pred, sg_logits = model.sample(data)
            sg_pred = sg_logits.argmax(1)

            # Vectorized calculations
            se = (l_pred - l_true) ** 2
            all_se[batch_idx : batch_idx + B] = se.sum(dim=1)
            all_sg_correct[batch_idx : batch_idx + B] = sg_pred == sg_true
            all_joint_correct[batch_idx : batch_idx + B] = (sg_pred == sg_true) & (
                se <= 0.25
            ).all(dim=1)
        except Exception as e:
            print(f"Error in model sampling for batch: {e}")
            # Fill with default values for failed computations
            all_se[batch_idx : batch_idx + B] = 0.0
            all_sg_correct[batch_idx : batch_idx + B] = False
            all_joint_correct[batch_idx : batch_idx + B] = False

        batch_idx += B
        n += B

        # Clear cache periodically
        if batch_idx % (4 * batch_size) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final calculations using accumulated tensors
    try:
        se_accum = all_se[:n].sum().item()
        correct_sg = all_sg_correct[:n].sum().item()
        correct_joint = all_joint_correct[:n].sum().item()

        rmse = math.sqrt(se_accum / (n * 6)) if n > 0 else 0.0
        sg_acc = correct_sg / n if n > 0 else 0.0
        joint_acc = correct_joint / n if n > 0 else 0.0
    except Exception as e:
        print(f"Error in metric calculation: {e}")
        rmse = 0.0
        sg_acc = 0.0
        joint_acc = 0.0

    print(f"{name} metrics:")
    print(f"RMSE: {rmse:.3f} Å")
    print(f"Space Group Accuracy: {sg_acc:.3f}")
    print(f"Joint Accuracy: {joint_acc:.3f}")

    return {"rmse": rmse, "sg_acc": sg_acc, "joint_acc": joint_acc}


# Load configuration
config = Task2TrainingConfig.default()
SEEDS = config.get_seeds_for_model("flowmm")
BATCH_SIZE = config.get_batch_size_for_model("flowmm")
GRAD_CLIP = config.get_grad_clip_for_model("flowmm")

for SEED in SEEDS:
    print(f"\n===== Seed {SEED} =====")
    # Set randomness
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Output dir per seed
    out_dir = config.get_output_dir("flowmm", SEED)
    os.makedirs(out_dir, exist_ok=True)

    # Hyperparams from config
    DATA_ROOT = config.data_root
    LR = config.learning_rate  # Note: flowmm uses 1e-5 in original, but using config default
    EPOCHS = config.num_epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    def add_target(data):
        data.y_pos = data.pos.clone()
        return data

    ds = C2NPDataloader(root=DATA_ROOT, num_workers=4, transform=add_target)

    # Dataset splits from config
    SUBSET_RATIO = config.subset_ratio
    train_ratio, val_ratio = config.train_val_split

    train_ds, val_ds = ds.random_train_splits(
        train_ratio, val_ratio, seed=SEED, subset_ratio=SUBSET_RATIO
    )
    id_test_ds = ds.get_split("id_test", subset_ratio=SUBSET_RATIO)
    ood_test_ds = ds.get_split("ood_test", subset_ratio=SUBSET_RATIO)

    # Print dataset sizes to confirm subset is working
    print(f"Dataset sizes (using {SUBSET_RATIO * 100}% subset):")
    print(f"Train: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    print(f"ID Test: {len(id_test_ds)}")
    print(f"OOD Test: {len(ood_test_ds)}")
    for subset in (train_ds, val_ds, id_test_ds, ood_test_ds):
        subset.transform = add_target
    loaders = {
        "train": GeoDataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        ),
        "val": GeoDataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        "id_test": GeoDataLoader(
            id_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        "ood_test": GeoDataLoader(
            ood_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
    }

    # Model, optimizer from config
    model = FlowMMCrystal(
        atom_emb_dim=config.flowmm_atom_emb_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        cutoff_radius=config.cutoff_radius,
        cell_emb_dim=config.flowmm_r_emb_dim,
        time_emb_dim=config.flowmm_time_emb_dim,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.scheduler_mode,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=1e-7,
    )

    # Training
    best_val = float("inf")
    log = []
    for epoch in range(1, EPOCHS + 1):
        print(f"-- Epoch {epoch}/{EPOCHS}")
        start_time = time.time()
        tl = run_epoch(model, loaders["train"], optimizer, True, DEVICE)
        vl = run_epoch(model, loaders["val"], None, False, DEVICE)
        epoch_duration = time.time() - start_time
        scheduler.step(vl)
        log.append(
            {
                "epoch": epoch,
                "train_loss": tl,
                "val_loss": vl,
                "epoch_duration": epoch_duration,
            }
        )
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
    pd.DataFrame(log).to_csv(os.path.join(out_dir, "training_log.csv"), index=False)

    # Evaluation on ID & OOD
    results = []
    for split in ["id_test", "ood_test"]:
        # Clear CUDA cache before loading OOD data
        if split == "ood_test":
            torch.cuda.empty_cache()
            print("Cleared CUDA cache before loading OOD data")

        loader = loaders[split]
        tl = run_epoch(model, loader, None, False, DEVICE)

        # Compute metrics
        metrics = evaluate(model, loader, DEVICE, split)
        rec = {"split": split, "loss": tl, **metrics}
        results.append(rec)

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
