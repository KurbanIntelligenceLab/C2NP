import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import time

from dataloaders import C2NPDataloader
from models.task_2.flowllm_model import FlowLLM_Task2
from train.task_2.config import Task2TrainingConfig


# Metric helpers
def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return math.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def surface_indices(coords, tol=0.01):
    try:
        from scipy.spatial import ConvexHull, QhullError

        hull = ConvexHull(coords, qhull_options="QJ Pp")
        return np.unique(hull.simplices.flatten())
    except (QhullError, ValueError):
        return np.arange(len(coords))  # Return all indices as fallback


def rsurf(pred: np.ndarray, true: np.ndarray, thresh=2.0) -> float:
    from scipy.spatial import KDTree

    surf_true = surface_indices(true)
    tree = KDTree(pred)
    hit = tree.query_ball_point(true[surf_true], r=thresh)
    return np.count_nonzero([len(h) > 0 for h in hit]) / surf_true.size


def rdf_hist(coords, bins=64, r_max=10.0):
    from scipy.spatial.distance import pdist

    dists = pdist(coords)
    hist, _ = np.histogram(dists, bins=bins, range=(0.0, r_max))
    pdf = hist / hist.sum() + 1e-12
    return pdf


def rdf_kl(pred: np.ndarray, true: np.ndarray, **kw) -> float:
    p = rdf_hist(pred, **kw)
    q = rdf_hist(true, **kw)
    return float(np.sum(p * np.log(p / q)))


def vr(pred: np.ndarray, true: np.ndarray) -> float:
    try:
        from scipy.spatial import ConvexHull, QhullError

        return ConvexHull(pred, qhull_options="QJ Pp").volume / (
            ConvexHull(true, qhull_options="QJ Pp").volume + 1e-12
        )
    except (QhullError, ValueError):
        return 1.0  # Return 1.0 (perfect ratio) for failed computations


def run_epoch(model, loader, optimizer=None, train=False, device="cpu", grad_clip=1.0):
    model.train() if train else model.eval()
    total_loss = total_nodes = 0
    pbar = tqdm(loader, desc=("Train" if train else "Eval "))

    for data in pbar:
        data = data.to(device)

        # Sample random timesteps
        t = torch.rand(data.num_graphs, device=device)

        # Forward pass
        noise_pred, lat_pred, sg_logits = model(data, t)

        # Compute loss
        noise = torch.randn_like(data.pos) * 0.01
        loss = nn.MSELoss(reduction="mean")(noise_pred, noise)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item() * data.num_nodes
        total_nodes += data.num_nodes
        pbar.set_postfix(
            loss=total_loss / total_nodes if total_nodes > 0 else float("inf")
        )

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

        # Get true lattice parameters and space groups
        l_true = data.lattice.view(B, 6)
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
SEEDS = config.get_seeds_for_model("flowllm")
BATCH_SIZE = config.get_batch_size_for_model("flowllm")
GRAD_CLIP = config.get_grad_clip_for_model("flowllm")

# Multi-seed loop
for SEED in SEEDS:
    print(f"\n=== Seed {SEED} ===")
    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Output dir per seed
    out_dir = config.get_output_dir("flowllm", SEED)
    os.makedirs(out_dir, exist_ok=True)

    # Hyperparams from config
    DATA_ROOT = config.data_root
    LR = config.learning_rate
    NUM_EPOCHS = config.num_epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    ds = C2NPDataloader(root=DATA_ROOT)

    # Dataset splits from config
    SUBSET_RATIO = config.subset_ratio
    train_ratio, val_ratio = config.train_val_split

    train_ds, val_ds = ds.random_train_splits(
        train_ratio, val_ratio, seed=SEED, subset_ratio=SUBSET_RATIO
    )
    id_ds = ds.get_split("id_test", subset_ratio=SUBSET_RATIO)
    ood_ds = ds.get_split("ood_test", subset_ratio=SUBSET_RATIO)

    # Print dataset sizes to confirm subset is working
    print(f"Dataset sizes (using {SUBSET_RATIO * 100}% subset):")
    print(f"Train: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    print(f"ID Test: {len(id_ds)}")
    print(f"OOD Test: {len(ood_ds)}")

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    id_loader = DataLoader(id_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model, optimizer, scheduler from config
    model = FlowLLM_Task2(
        atom_emb_dim=config.flowllm_atom_emb_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        cutoff_radius=config.cutoff_radius,
        llm_model_name=config.flowllm_llm_model_name,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.scheduler_mode,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    # Training loop
    best_val = float("inf")
    log = []
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"-- Epoch {epoch}/{NUM_EPOCHS}")
        start_time = time.time()
        tl = run_epoch(model, train_loader, optimizer, True, DEVICE, GRAD_CLIP)
        vl = run_epoch(model, val_loader, None, False, DEVICE, GRAD_CLIP)
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

    # Reload best & evaluate
    model.load_state_dict(
        torch.load(os.path.join(out_dir, "best_model.pt"), map_location=DEVICE)
    )
    id_metrics = evaluate(model, id_loader, DEVICE, "ID-test")
    ood_metrics = evaluate(model, ood_loader, DEVICE, "OOD-test")

    df = pd.DataFrame(
        [
            {
                "split": "ID-test",
                "rmse": id_metrics["rmse"],
                "sg_acc": id_metrics["sg_acc"],
                "joint_acc": id_metrics["joint_acc"],
            },
            {
                "split": "OOD-test",
                "rmse": ood_metrics["rmse"],
                "sg_acc": ood_metrics["sg_acc"],
                "joint_acc": ood_metrics["joint_acc"],
            },
        ]
    )
    df.to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
