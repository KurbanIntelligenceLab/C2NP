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
from scipy.spatial import ConvexHull, distance
from torch import nn, optim
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

from dataloaders import C2NPDataloader
from models.task_1.adit_model import ADiTUnitCell

# Allowlist PyG globals for weights_only loading
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# Set memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Geometry metric functions
def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).mean())


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    return max(
        distance.directed_hausdorff(a, b)[0], distance.directed_hausdorff(b, a)[0]
    )


def delta_hull_vol(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return abs(
            ConvexHull(a, qhull_options="QJ").volume
            - ConvexHull(b, qhull_options="QJ").volume
        )
    except:
        return 0.0  # Return 0 for failed computations


def rdf_energy(a: np.ndarray, b: np.ndarray, r_max=10.0, bins=128) -> float:
    def rdf(x):
        dists = distance.pdist(x)
        hist, _ = np.histogram(dists, bins=bins, range=(0, r_max), density=True)
        return hist

    return np.square(rdf(a) - rdf(b)).sum()


def volume_ratio(volume: float, radius: float) -> float:
    try:
        sphere_vol = 4 / 3 * math.pi * radius**3
        return volume / sphere_vol
    except:
        return 1.0  # Return 1.0 (perfect ratio) for failed computations


def run_epoch(model, loader, optimizer=None, train=False, device=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = total_nodes = 0
    pbar = tqdm(loader, desc=("Train " if train else "Eval  ") + "batch")

    for data in pbar:
        data = data.to(device)
        batch_size = data.num_graphs

        # Sample random timesteps
        t = torch.rand(batch_size, device=device)

        # Add noise to positions
        noise = torch.randn_like(data.pos) * 0.1

        # Predict noise
        noise_pred = model(data, t)

        # Compute loss
        loss = nn.MSELoss(reduction="mean")(noise_pred, noise)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        n = data.num_nodes
        total_loss += loss.item() * n
        total_nodes += n
        pbar.set_postfix(loss=total_loss / total_nodes)

        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / total_nodes


def compute_metrics(pred, gt, radius=None):
    """Compute metrics for a single structure"""
    metrics = {}

    # RMSD
    metrics["rmsd"] = rmsd(pred, gt)

    # Hausdorff distance
    metrics["haus"] = hausdorff(pred, gt)

    # Delta hull volume
    metrics["dhull"] = delta_hull_vol(pred, gt)

    # RDF energy
    metrics["rdfE"] = rdf_energy(pred, gt)

    # Volume ratio if radius is provided
    if radius is not None:
        try:
            hull_vol = ConvexHull(pred, qhull_options="QJ").volume
            metrics["vratio"] = volume_ratio(hull_vol, float(radius))
        except:
            metrics["vratio"] = 1.0

    return metrics


def evaluate_model(model, loader, device):
    """Evaluate model with memory-efficient sampling and metric computation"""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating"):
            data = data.to(device)

            # Sample in smaller chunks
            pred = model.sample(data, num_steps=1000, chunk_size=50)
            pred = pred.detach().cpu().numpy()
            gt = data.y_pos.detach().cpu().numpy()
            ptr = data.ptr.cpu().numpy()

            # Process each structure in the batch
            for i in range(len(ptr) - 1):
                s, e = ptr[i], ptr[i + 1]
                if e - s < 2:  # Skip structures with less than 2 atoms
                    continue

                Pp, Pg = pred[s:e], gt[s:e]
                radius = data.radius[i].item() if hasattr(data, "radius") else None

                # Compute metrics
                metrics = compute_metrics(Pp, Pg, radius)
                all_metrics.append(metrics)

            # Clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Aggregate metrics
    mean_metrics = {}
    for k in all_metrics[0].keys():
        values = [m[k] for m in all_metrics if not (np.isinf(m[k]) or np.isnan(m[k]))]
        mean_metrics[k] = float(np.mean(values)) if values else 0.0

    return mean_metrics


for SEED in [42, 50, 60]:
    print(f"\n===== Seed {SEED} =====")
    # Set randomness
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Output dir per seed
    out_dir = os.path.join("results/task_1", "adit", str(SEED))
    os.makedirs(out_dir, exist_ok=True)

    # Hyperparams
    DATA_ROOT = "C2NP"
    BATCH_SIZE = 1
    LR = 1e-4
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    def add_target(data):
        data.y_pos = data.pos.clone()
        data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
        return data

    def clean(data):
        data.y_pos = data.pos.clone()
        if not hasattr(data, "cell_ptr"):
            data.cell_ptr = torch.tensor([0, data.cell_pos.size(0)], dtype=torch.long)
        return data

    ds = C2NPDataloader(root=DATA_ROOT, num_workers=4, transform=add_target)

    # Use only 20% of each split to prevent crashes
    SUBSET_RATIO = 1

    train_ds, val_ds = ds.random_train_splits(
        0.8, 0.2, seed=SEED, subset_ratio=SUBSET_RATIO
    )
    id_test_ds = ds.get_split("id_test", subset_ratio=SUBSET_RATIO)
    ood_test_ds = ds.get_split("ood_test", subset_ratio=SUBSET_RATIO)

    # Print dataset sizes to confirm subset is working
    print(f"Dataset sizes (using {SUBSET_RATIO*100}% subset):")
    print(f"Train: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    print(f"ID Test: {len(id_test_ds)}")
    print(f"OOD Test: {len(ood_test_ds)}")

    for subset in (train_ds, val_ds, id_test_ds, ood_test_ds):
        subset.transform = clean
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

    # Model, optimizer
    model = ADiTUnitCell(
        atom_emb_dim=4,
        hidden_dim=4,
        num_layers=1,
        cutoff_radius=5.0,
        r_emb_dim=4,
        time_emb_dim=4,
        num_heads=2,
        dropout=0.1,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, 5)

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
        loader = loaders[split]
        tl = run_epoch(model, loader, None, False, DEVICE)

        # Compute metrics using memory-efficient evaluation
        metrics = evaluate_model(model, loader, DEVICE)

        rec = {"split": split, "loss": tl, **metrics}
        results.append(rec)

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
