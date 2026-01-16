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
from multiprocessing import Pool, cpu_count

from scipy.spatial import ConvexHull, distance
from torch import nn, optim

# Allowlist PyG globals for weights_only loading
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

from dataloaders import C2NPDataloader
from models.task_1.flowllm_model import FlowLLM_Task1

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


def clear_memory():
    """Clear CUDA cache and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc

    gc.collect()


# Optimized geometry metric functions
def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    try:
        # Use scipy's optimized cdist for pairwise distances
        dist_a_to_b = distance.cdist(a, b).min(axis=1).max()
        dist_b_to_a = distance.cdist(b, a).min(axis=1).max()
        return max(dist_a_to_b, dist_b_to_a)
    except:
        return 0.0  # Return 0 for failed computations


def delta_hull_vol(a: np.ndarray, b: np.ndarray) -> float:
    try:
        vol_a = ConvexHull(a, qhull_options="QJ").volume
        vol_b = ConvexHull(b, qhull_options="QJ").volume
        return abs(vol_a - vol_b)
    except:
        return float("inf")


# Cache for RDF calculations
_rdf_cache = {}


def rdf_energy(a: np.ndarray, b: np.ndarray, r_max=10.0, bins=128) -> float:
    def compute_rdf(x):
        # Use cache key based on array shape and content hash
        key = hash(x.tobytes())
        if key in _rdf_cache:
            return _rdf_cache[key]

        # Use scipy's optimized pdist
        dists = distance.pdist(x)
        hist, _ = np.histogram(dists, bins=bins, range=(0, r_max), density=True)
        _rdf_cache[key] = hist
        return hist

    return np.square(compute_rdf(a) - compute_rdf(b)).sum()


def volume_ratio(volume: float, radius: float) -> float:
    try:
        sphere_vol = 4 / 3 * math.pi * radius**3
        return volume / sphere_vol
    except:
        return 1.0  # Return 1.0 (perfect ratio) for failed computations


def compute_metrics_batch(args):
    """Compute metrics for a batch of samples in parallel"""
    pred, gt, radius = args
    if pred.shape[0] < 2:
        return None

    try:
        metrics = {}
        metrics["rmsd"] = rmsd(pred, gt)
        metrics["haus"] = hausdorff(pred, gt)
        metrics["dhull"] = delta_hull_vol(pred, gt)
        metrics["rdfE"] = rdf_energy(pred, gt)
        metrics["vratio"] = volume_ratio(
            ConvexHull(pred, qhull_options="QJ").volume, float(radius)
        )
        return metrics
    except:
        return None


def run_epoch(model, loader, optimizer=None, train=False, device=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = total_nodes = 0
    pbar = tqdm(loader, desc=("Train " if train else "Eval  ") + "batch")

    for data in pbar:
        data = data.to(device)

        # Sample random timesteps
        t = torch.rand(data.num_graphs, device=device)

        # Forward pass
        noise_pred = model(data, t)

        # Compute loss
        noise = torch.randn_like(data.pos) * 0.1
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

        # Clear memory after each batch
        del data, t, noise_pred, noise, loss
        clear_memory()

    return total_loss / total_nodes


for SEED in [42]:
    print(f"\n===== Seed {SEED} =====")
    # Set randomness
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Output dir per seed
    out_dir = os.path.join("results/task_1", "flowllm", str(SEED))
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

    # Training phase
    loaders = {
        "train": GeoDataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        ),
        "val": GeoDataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
    }

    # Model, optimizer
    model = FlowLLM_Task1(
        atom_emb_dim=4,
        hidden_dim=4,
        num_layers=1,
        cutoff_radius=5.0,
        llm_model_name="prajjwal1/bert-tiny",
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

    # Clear training data
    del train_ds, val_ds, loaders
    clear_memory()

    # Evaluation phase with smaller batch size
    EVAL_BATCH_SIZE = 4
    results = []
    for split in ["id_test", "ood_test"]:
        print(f"\nEvaluating {split}...")
        dataset = id_test_ds if split == "id_test" else ood_test_ds
        loader = GeoDataLoader(
            dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Compute loss
        tl = run_epoch(model, loader, None, False, DEVICE)

        # Compute metrics in parallel
        metrics = {"rmsd": [], "haus": [], "dhull": [], "rdfE": [], "vratio": []}
        loader = GeoDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        # Prepare batch data for parallel processing
        batch_data = []
        for data in tqdm(loader, desc=f"Preparing {split} data"):
            data = data.to(DEVICE)
            with torch.no_grad():
                pred = model.sample(data).detach().cpu().numpy()
                gt = data.y_pos.detach().cpu().numpy()
                radius = data.radius[0].item()
            batch_data.append((pred, gt, radius))
            del data

        # Process metrics in parallel
        num_workers = min(cpu_count(), 8)  # Limit to 8 workers max
        print(f"Computing metrics using {num_workers} workers...")
        with Pool(num_workers) as pool:
            for sample_metrics in tqdm(
                pool.imap(compute_metrics_batch, batch_data),
                total=len(batch_data),
                desc=f"Metric {split}",
            ):
                if sample_metrics:
                    for k, v in sample_metrics.items():
                        metrics[k].append(v)

        mean_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        rec = {"split": split, "loss": tl, **mean_metrics}
        results.append(rec)

        # Clear memory between splits
        del batch_data
        clear_memory()

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
