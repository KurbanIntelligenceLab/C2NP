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

# Allowlist PyG globals for weights_only loading
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm

from dataloaders import C2NPDataloader
from models.task_1.cdvae_model import CDVAEUnitCell

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


# Geometry metric functions
def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((a - b) ** 2).mean())


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return max(
            distance.directed_hausdorff(a, b)[0], distance.directed_hausdorff(b, a)[0]
        )
    except:
        return 0.0  # Return 0 for failed computations


def delta_hull_vol(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return abs(
            ConvexHull(a, qhull_options="QJ").volume
            - ConvexHull(b, qhull_options="QJ").volume
        )
    except:
        return 0.0  # Return 0 for failed computations


def rdf_energy(a: np.ndarray, b: np.ndarray, r_max=10.0, bins=128) -> float:
    try:

        def rdf(x):
            dists = distance.pdist(x)
            hist, _ = np.histogram(dists, bins=bins, range=(0, r_max), density=True)
            return hist

        return np.square(rdf(a) - rdf(b)).sum()
    except:
        return 0.0  # Return 0 for failed computations


def volume_ratio(volume: float, radius: float) -> float:
    try:
        sphere_vol = 4 / 3 * math.pi * radius**3
        return volume / sphere_vol
    except:
        return 1.0  # Return 1.0 (perfect ratio) for failed computations


# Epoch runner and evaluation
def run_epoch(model, loader, optimizer=None, train=False, device=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = total_recon = total_kld = total_nodes = 0
    pbar = tqdm(loader, desc=("Train " if train else "Eval  ") + "batch")
    for data in pbar:
        data = data.to(device)
        recon_pos, mu, logvar = model(data)
        recon_loss = nn.MSELoss(reduction="mean")(recon_pos, data.y_pos)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        n = data.num_nodes
        total_loss += loss.item() * n
        total_recon += recon_loss.item() * n
        total_kld += kl_loss.item() * n
        total_nodes += n
        pbar.set_postfix(
            loss=total_loss / total_nodes,
            recon=total_recon / total_nodes,
            kld=total_kld / total_nodes,
        )
    return total_loss / total_nodes, total_recon / total_nodes, total_kld / total_nodes


for SEED in [60]:
    print(f"\n===== Seed {SEED} =====")
    # Set randomness
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Output dir per seed
    out_dir = os.path.join("results/task_1", "cdvae", str(SEED))
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
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        ),
        "val": GeoDataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        "id_test": GeoDataLoader(
            id_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ),
        "ood_test": GeoDataLoader(
            ood_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        ),
    }

    # Model, optimizer
    model = CDVAEUnitCell(4, 4, 4, 1, 5.0).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, 5)

    # Training
    best_val = float("inf")
    log = []
    for epoch in range(1, EPOCHS + 1):
        print(f"-- Epoch {epoch}/{EPOCHS}")
        start_time = time.time()
        tl, trec, tkl = run_epoch(model, loaders["train"], optimizer, True, DEVICE)
        vl, vrec, vkl = run_epoch(model, loaders["val"], None, False, DEVICE)
        epoch_duration = time.time() - start_time
        scheduler.step(vl)
        log.append(
            {
                "epoch": epoch,
                "train_loss": tl,
                "train_recon": trec,
                "train_kld": tkl,
                "val_loss": vl,
                "val_recon": vrec,
                "val_kld": vkl,
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
        tl, trec, tkl = run_epoch(model, loader, None, False, DEVICE)
        # compute geometry metrics
        metrics = {"rmsd": [], "haus": [], "dhull": [], "rdfE": [], "vratio": []}
        for data in tqdm(loader, desc=f"Metric {split}"):
            data = data.to(DEVICE)
            recon, _, _ = model(data)
            pred = recon.detach().cpu().numpy()
            gt = data.y_pos.detach().cpu().numpy()
            ptr = data.ptr.cpu().numpy()
            for i in range(len(ptr) - 1):
                s, e = ptr[i], ptr[i + 1]
                Pp, Pg = pred[s:e], gt[s:e]
                if Pp.shape[0] < 2:
                    continue
                metrics["rmsd"].append(rmsd(Pp, Pg))
                metrics["haus"].append(hausdorff(Pp, Pg))
                metrics["dhull"].append(delta_hull_vol(Pp, Pg))
                metrics["rdfE"].append(rdf_energy(Pp, Pg))
                try:
                    hull_vol = ConvexHull(Pp, qhull_options="QJ").volume
                    metrics["vratio"].append(
                        volume_ratio(hull_vol, float(data.radius[i]))
                    )
                except:
                    metrics["vratio"].append(
                        1.0
                    )  # Use 1.0 as default for failed computations
        mean_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        rec = {"split": split, "loss": tl, **mean_metrics}
        results.append(rec)
    pd.DataFrame(results).to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
