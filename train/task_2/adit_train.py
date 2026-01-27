import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from dataloaders import C2NPDataloader
from models.task_2.adit_model import ADiT_Task2
from train.task_2.config import Task2TrainingConfig

# Allowlist PyG globals
torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])

# Set memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Loss functions
def reg_loss_fn():
    return nn.MSELoss()


def cls_loss_fn():
    return nn.CrossEntropyLoss()


# Strip unit-cell fields to avoid leakage
def strip_global(data):
    for k in ("cell_pos", "cell_cell", "cell_z"):
        if hasattr(data, k):
            delattr(data, k)
    return data


# Single-epoch runner
def run_epoch(model, loader, optimizer=None, train=False, device="cpu", cls_w=0.5, grad_clip=1.0):
    if train:
        model.train()
    else:
        model.eval()
    tot = reg_tot = cls_tot = correct = n_graph = 0
    pbar = tqdm(loader, desc=("Train" if train else "Eval "))
    reg_loss = reg_loss_fn()
    cls_loss = cls_loss_fn()

    for batch in pbar:
        batch = batch.to(device)
        B = int(batch.ptr.size(0) - 1)

        # True targets
        l_true = batch.lattice.view(B, 6)
        sg_true = batch.spacegroup.view(-1)

        with torch.set_grad_enabled(train):
            # Sample random timesteps
            t = torch.rand(B, device=device)

            # Add noise to lattice parameters
            noise = torch.randn_like(l_true)
            alpha = 1 - model.get_noise_schedule(t)
            alpha = alpha.view(-1, 1)

            # Predict noise and space group
            noise_pred, sg_logits = model(batch, t)

            # Compute losses
            loss_r = reg_loss(noise_pred, noise)
            loss_c = cls_loss(sg_logits, sg_true)
            loss = loss_r + cls_w * loss_c

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        tot += loss.item() * B
        reg_tot += loss_r.item() * B
        cls_tot += loss_c.item() * B
        correct += (sg_logits.argmax(1) == sg_true).sum().item()
        n_graph += B

        pbar.set_postfix(
            loss=tot / n_graph,
            reg=reg_tot / n_graph,
            cls=cls_tot / n_graph,
            acc=correct / n_graph,
        )

        # Clear cache after each batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return tot / n_graph, reg_tot / n_graph, cls_tot / n_graph, correct / n_graph


def evaluate_model(model, loader, device):
    """Evaluate model with memory-efficient sampling"""
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

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            B = batch.ptr.size(0) - 1
            l_true = batch.lattice.view(B, 6)
            sg_true = batch.spacegroup.view(-1)

            # Generate samples with smaller chunk size
            try:
                l_pred, sg_logits = model.sample(batch, num_steps=1000, chunk_size=50)
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
        return rmse, sg_acc, joint_acc
    except Exception as e:
        print(f"Error in metric calculation: {e}")
        return 0.0, 0.0, 0.0  # Return default values for failed computations


# Load configuration
config = Task2TrainingConfig.default()
SEEDS = config.get_seeds_for_model("adit")
BATCH_SIZE = config.get_batch_size_for_model("adit")
GRAD_CLIP = config.get_grad_clip_for_model("adit")

for SEED in SEEDS:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Output directory
    out_dir = config.get_output_dir("adit", SEED)
    os.makedirs(out_dir, exist_ok=True)

    # Hyperparams from config
    DATA_ROOT = config.data_root
    LR = config.learning_rate
    NUM_EPOCHS = config.num_epochs
    CLS_W = config.cls_weight
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with strip_global transform
    ds = C2NPDataloader(root=DATA_ROOT, transform=strip_global)
    # Compute number of spacegroup classes on full train split
    otrain_full = ds.get_split("train")
    NUM_SG = int(otrain_full.data.spacegroup.max()) + 1

    # Dataset splits from config
    SUBSET_RATIO = config.subset_ratio
    train_ratio, val_ratio = config.train_val_split

    # Random in-dist splits and OOD
    train_ds, val_ds = ds.random_train_splits(
        train_ratio, val_ratio, seed=SEED, subset_ratio=SUBSET_RATIO
    )
    id_test_ds = ds.get_split("id_test", subset_ratio=SUBSET_RATIO)
    ood_ds = ds.get_split("ood_test", subset_ratio=SUBSET_RATIO)

    # Print dataset sizes to confirm subset is working
    print(f"Dataset sizes (using {SUBSET_RATIO * 100}% subset):")
    print(f"Train: {len(train_ds)}")
    print(f"Val: {len(val_ds)}")
    print(f"ID Test: {len(id_test_ds)}")
    print(f"OOD Test: {len(ood_ds)}")

    # Ensure strip_global applied
    for subset in (train_ds, val_ds, id_test_ds, ood_ds):
        subset.transform = strip_global

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    id_loader = DataLoader(id_test_ds, batch_size=BATCH_SIZE, shuffle=False)
    ood_loader = DataLoader(ood_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model, optimizer, scheduler from config
    model = ADiT_Task2(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        cutoff=config.cutoff_radius,
        num_spacegroups=NUM_SG,
        num_heads=config.adit_num_heads,
        dropout=config.adit_dropout,
        chunk_size=config.adit_chunk_size,
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
        tl = run_epoch(model, train_loader, optimizer, True, DEVICE, CLS_W, GRAD_CLIP)
        vl = run_epoch(model, val_loader, None, False, DEVICE, CLS_W, GRAD_CLIP)
        epoch_duration = time.time() - start_time
        scheduler.step(vl[0])
        log.append(
            {
                "epoch": epoch,
                "train_loss": tl[0],
                "train_reg": tl[1],
                "train_cls": tl[2],
                "train_acc": tl[3],
                "val_loss": vl[0],
                "val_reg": vl[1],
                "val_cls": vl[2],
                "val_acc": vl[3],
                "epoch_duration": epoch_duration,
            }
        )
        if vl[0] < best_val:
            best_val = vl[0]
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    # Save training history
    pd.DataFrame(log).to_csv(os.path.join(out_dir, "training_log.csv"), index=False)

    # Reload best & evaluate
    model.load_state_dict(
        torch.load(os.path.join(out_dir, "best_model.pt"), map_location=DEVICE)
    )

    # Evaluate on ID and OOD test sets
    id_metrics = evaluate_model(model, id_loader, DEVICE)
    ood_metrics = evaluate_model(model, ood_loader, DEVICE)

    # Save test results
    df = pd.DataFrame(
        [
            {
                "split": "ID-test",
                "rmse": id_metrics[0],
                "sg_acc": id_metrics[1],
                "joint_acc": id_metrics[2],
            },
            {
                "split": "OOD-test",
                "rmse": ood_metrics[0],
                "sg_acc": ood_metrics[1],
                "joint_acc": ood_metrics[2],
            },
        ]
    )
    df.to_csv(os.path.join(out_dir, "test_results.csv"), index=False)
