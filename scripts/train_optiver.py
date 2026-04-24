#!/usr/bin/env python3
"""
train_optiver.py — Train adapted DeepLOB on Optiver LOB data
=============================================================
Trains a DeepLOBLite model (adapted for 2-level LOB, 8 features) on the
Optiver Realized Volatility Prediction dataset.

Three training phases:
  Phase 1 — Universal base training:
            train on source-stock pool, evaluate zero-shot on unseen stocks
  Phase 2 — Transfer learning:
            fine-tune on each unseen stock (freeze conv + inception)
  Phase 3 — Specific-stock out-of-sample:
            train from scratch on an early segment of a single stock and test
            on its later segment

Architecture change from original DeepLOB (10-level → 2-level):
  • Input: (B, 1, T, 8)  instead of (B, 1, T, 40)
  • Conv Block 3: kernel (1,2) instead of (1,10)  [matches 2 remaining width]
  • All other layers identical — LSTM and FC dimensions unchanged

Usage
-----
    python scripts/train_optiver.py [--epochs 50] [--transfer-epochs 20]

Prerequisites
-------------
    Run scripts/prepare_optiver.py first to generate data/optiver_processed/.

Outputs (all under results/optiver/)
--------
    models/optiver_base.pt              — base model (best val on train stocks)
    results/optiver/base_metrics.pkl    — per-horizon metrics on held-out stocks
    results/optiver/transfer_metrics.pkl — transfer results per target stock
    results/optiver/base_loss.png       — training loss curve
    results/optiver/transfer_comparison.png — before/after fine-tune accuracy
    results/optiver/cm_base.png         — confusion matrices on held-out set
"""
import argparse
import gc
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    cohen_kappa_score, matthews_corrcoef,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "optiver_processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
RESULT_DIR  = os.path.join(BASE_DIR, "results", "optiver")
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train DeepLOBLite on Optiver data")
    p.add_argument("--epochs",           type=int,   default=20,   help="Base training epochs")
    p.add_argument("--transfer-epochs",  type=int,   default=8,    help="Transfer fine-tune epochs")
    p.add_argument("--batch-size",       type=int,   default=64,   help="Mini-batch size")
    p.add_argument("--lr",               type=float, default=1e-4, help="Base learning rate")
    p.add_argument("--transfer-lr",      type=float, default=5e-5, help="Transfer fine-tune LR")
    p.add_argument("--weight-decay",     type=float, default=1e-4, help="Adam weight decay")
    p.add_argument("--patience",         type=int,   default=5,    help="Early-stopping patience")
    p.add_argument("--min-epochs",       type=int,   default=5,    help="Minimum epochs before early stop")
    p.add_argument("--lookback",         type=int,   default=50,   help="LOB lookback T (events)")
    p.add_argument("--horizon",          type=int,   default=1,
                   help="Primary horizon index (0-based, maps to horizons list in prepare_optiver)")
    p.add_argument("--base-max-per-stock", type=int, default=2000,
                   help="Max sampled windows per train stock for base training")
    p.add_argument("--transfer-max-samples", type=int, default=6000,
                   help="Max sampled windows per transfer stock")
    p.add_argument("--max-transfer-stocks", type=int, default=5,
                     help="Number of held-out stocks used for transfer-learning study")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="DataLoader workers (-1: auto, 0 on CPU / 4 on GPU)")
    p.add_argument("--specific-epochs",  type=int,   default=20,   help="Specific-stock OOS epochs")
    p.add_argument("--force",            action="store_true",      help="Retrain even if saved")
    return p.parse_args()

HORIZONS = [1, 2, 3, 5, 10]   # must match prepare_optiver.py defaults

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class OptiverLOBDataset(data.Dataset):
    """Sliding-window LOB dataset for a single Optiver stock.

    Parameters
    ----------
    X : (N, 8)  normalised LOB feature matrix
    y : (N,)    integer labels  (0=Down, 1=Stat, 2=Up, -1=invalid)
    T : int     lookback window length
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        T: int = 50,
        max_samples: int | None = None,
        time_id: np.ndarray | None = None,
        mid: np.ndarray | None = None,
    ):
        self.T = T
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.time_id = (
            time_id.astype(np.int32, copy=False)
            if time_id is not None
            else np.zeros(len(X), dtype=np.int32)
        )
        self.mid = (
            mid.astype(np.float32, copy=False)
            if mid is not None
            else np.zeros(len(X), dtype=np.float32)
        )

        valid_end = np.arange(T, len(X) + 1)
        valid_end = valid_end[self.y[valid_end - 1] >= 0]
        if max_samples is not None and len(valid_end) > max_samples:
            idx = np.linspace(0, len(valid_end) - 1, num=max_samples, dtype=int)
            valid_end = valid_end[idx]
        self.valid_end = valid_end.astype(np.int64, copy=False)

    def __len__(self):
        return len(self.valid_end)

    def __getitem__(self, idx):
        end = int(self.valid_end[idx])
        x = self.X[end - self.T : end][np.newaxis, :, :]
        y = int(self.y[end - 1])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def load_stock_dataset(stock_id: int, horizon_k: int, T: int, max_samples: int | None = None) -> OptiverLOBDataset | None:
    """Load one stock's pre-processed .npz and wrap in OptiverLOBDataset."""
    path = os.path.join(DATA_DIR, f"stock_{stock_id}_data.npz")
    if not os.path.exists(path):
        return None
    npz = np.load(path)
    X = npz["X"]
    time_id = npz["time_id"] if "time_id" in npz else None
    mid = npz["mid"] if "mid" in npz else None
    key = f"y_{horizon_k}"
    if key not in npz:
        return None
    y = npz[key].astype(np.int32)
    ds = OptiverLOBDataset(X, y, T=T, max_samples=max_samples, time_id=time_id, mid=mid)
    if len(ds) == 0:
        return None
    return ds


def concat_datasets(stock_ids: list, horizon_k: int, T: int,
                    val_frac: float = 0.15, max_per_stock: int | None = None) -> tuple:
    """Concatenate datasets from multiple stocks, split into train/val."""
    all_datasets = []
    skipped = []
    for sid in stock_ids:
        ds = load_stock_dataset(sid, horizon_k, T, max_samples=max_per_stock)
        if ds is None or len(ds) < 100:
            skipped.append(sid)
            continue
        all_datasets.append(ds)

    if not all_datasets:
        return None, None

    combined = data.ConcatDataset(all_datasets)
    n        = len(combined)
    n_val    = max(100, int(n * val_frac))
    n_train  = n - n_val
    train_ds, val_ds = data.random_split(combined, [n_train, n_val],
                                          generator=torch.Generator().manual_seed(42))
    if skipped:
        print(f"  Skipped stocks (no data): {skipped}")
    print(f"  Combined: {n:,} samples ({n_train:,} train, {n_val:,} val) "
          f"from {len(all_datasets)} stocks")
    return train_ds, val_ds


def dataset_labels(ds: data.Dataset) -> np.ndarray:
    """Extract labels without materializing input windows."""
    if isinstance(ds, OptiverLOBDataset):
        return ds.y[ds.valid_end - 1].astype(np.int64, copy=False)
    if isinstance(ds, data.Subset):
        base_labels = dataset_labels(ds.dataset)
        return base_labels[np.asarray(ds.indices, dtype=np.int64)]
    if isinstance(ds, data.ConcatDataset):
        return np.concatenate([dataset_labels(sub_ds) for sub_ds in ds.datasets], axis=0)
    raise TypeError(f"Unsupported dataset type for label extraction: {type(ds)!r}")


def make_class_weights(ds: data.Dataset, num_classes: int = 3) -> tuple[torch.Tensor, np.ndarray]:
    """Inverse-sqrt class weights to reduce majority-class collapse."""
    labels = dataset_labels(ds)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights = np.sqrt(safe_counts.sum() / safe_counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.astype(np.int64)


def make_weighted_sampler(ds: data.Dataset, num_classes: int = 3) -> WeightedRandomSampler:
    """Balanced sampler so each mini-batch sees minority classes more often."""
    labels = dataset_labels(ds)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1.0)
    sample_weights = inv[labels]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
    )


def make_temporal_splits(
    ds: data.Dataset,
    train_frac: float = 0.7,
    val_frac_within_train: float = 0.2,
    min_train: int = 100,
    min_val: int = 20,
    min_test: int = 50,
):
    """Sequential train/val/test split to avoid temporal leakage."""
    n_total = len(ds)
    n_prefix = int(n_total * train_frac)
    n_test = n_total - n_prefix
    if n_prefix <= 0 or n_test < min_test:
        return None

    n_val = max(min_val, int(n_prefix * val_frac_within_train))
    n_train = n_prefix - n_val
    if n_train < min_train:
        return None

    train_ds = data.Subset(ds, range(0, n_train))
    val_ds = data.Subset(ds, range(n_train, n_prefix))
    test_ds = data.Subset(ds, range(n_prefix, n_total))
    return train_ds, val_ds, test_ds, n_prefix, n_total


# ---------------------------------------------------------------------------
# DeepLOBLite — adapted for 2-level LOB (8 features)
# ---------------------------------------------------------------------------
class DeepLOBLite(nn.Module):
    """DeepLOB adapted for 2-level LOB (8 input features instead of 40).

    Architecture change vs. original DeepLOB (10 price levels):
      Block 3 uses kernel (1,2) instead of (1,10) to match the 2 remaining
      spatial positions after two (1,2) stride-2 convolutions.

    Input:  (B, 1, T, 8)
    Output: (B, 3) — logits
    """
    def __init__(self, y_len: int = 3):
        super().__init__()
        self.y_len = y_len

        # Block 1: merge bid/ask pairs → (B, 32, T, 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        # Block 2: merge levels → (B, 32, T, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.Tanh(), nn.BatchNorm2d(32),
        )
        # Block 3: collapse to 1 → (B, 32, T, 1)   [key change: kernel (1,2)]
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(2, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1), padding=(1, 0)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )

        # Inception module (identical to original)
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )

        # LSTM + head (identical to original)
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1  = nn.Linear(64, y_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(1, x.size(0), 64, device=x.device)
        c0 = torch.zeros(1, x.size(0), 64, device=x.device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, x.shape[1], x.shape[2])

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        return self.fc1(x)

    def get_conv_params(self):
        """Parameters of the conv + inception layers (frozen during transfer)."""
        return (list(self.conv1.parameters()) +
                list(self.conv2.parameters()) +
                list(self.conv3.parameters()) +
                list(self.inp1.parameters()) +
                list(self.inp2.parameters()) +
                list(self.inp3.parameters()))

    def get_head_params(self):
        """LSTM + FC head parameters (fine-tuned during transfer learning)."""
        return list(self.lstm.parameters()) + list(self.fc1.parameters())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_loop(model, criterion, optimizer, train_loader, val_loader,
               epochs, save_path, device, desc="Epochs", patience=5, min_epochs=5):
    """Generic training loop with macro-F1 early stopping."""
    train_losses = []
    val_losses   = []
    val_macro_f1 = []
    best_val_loss = np.inf
    best_macro_f1 = -np.inf
    wait = 0

    for ep in tqdm(range(epochs), desc=desc):
        model.train()
        t0 = datetime.now()
        tr_loss = []
        for inputs, targets in train_loader:
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            tr_loss.append(loss.item())
        tr_loss = np.mean(tr_loss)

        model.eval()
        va_loss = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.long)
                logits = model(inputs)
                va_loss.append(criterion(logits, targets).item())
                preds = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(targets.cpu().numpy().tolist())
        va_loss = np.mean(va_loss)
        _, _, f1_vals, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )
        macro_f1 = float(np.mean(f1_vals))

        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))
        val_macro_f1.append(macro_f1)

        improved = (
            macro_f1 > best_macro_f1 + 1e-6 or
            (abs(macro_f1 - best_macro_f1) <= 1e-6 and va_loss < best_val_loss - 1e-6)
        )
        if improved:
            torch.save(model.state_dict(), save_path)
            best_val_loss = float(va_loss)
            best_macro_f1 = macro_f1
            wait = 0
        else:
            wait += 1

        dt = datetime.now() - t0
        print(f"{desc} {ep+1}/{epochs} | Train {tr_loss:.4f} | Val {va_loss:.4f} | "
              f"Val macro-F1 {macro_f1:.4f} | Best macro-F1 {best_macro_f1:.4f} | Δt {dt}")

        if ep + 1 >= min_epochs and wait >= patience:
            print(f"Early stopping at epoch {ep+1} (patience={patience}).")
            break

    return (
        np.asarray(train_losses, dtype=np.float32),
        np.asarray(val_losses, dtype=np.float32),
        np.asarray(val_macro_f1, dtype=np.float32),
    )


def evaluate(model, test_loader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float)
            probs  = torch.softmax(model(inputs), dim=1).cpu().numpy()
            y_true.extend(targets.numpy())
            y_pred.extend(probs.argmax(axis=1))
            y_probs.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def compute_metrics(y_true, y_pred):
    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                      average=None, labels=[0, 1, 2], zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred,
                        average="weighted", zero_division=0)
    return {"accuracy": acc, "kappa": kappa, "mcc": mcc, "f1": f1, "f1_w": f1_w}


def loader_kwargs(num_workers: int, use_pin_memory: bool) -> dict:
    kwargs = {"num_workers": num_workers, "pin_memory": use_pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_loss_curve(train_losses, val_losses, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train"); ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_transfer_comparison(results_before, results_after, save_path):
    """Bar chart comparing accuracy before/after transfer fine-tuning per stock."""
    stocks = [r["stock_id"] for r in results_before]
    acc_before = [r["metrics"]["accuracy"] for r in results_before]
    acc_after  = [r["metrics"]["accuracy"] for r in results_after]

    x = np.arange(len(stocks))
    fig, ax = plt.subplots(figsize=(max(10, len(stocks) * 0.6), 5))
    ax.bar(x - 0.2, acc_before, 0.4, label="Base model (frozen)")
    ax.bar(x + 0.2, acc_after,  0.4, label="Fine-tuned (transfer)")
    ax.set_xticks(x); ax.set_xticklabels([f"s{s}" for s in stocks], rotation=45, fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_title("Transfer Learning: Base vs Fine-tuned")
    ax.legend(); ax.axhline(1/3, color="gray", linestyle="--", label="random")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_transfer_regimes(base_results, transfer_results, specific_results, save_path):
    """Compare three regimes: zero-shot, fine-tuned, and specific-stock OOS."""
    base_map = {r["stock_id"]: r["metrics"]["accuracy"] for r in base_results}
    transfer_map = {r["stock_id"]: r["after"]["accuracy"] for r in transfer_results}
    specific_map = {r["stock_id"]: r["metrics"]["accuracy"] for r in specific_results}
    stocks = sorted(set(base_map) & set(transfer_map) & set(specific_map))
    if not stocks:
        return

    x = np.arange(len(stocks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(stocks) * 0.7), 5))
    ax.bar(x - width, [base_map[s] for s in stocks], width, label="Universal zero-shot")
    ax.bar(x, [transfer_map[s] for s in stocks], width, label="Fine-tuned transfer")
    ax.bar(x + width, [specific_map[s] for s in stocks], width, label="Specific-stock OOS")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{s}" for s in stocks], rotation=45, fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Transfer Regimes on Held-out Stocks")
    ax.axhline(1 / 3, color="gray", linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_cm_grid(results_list, save_path, title="Confusion Matrices"):
    n = len(results_list)
    if n == 0:
        return
    cols = min(5, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]
    for i, res in enumerate(results_list):
        cm  = confusion_matrix(res["y_true"], res["y_pred"])
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(cmn, display_labels=["D", "S", "U"]).plot(
            ax=axes[i], colorbar=False)
        axes[i].set_title(res.get("label", f"s{i}"), fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_workers = args.num_workers if args.num_workers >= 0 else (4 if torch.cuda.is_available() else 0)
    use_pin_memory = torch.cuda.is_available()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"DataLoader workers: {num_workers}")

    # Load stock split
    split_path = os.path.join(DATA_DIR, "stock_split.json")
    if not os.path.exists(split_path):
        print("ERROR: Stock split not found. Run scripts/prepare_optiver.py first.")
        sys.exit(1)

    with open(split_path) as f:
        split = json.load(f)
    train_ids    = split["train"]
    transfer_ids = split["transfer"]
    print(f"Train stocks: {len(train_ids)}  Transfer stocks: {len(transfer_ids)}")

    if args.max_transfer_stocks < len(transfer_ids):
        pick = np.linspace(0, len(transfer_ids) - 1, num=args.max_transfer_stocks, dtype=int)
        transfer_eval_ids = [transfer_ids[i] for i in pick]
    else:
        transfer_eval_ids = transfer_ids
    print(f"Transfer-learning study stocks: {transfer_eval_ids}")

    horizon_k = HORIZONS[args.horizon]
    T = args.lookback
    tag = f"k{horizon_k}"
    base_path = os.path.join(MODEL_DIR, f"optiver_base_{tag}.pt")
    base_state_path = os.path.join(MODEL_DIR, f"optiver_base_{tag}_state.pt")
    base_tmp_path = os.path.join(MODEL_DIR, f"optiver_base_{tag}.tmp")

    # =========================================================
    # PHASE 1: Base training on train_ids
    # =========================================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: Base Training  (horizon k={horizon_k}, T={T})")
    print(f"{'-'*60}")

    base_metrics_path = os.path.join(RESULT_DIR, f"base_metrics_{tag}.pkl")

    if not args.force and os.path.exists(base_path) and os.path.exists(base_metrics_path):
        print("  -> Skipping (results exist). Use --force to retrain.")
        with open(base_metrics_path, "rb") as f:
            base_metrics = pickle.load(f)
    else:
        print("  Building combined dataset from train stocks...")
        train_ds, val_ds = concat_datasets(
            train_ids,
            horizon_k,
            T,
            max_per_stock=args.base_max_per_stock,
        )
        if train_ds is None:
            print("ERROR: No training data found. Check data/optiver_processed/.")
            sys.exit(1)

        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=make_weighted_sampler(train_ds),
            **loader_kwargs(num_workers, use_pin_memory),
        )
        val_loader = data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_kwargs(num_workers, use_pin_memory),
        )

        model     = DeepLOBLite(y_len=3).to(device)
        class_weights, class_counts = make_class_weights(train_ds)
        print(f"  Base train class counts: {class_counts.tolist()}  weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        tr_losses, va_losses, va_macro_f1 = train_loop(
            model,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            args.epochs,
            base_tmp_path,
            device,
            desc="Base training",
            patience=args.patience,
            min_epochs=args.min_epochs,
        )

        # Load best checkpoint
        model.load_state_dict(torch.load(base_tmp_path, map_location=device))
        torch.save(model, base_path)
        os.replace(base_tmp_path, base_state_path)

        plot_loss_curve(tr_losses, va_losses,
                        f"Base Training Loss (k={horizon_k})",
                        os.path.join(RESULT_DIR, f"base_loss_{tag}.png"))

        # Evaluate on held-out transfer stocks
        print("\n  Evaluating base model on transfer stocks (zero-shot)...")
        base_metrics = {"per_stock": [], "horizon_k": horizon_k}
        cm_results   = []
        for sid in transfer_eval_ids:
            ds = load_stock_dataset(sid, horizon_k, T, max_samples=args.transfer_max_samples)
            if ds is None or len(ds) < 50:
                continue
            loader = data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )
            y_true, y_pred, y_probs = evaluate(model, loader, device)
            m = compute_metrics(y_true, y_pred)
            time_id = ds.time_id[ds.valid_end - 1]
            sample_end = ds.valid_end.copy()
            base_metrics["per_stock"].append({
                "stock_id": sid,
                "metrics": m,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_probs": y_probs,
                "time_id": time_id,
                "sample_end": sample_end,
            })
            cm_results.append({"label": f"stock {sid}", "y_true": y_true, "y_pred": y_pred})
            print(f"    stock {sid}: acc={m['accuracy']:.4f}  κ={m['kappa']:.4f}")

        with open(base_metrics_path, "wb") as f:
            pickle.dump(base_metrics, f)

        if cm_results:
            plot_cm_grid(cm_results, os.path.join(RESULT_DIR, f"cm_base_{tag}.png"),
                         title="Base Model on Transfer Stocks (zero-shot)")

        del train_ds, val_ds, train_loader, val_loader, model
        gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # =========================================================
    # PHASE 2: Transfer learning fine-tune on each transfer stock
    # =========================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: Transfer Learning Fine-tuning")
    print(f"{'-'*60}")

    transfer_path = os.path.join(RESULT_DIR, f"transfer_metrics_{tag}.pkl")

    if not args.force and os.path.exists(transfer_path):
        print("  -> Skipping (results exist). Use --force to retrain.")
        with open(transfer_path, "rb") as f:
            transfer_results = pickle.load(f)
    else:
        transfer_results = {"per_stock": [], "horizon_k": horizon_k}
        results_before   = []
        results_after    = []

        for sid in transfer_eval_ids:
            ds = load_stock_dataset(sid, horizon_k, T, max_samples=args.transfer_max_samples)
            if ds is None or len(ds) < 200:
                print(f"  stock {sid}: skip (insufficient data)")
                continue

            temporal = make_temporal_splits(ds)
            if temporal is None:
                print(f"  stock {sid}: skip (cannot form temporal split)")
                continue
            ft_train, ft_val, test_ds, n_ft, n_total = temporal

            ft_loader = data.DataLoader(
                ft_train,
                batch_size=min(32, len(ft_train)),
                sampler=make_weighted_sampler(ft_train),
                **loader_kwargs(num_workers, use_pin_memory),
            )
            ftv_loader = data.DataLoader(
                ft_val,
                batch_size=min(32, len(ft_val)),
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )
            test_loader = data.DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )

            # Load base model
            model = torch.load(base_path, map_location=device)

            # Evaluate BEFORE fine-tuning
            y_true_b, y_pred_b, y_prob_b = evaluate(model, test_loader, device)
            m_before = compute_metrics(y_true_b, y_pred_b)
            results_before.append({"stock_id": sid, "metrics": m_before})

            # Freeze conv + inception; train LSTM + FC only
            for param in model.parameters():
                param.requires_grad = False
            for param in model.get_head_params():
                param.requires_grad = True

            ft_class_weights, ft_class_counts = make_class_weights(ft_train)
            print(f"  stock {sid} fine-tune class counts: {ft_class_counts.tolist()}  weights: {ft_class_weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=ft_class_weights.to(device))
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.transfer_lr,
                weight_decay=args.weight_decay,
            )

            ft_save = os.path.join(MODEL_DIR, f"optiver_transfer_s{sid}_{tag}.pt")
            ft_model_path = os.path.join(MODEL_DIR, f"optiver_transfer_s{sid}_{tag}_model.pt")
            train_loop(model, criterion, optimizer,
                       ft_loader, ftv_loader,
                       args.transfer_epochs, ft_save, device,
                       desc=f"TL stock {sid}",
                       patience=min(args.patience, 3),
                       min_epochs=min(args.min_epochs, 3))

            # Load best fine-tuned checkpoint
            model.load_state_dict(torch.load(ft_save, map_location=device))
            torch.save(model, ft_model_path)

            # Evaluate AFTER fine-tuning
            y_true_a, y_pred_a, y_prob_a = evaluate(model, test_loader, device)
            m_after = compute_metrics(y_true_a, y_pred_a)
            results_after.append({"stock_id": sid, "metrics": m_after})

            test_end = ds.valid_end[n_ft:n_total].copy()
            test_time_id = ds.time_id[test_end - 1]

            print(f"  stock {sid}: "
                  f"acc before={m_before['accuracy']:.4f} → after={m_after['accuracy']:.4f} "
                  f"(Δ={m_after['accuracy']-m_before['accuracy']:+.4f})")

            transfer_results["per_stock"].append({
                "stock_id": sid,
                "before": m_before,
                "after":  m_after,
                "y_true": y_true_a,
                "y_pred_before": y_pred_b,
                "y_pred_after": y_pred_a,
                "y_probs_before": y_prob_b,
                "y_probs_after": y_prob_a,
                "time_id": test_time_id,
                "sample_end": test_end,
            })

            del model, test_ds
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        with open(transfer_path, "wb") as f:
            pickle.dump(transfer_results, f)

        if results_before and results_after:
            plot_transfer_comparison(
                results_before, results_after,
                os.path.join(RESULT_DIR, f"transfer_comparison_{tag}.png"))

    # =========================================================
    # PHASE 3: Specific-stock out-of-sample training
    # =========================================================
    print(f"\n{'='*60}")
    print("PHASE 3: Specific-stock Temporal Out-of-sample")
    print(f"{'-'*60}")

    specific_path = os.path.join(RESULT_DIR, f"specific_metrics_{tag}.pkl")
    if not args.force and os.path.exists(specific_path):
        print("  -> Skipping (results exist). Use --force to retrain.")
        with open(specific_path, "rb") as f:
            specific_results = pickle.load(f)
    else:
        specific_results = {"per_stock": [], "horizon_k": horizon_k}
        for sid in transfer_eval_ids:
            ds = load_stock_dataset(sid, horizon_k, T, max_samples=args.transfer_max_samples)
            if ds is None or len(ds) < 200:
                print(f"  stock {sid}: skip (insufficient data)")
                continue

            temporal = make_temporal_splits(ds)
            if temporal is None:
                print(f"  stock {sid}: skip (cannot form temporal split)")
                continue
            train_ds, val_ds, test_ds, n_prefix, _ = temporal

            train_loader = data.DataLoader(
                train_ds,
                batch_size=min(args.batch_size, len(train_ds)),
                sampler=make_weighted_sampler(train_ds),
                **loader_kwargs(num_workers, use_pin_memory),
            )
            val_loader = data.DataLoader(
                val_ds,
                batch_size=min(args.batch_size, len(val_ds)),
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )
            test_loader = data.DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )

            model = DeepLOBLite(y_len=3).to(device)
            sp_class_weights, sp_class_counts = make_class_weights(train_ds)
            print(f"  stock {sid} specific-train class counts: {sp_class_counts.tolist()}  weights: {sp_class_weights.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=sp_class_weights.to(device))
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            save_path = os.path.join(MODEL_DIR, f"optiver_specific_s{sid}_{tag}.pt")

            train_loop(
                model,
                criterion,
                optimizer,
                train_loader,
                val_loader,
                args.specific_epochs,
                save_path,
                device,
                desc=f"Specific stock {sid}",
                patience=args.patience,
                min_epochs=args.min_epochs,
            )

            model.load_state_dict(torch.load(save_path, map_location=device))
            y_true, y_pred, y_probs = evaluate(model, test_loader, device)
            metrics = compute_metrics(y_true, y_pred)
            test_end = ds.valid_end[n_prefix:].copy()
            test_time_id = ds.time_id[test_end - 1]

            specific_results["per_stock"].append({
                "stock_id": sid,
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_probs": y_probs,
                "time_id": test_time_id,
                "sample_end": test_end,
            })
            print(f"  stock {sid}: specific-stock acc={metrics['accuracy']:.4f}  κ={metrics['kappa']:.4f}")

            del model, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        with open(specific_path, "wb") as f:
            pickle.dump(specific_results, f)

    plot_transfer_regimes(
        base_metrics.get("per_stock", []),
        transfer_results.get("per_stock", []),
        specific_results.get("per_stock", []),
        os.path.join(RESULT_DIR, f"transfer_regimes_{tag}.png"),
    )

    # =========================================================
    # Print summary
    # =========================================================
    print(f"\n{'='*60}")
    print("  Optiver Transfer Learning Summary")
    print(f"{'='*60}")
    rows = []
    for r in transfer_results["per_stock"]:
        rows.append({
            "Stock": r["stock_id"],
            "Acc Before": f"{r['before']['accuracy']:.4f}",
            "Acc After":  f"{r['after']['accuracy']:.4f}",
            "Δ Acc":      f"{r['after']['accuracy']-r['before']['accuracy']:+.4f}",
            "κ After":    f"{r['after']['kappa']:.4f}",
        })
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    specific_rows = []
    for r in specific_results["per_stock"]:
        specific_rows.append({
            "Stock": r["stock_id"],
            "Specific OOS Acc": f"{r['metrics']['accuracy']:.4f}",
            "Specific OOS κ": f"{r['metrics']['kappa']:.4f}",
        })
    if specific_rows:
        df_specific = pd.DataFrame(specific_rows)
        print("\nSpecific-stock temporal OOS:")
        print(df_specific.to_string(index=False))

    print(f"\nAll results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
