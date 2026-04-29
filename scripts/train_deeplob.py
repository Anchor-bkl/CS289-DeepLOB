#!/usr/bin/env python3
"""
train_deeplob.py — Standalone GPU training script for DeepLOB (FI-2010)
=======================================================================
Trains one classification DeepLOB model per prediction horizon.

The active FI workflow keeps the original five paper horizons and only the
3-way classification objective. Results are saved to RESULT_DIR so downstream
analysis can load them.

Usage
-----
    python scripts/train_deeplob.py [--epochs 100] [--batch-size 32] [--lr 1e-3]

Outputs (all under results/)
--------
    models/deeplob_k{idx}.pt          — best-val model (full model object)
    results/preds_k{idx}.npz          — predictions on test set
    results/losses_k{idx}.npz         — train/val loss and monitor metadata
    results/loss_k{idx}.png           — loss curve plot
    results/cm_k{idx}.png             — confusion matrix
    results/all_results.pkl           — dict with all metrics per horizon
    results/performance_summary.csv   — classification summary table
"""
import argparse
import gc
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

FI_HORIZON_EVENTS = [10, 20, 30, 50, 100]
CLASSIFICATION_MONITORS = ["val_acc", "val_loss", "val_macro_f1", "val_mcc"]
PLOT_STYLE = "seaborn-v0_8-whitegrid"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.environ.get("FI_MODEL_DIR", os.path.join(BASE_DIR, "models"))
RESULT_DIR = os.environ.get("FI_RESULT_DIR", os.path.join(BASE_DIR, "results"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepLOB on FI-2010 dataset")
    p.add_argument("--epochs", type=int, default=100, help="Max training epochs (default: 100)")
    p.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--lookback", type=int, default=100, help="LOB lookback window T (default: 100)")
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience (default: 20)")
    p.add_argument("--min-epochs", type=int, default=20, help="Minimum epochs before early stopping (default: 20)")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Adam weight decay regularization (default: 1e-4)")
    p.add_argument("--dropout", type=float, default=0.2, help="Dropout before classifier head (default: 0.2)")
    p.add_argument("--grad-clip", type=float, default=None,
                   help="Optional gradient clipping max norm; omitted uses the selected profile default")
    p.add_argument("--label-smoothing", type=float, default=None,
                   help="Optional cross-entropy label smoothing; omitted uses the selected profile default")
    p.add_argument("--class-weight-mode", choices=["none", "balanced_sqrt"], default=None,
                   help="Optional class weighting; omitted uses the selected profile default")
    p.add_argument("--monitor", choices=CLASSIFICATION_MONITORS, default=None,
                   help="Optional checkpoint / early-stop monitor; omitted uses the selected profile default")
    p.add_argument("--horizon-profile", choices=["legacy", "paper", "adaptive"], default="paper",
                   help="Use a shared paper-like setup or per-horizon tuned defaults before explicit CLI overrides")
    p.add_argument("--horizons", type=int, nargs="*", default=[0, 1, 2, 3, 4], choices=[0, 1, 2, 3, 4],
                   help="0-based horizon indices to retrain. Defaults to the original 5 paper horizons.")
    p.add_argument("--k4-lr", type=float, default=7e-4,
                   help="Tuned learning rate for k_idx=4 / 100-event horizon")
    p.add_argument("--k4-weight-decay", type=float, default=3e-4,
                   help="Tuned weight decay for k_idx=4 / 100-event horizon")
    p.add_argument("--k4-dropout", type=float, default=0.30,
                   help="Tuned classifier dropout for k_idx=4 / 100-event horizon")
    p.add_argument("--k4-patience", type=int, default=12,
                   help="Tuned early-stopping patience for k_idx=4 / 100-event horizon")
    p.add_argument("--k4-min-epochs", type=int, default=8,
                   help="Tuned minimum epochs for k_idx=4 / 100-event horizon")
    p.add_argument("--k4-label-smoothing", type=float, default=0.02,
                   help="Tuned label smoothing for k_idx=4 / 100-event horizon")
    p.add_argument("--k4-grad-clip", type=float, default=1.0,
                   help="Tuned gradient clipping max norm for k_idx=4 / 100-event horizon")
    p.add_argument("--k4-monitor", choices=CLASSIFICATION_MONITORS, default="val_loss",
                   help="Checkpoint / early-stop monitor for k_idx=4 / 100-event horizon")
    p.add_argument("--force", action="store_true", help="Retrain even if results exist")
    return p.parse_args()


def horizon_config(args, k_idx):
    cfg = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "monitor": "val_loss",
        "label_smoothing": 0.0,
        "class_weight_mode": "none",
        "grad_clip": 0.0,
    }
    if args.horizon_profile == "paper":
        cfg.update({
            "monitor": "val_acc",
            "label_smoothing": 0.0,
            "class_weight_mode": "none",
            "grad_clip": 0.0,
        })
    elif args.horizon_profile == "adaptive":
        adaptive = {
            0: {
                "lr": 6e-4,
                "weight_decay": 3e-4,
                "dropout": 0.30,
                "patience": 18,
                "min_epochs": 14,
                "monitor": "val_macro_f1",
                "label_smoothing": 0.02,
                "class_weight_mode": "balanced_sqrt",
                "grad_clip": 0.75,
            },
            1: {
                "lr": 7e-4,
                "weight_decay": 3e-4,
                "dropout": 0.25,
                "patience": 18,
                "min_epochs": 14,
                "monitor": "val_macro_f1",
                "label_smoothing": 0.02,
                "class_weight_mode": "balanced_sqrt",
                "grad_clip": 0.75,
            },
            2: {
                "lr": 8e-4,
                "weight_decay": 1e-4,
                "dropout": 0.20,
                "patience": 14,
                "min_epochs": 10,
                "monitor": "val_mcc",
                "label_smoothing": 0.0,
                "class_weight_mode": "none",
                "grad_clip": 1.0,
            },
            3: {
                "lr": 7e-4,
                "weight_decay": 2e-4,
                "dropout": 0.25,
                "patience": 10,
                "min_epochs": 8,
                "monitor": "val_loss",
                "label_smoothing": 0.0,
                "class_weight_mode": "none",
                "grad_clip": 1.0,
            },
            4: {
                "lr": 7e-4,
                "weight_decay": 3e-4,
                "dropout": 0.30,
                "patience": 12,
                "min_epochs": 8,
                "monitor": "val_loss",
                "label_smoothing": 0.02,
                "class_weight_mode": "none",
                "grad_clip": 1.0,
            },
        }
        cfg.update(adaptive[k_idx])
    if args.horizon_profile == "adaptive" and k_idx == 4:
        cfg.update({
            "lr": args.k4_lr,
            "weight_decay": args.k4_weight_decay,
            "dropout": args.k4_dropout,
            "patience": args.k4_patience,
            "min_epochs": args.k4_min_epochs,
            "monitor": args.k4_monitor,
            "label_smoothing": args.k4_label_smoothing,
            "grad_clip": args.k4_grad_clip,
        })

    explicit_overrides = {
        "monitor": args.monitor,
        "label_smoothing": args.label_smoothing,
        "class_weight_mode": args.class_weight_mode,
        "grad_clip": args.grad_clip,
    }
    for key, value in explicit_overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def make_class_weights(labels: torch.Tensor, num_classes: int = 3) -> tuple[torch.Tensor, np.ndarray]:
    counts = np.bincount(labels.cpu().numpy(), minlength=num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    weights = np.sqrt(safe_counts.sum() / safe_counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.astype(np.int64)


def make_classification_criterion(label_smoothing, class_weights=None):
    try:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    except TypeError:
        if label_smoothing:
            print("WARNING: installed PyTorch does not support label_smoothing; using plain CE.")
        return nn.CrossEntropyLoss(weight=class_weights)


def load_data():
    dec_data = np.loadtxt(os.path.join(DATA_DIR, "Train_Dst_NoAuction_DecPre_CF_7.txt"))
    split_col = int(np.floor(dec_data.shape[1] * 0.8))
    dec_train = dec_data[:, :split_col]
    dec_val = dec_data[:, split_col:]

    t1 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_7.txt"))
    t2 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_8.txt"))
    t3 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_9.txt"))
    dec_test = np.hstack((t1, t2, t3))

    print(f"Train: {dec_train.shape}  Val: {dec_val.shape}  Test: {dec_test.shape}")
    return dec_train, dec_val, dec_test


def prepare_x(raw):
    return np.array(raw[:40, :].T)


def get_label(raw):
    return np.array(raw[-5:, :].T)


def data_classification(x, y, lookback):
    n_rows, n_features = x.shape
    data_y = y[lookback - 1:n_rows]
    data_x = np.zeros((n_rows - lookback + 1, lookback, n_features), dtype=x.dtype)
    for idx in range(lookback, n_rows + 1):
        data_x[idx - lookback] = x[idx - lookback:idx, :]
    return data_x, data_y


class LOBDataset(data.Dataset):
    """FI-2010 LOB dataset for a given prediction horizon index."""

    def __init__(self, raw_data, k, lookback=100):
        x = prepare_x(raw_data).astype(np.float32)
        y = get_label(raw_data)
        x, y = data_classification(x, y, lookback)
        y = y[:, k] - 1

        self.x = torch.unsqueeze(torch.from_numpy(x), 1)
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))
        self.length = len(self.x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DeepLOB(nn.Module):
    """DeepLOB: CNN + Inception + LSTM for LOB mid-price classification."""

    def __init__(self, y_len=3, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64, y_len)

    def forward(self, x):
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
        x = self.dropout(x)
        return self.fc1(x)


def batch_gd(model, criterion, optimizer, train_loader, val_loader,
             epochs, model_path, device, patience=20, min_epochs=20,
             monitor="val_acc", grad_clip=0.0):
    train_losses = []
    val_losses = []
    val_accs = []
    val_macro_f1s = []
    val_mccs = []
    best_val_loss = np.inf
    best_val_acc = -np.inf
    best_val_epoch = -1
    wait = 0

    for ep in tqdm(range(epochs), desc="Epochs"):
        model.train()
        t0 = datetime.now()
        batch_losses = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.int64)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        model.eval()
        val_batch_losses = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.int64)
                logits = model(inputs)
                loss = criterion(logits, targets)
                preds = logits.argmax(dim=1)
                val_batch_losses.append(loss.item())
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        val_loss = float(np.mean(val_batch_losses))
        val_acc = float(accuracy_score(y_true, y_pred))
        _, _, f1_vals, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0
        )
        val_macro_f1 = float(np.mean(f1_vals))
        val_mcc = float(matthews_corrcoef(y_true, y_pred))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_macro_f1s.append(val_macro_f1)
        val_mccs.append(val_mcc)

        if monitor == "val_loss":
            improved = (
                val_loss < best_val_loss - 1e-6 or
                (abs(val_loss - best_val_loss) <= 1e-6 and val_acc > best_val_acc + 1e-6)
            )
        elif monitor == "val_macro_f1":
            best_metric = max(val_macro_f1s[:-1], default=-np.inf)
            improved = (
                val_macro_f1 > best_metric + 1e-6 or
                (abs(val_macro_f1 - best_metric) <= 1e-6 and val_loss < best_val_loss - 1e-6)
            )
        elif monitor == "val_mcc":
            best_metric = max(val_mccs[:-1], default=-np.inf)
            improved = (
                val_mcc > best_metric + 1e-6 or
                (abs(val_mcc - best_metric) <= 1e-6 and val_loss < best_val_loss - 1e-6)
            )
        else:
            improved = (
                val_acc > best_val_acc + 1e-6 or
                (abs(val_acc - best_val_acc) <= 1e-6 and val_loss < best_val_loss - 1e-6)
            )

        if improved:
            torch.save(model, model_path)
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_epoch = ep
            wait = 0
            print(
                f"  [Epoch {ep+1}] Saved ({monitor}; val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}, val_macro_f1={val_macro_f1:.4f}, val_mcc={val_mcc:.4f})"
            )
        else:
            wait += 1

        dt = datetime.now() - t0
        print(
            f"Epoch {ep+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | Val macro-F1: {val_macro_f1:.4f} | "
            f"Val MCC: {val_mcc:.4f} | Best ep: {best_val_epoch+1} | Δt: {dt}"
        )

        if ep + 1 >= min_epochs and wait >= patience:
            print(f"Early stopping triggered at epoch {ep+1} (patience={patience}).")
            break

    return (
        np.array(train_losses, dtype=np.float32),
        np.array(val_losses, dtype=np.float32),
        np.array(val_accs, dtype=np.float32),
        np.array(val_macro_f1s, dtype=np.float32),
        np.array(val_mccs, dtype=np.float32),
        best_val_epoch + 1,
        float(best_val_acc),
        float(best_val_loss),
    )


def evaluate_model(model_path, test_loader, device):
    model = torch.load(model_path, map_location=device)
    model.eval()

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            y_true.extend(targets.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def style_axis(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fbfbfd")
    if grid:
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.7)
    else:
        ax.grid(False)


def plot_loss(train_losses, val_losses, val_accs, val_macro_f1s, val_mccs,
              k_label, save_path, best_epoch=None, monitor=None):
    epochs = np.arange(1, len(train_losses) + 1)
    best_epoch = int(best_epoch) if best_epoch is not None else int(np.argmin(val_losses) + 1)
    best_epoch = max(1, min(best_epoch, len(epochs)))
    best_idx = best_epoch - 1
    zoom_start = max(1, min(best_epoch, len(epochs)) - 3)
    zoom_start = min(zoom_start, max(1, len(epochs) - max(10, len(epochs) // 2) + 1))
    zoom_mask = epochs >= zoom_start

    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.0), constrained_layout=True)
        loss_ax = axes[0, 0]
        zoom_ax = axes[0, 1]
        metric_ax = axes[1, 0]
        gap_ax = axes[1, 1]

        loss_ax.plot(epochs, train_losses, label="Train loss", linewidth=2.2, color="#355070")
        loss_ax.plot(epochs, val_losses, label="Val loss", linewidth=2.2, color="#e56b6f")
        loss_ax.axvline(best_epoch, color="#2a9d8f", linestyle="--", linewidth=1.4,
                        label=f"Best epoch ({monitor or 'monitor'})")
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_ax.set_title(f"Loss Curves — {k_label}")
        loss_ax.legend(fontsize=9, frameon=True)
        style_axis(loss_ax)

        zoom_train = train_losses[zoom_mask]
        zoom_val = val_losses[zoom_mask]
        zoom_epochs = epochs[zoom_mask]
        zoom_ax.plot(zoom_epochs, zoom_train, label="Train loss", linewidth=2.0, color="#355070")
        zoom_ax.plot(zoom_epochs, zoom_val, label="Val loss", linewidth=2.0, color="#e56b6f")
        zoom_ax.axvline(best_epoch, color="#2a9d8f", linestyle="--", linewidth=1.4)
        zoom_lo = float(min(np.min(zoom_train), np.min(zoom_val)))
        zoom_hi = float(max(np.max(zoom_train), np.max(zoom_val)))
        zoom_pad = max(0.01, 0.08 * (zoom_hi - zoom_lo if zoom_hi > zoom_lo else (zoom_hi if zoom_hi else 1.0)))
        zoom_ax.set_ylim(zoom_lo - zoom_pad, zoom_hi + zoom_pad)
        zoom_ax.set_xlabel("Epoch")
        zoom_ax.set_ylabel("Loss")
        zoom_ax.set_title("Late-stage zoom")
        style_axis(zoom_ax)

        metric_items = []
        if len(val_accs):
            metric_ax.plot(epochs, val_accs, linewidth=2.0, color="#277da1", label="Val acc")
            metric_items.append(("Val acc", float(val_accs[best_idx])))
        if len(val_macro_f1s):
            metric_ax.plot(epochs, val_macro_f1s, linewidth=2.0, color="#f4a261", label="Val macro-F1")
            metric_items.append(("Val macro-F1", float(val_macro_f1s[best_idx])))
        if len(val_mccs):
            metric_ax.plot(epochs, val_mccs, linewidth=2.0, color="#6d597a", label="Val MCC")
            metric_items.append(("Val MCC", float(val_mccs[best_idx])))
        metric_ax.axvline(best_epoch, color="#2a9d8f", linestyle="--", linewidth=1.4)
        metric_ax.set_xlabel("Epoch")
        metric_ax.set_ylabel("Metric value")
        metric_ax.set_title("Validation metrics")
        metric_ax.legend(fontsize=9, frameon=True, loc="lower right")
        style_axis(metric_ax)

        gap = val_losses - train_losses
        gap_ax.plot(epochs, gap, color="#c1121f", linewidth=2.0)
        gap_ax.axhline(0.0, color="#3d405b", linewidth=1.0, linestyle=":")
        gap_ax.axvline(best_epoch, color="#2a9d8f", linestyle="--", linewidth=1.4)
        gap_ax.set_xlabel("Epoch")
        gap_ax.set_ylabel("Val - Train")
        gap_ax.set_title(f"Generalization gap | final={gap[-1]:+.4f}")
        style_axis(gap_ax)

        summary_lines = [
            f"Monitor: {monitor or 'monitor'}",
            f"Best epoch: {best_epoch}",
            f"Best val loss: {val_losses[best_idx]:.4f}",
        ]
        for name, value in metric_items:
            summary_lines.append(f"{name}: {value:.4f}")
        gap_ax.text(
            0.03,
            0.97,
            "\n".join(summary_lines),
            transform=gap_ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#d9d9e3", "alpha": 0.95},
        )

        fig.suptitle(f"DeepLOB Training Diagnostics — {k_label}", fontsize=15, fontweight="bold")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_cm(y_true, y_pred, k_label, save_path, acc=None, kappa=None, mcc=None):
    class_names = ["Down", "Stationary", "Up"]
    cm_val = confusion_matrix(y_true, y_pred)
    cm_norm = np.divide(
        cm_val.astype(float),
        cm_val.sum(axis=1, keepdims=True),
        out=np.zeros_like(cm_val, dtype=float),
        where=cm_val.sum(axis=1, keepdims=True) != 0,
    )
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
        ConfusionMatrixDisplay(cm_val, display_labels=class_names).plot(
            ax=axes[0], colorbar=False, cmap="Blues", values_format="d"
        )
        ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(
            ax=axes[1], colorbar=False, cmap="YlGnBu", values_format=".2f"
        )
        axes[0].set_title("Counts")
        axes[1].set_title("Row-normalised")
        for ax in axes:
            style_axis(ax, grid=False)
        summary = None
        if acc is not None and kappa is not None and mcc is not None:
            summary = f"Acc={acc:.4f} | Cohen κ={kappa:.4f} | MCC={mcc:.4f}"
        fig.suptitle(f"Confusion Matrix — {k_label}", fontsize=15, fontweight="bold")
        if summary:
            fig.text(0.5, 0.01, summary, ha="center", va="bottom", fontsize=10, color="#3d405b")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n--- Loading FI-2010 data ---")
    dec_train, dec_val, dec_test = load_data()

    k_values = [0, 1, 2, 3, 4]
    k_labels = ["k=1 (10 ev)", "k=2 (20 ev)", "k=3 (30 ev)", "k=4 (50 ev)", "k=5 (100 ev)"]
    retrain_horizons = set(args.horizons)
    print(f"Training profile: {args.horizon_profile}")
    print(f"Retrain horizons: {sorted(retrain_horizons)}")

    all_results = {}

    for k_idx, k_label in zip(k_values, k_labels):
        print(f"\n{'=' * 60}")
        print(f"Horizon {k_label}  (k_idx={k_idx})")
        print(f"{'-' * 60}")

        pred_file = os.path.join(RESULT_DIR, f"preds_k{k_idx}.npz")
        loss_file = os.path.join(RESULT_DIR, f"losses_k{k_idx}.npz")
        model_path = os.path.join(MODEL_DIR, f"deeplob_k{k_idx}.pt")
        cfg = horizon_config(args, k_idx)
        should_train = (
            k_idx in retrain_horizons and
            (args.force or not (os.path.exists(pred_file) and os.path.exists(model_path) and os.path.exists(loss_file)))
        )

        if not should_train and os.path.exists(pred_file) and os.path.exists(model_path) and os.path.exists(loss_file):
            print("  -> Already done, loading from disk.")
            pdata = np.load(pred_file)
            y_true = pdata["y_true"]
            y_pred = pdata["y_pred"]
            y_probs = pdata["y_probs"] if "y_probs" in pdata else np.array([], dtype=np.float32)

            loss_data = np.load(loss_file)
            train_losses = loss_data["train"]
            val_losses = loss_data["val"]
            val_accs = loss_data["val_acc"] if "val_acc" in loss_data else np.array([], dtype=np.float32)
            val_macro_f1s = loss_data["val_macro_f1"] if "val_macro_f1" in loss_data else np.array([], dtype=np.float32)
            val_mccs = loss_data["val_mcc"] if "val_mcc" in loss_data else np.array([], dtype=np.float32)
            best_epoch = int(loss_data["best_epoch"]) if "best_epoch" in loss_data else int(np.argmin(val_losses) + 1)
            best_val_acc = float(loss_data["best_val_acc"]) if "best_val_acc" in loss_data else float("nan")
            best_val_loss = float(loss_data["best_val_loss"]) if "best_val_loss" in loss_data else float(np.min(val_losses))
            monitor = str(loss_data["monitor"]) if "monitor" in loss_data else cfg["monitor"]
            profile = str(loss_data["profile"]) if "profile" in loss_data else args.horizon_profile
        elif k_idx not in retrain_horizons:
            print("  -> Missing artifacts and horizon was not selected for training; skipping.")
            continue
        else:
            print(
                "  -> Training with "
                f"lr={cfg['lr']} weight_decay={cfg['weight_decay']} dropout={cfg['dropout']} "
                f"label_smoothing={cfg['label_smoothing']} monitor={cfg['monitor']} "
                f"class_weight_mode={cfg['class_weight_mode']} grad_clip={cfg['grad_clip']}"
            )

            ds_train = LOBDataset(dec_train, k=k_idx, lookback=args.lookback)
            ds_val = LOBDataset(dec_val, k=k_idx, lookback=args.lookback)
            ds_test = LOBDataset(dec_test, k=k_idx, lookback=args.lookback)

            train_loader = data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                           num_workers=4, pin_memory=True)
            val_loader = data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)
            test_loader = data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                                          num_workers=4, pin_memory=True)

            model = DeepLOB(y_len=3, dropout=cfg["dropout"]).to(device)

            class_weights = None
            class_counts = None
            if cfg["class_weight_mode"] == "balanced_sqrt":
                class_weights, class_counts = make_class_weights(ds_train.y)
                class_weights = class_weights.to(device)
                print(f"  Class counts: {class_counts.tolist()}  loss_weights: {class_weights.detach().cpu().tolist()}")

            criterion = make_classification_criterion(cfg["label_smoothing"], class_weights=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

            (
                train_losses,
                val_losses,
                val_accs,
                val_macro_f1s,
                val_mccs,
                best_epoch,
                best_val_acc,
                best_val_loss,
            ) = batch_gd(
                model,
                criterion,
                optimizer,
                train_loader,
                val_loader,
                args.epochs,
                model_path,
                device,
                patience=cfg["patience"],
                min_epochs=cfg["min_epochs"],
                monitor=cfg["monitor"],
                grad_clip=cfg["grad_clip"],
            )
            monitor = cfg["monitor"]
            profile = args.horizon_profile

            y_true, y_pred, y_probs = evaluate_model(model_path, test_loader, device)

            np.savez_compressed(
                loss_file,
                train=train_losses,
                val=val_losses,
                train_cls=train_losses,
                val_cls=val_losses,
                val_acc=val_accs,
                val_macro_f1=val_macro_f1s,
                val_mcc=val_mccs,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                best_val_loss=best_val_loss,
                monitor=monitor,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                dropout=cfg["dropout"],
                label_smoothing=cfg["label_smoothing"],
                grad_clip=cfg["grad_clip"],
                class_weight_mode=cfg["class_weight_mode"],
                profile=profile,
            )
            np.savez_compressed(pred_file, y_true=y_true, y_pred=y_pred, y_probs=y_probs)

            del ds_train, ds_val, ds_test, train_loader, val_loader, test_loader, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        all_results[k_idx] = {
            "label": k_label,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "val_macro_f1s": val_macro_f1s,
            "val_mccs": val_mccs,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "monitor": monitor,
            "profile": profile,
            "accuracy": acc,
            "kappa": kappa,
            "mcc": mcc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
            "precision_w": prec_w,
            "recall_w": rec_w,
            "f1_w": f1_w,
        }

        print(classification_report(
            y_true,
            y_pred,
            target_names=["Down", "Stationary", "Up"],
            digits=4,
        ))

        plot_loss(
            train_losses,
            val_losses,
            val_accs,
            val_macro_f1s,
            val_mccs,
            k_label,
            os.path.join(RESULT_DIR, f"loss_k{k_idx}.png"),
            best_epoch=best_epoch,
            monitor=monitor,
        )
        plot_cm(
            y_true,
            y_pred,
            k_label,
            os.path.join(RESULT_DIR, f"cm_k{k_idx}.png"),
            acc=acc,
            kappa=kappa,
            mcc=mcc,
        )

    with open(os.path.join(RESULT_DIR, "all_results.pkl"), "wb") as file_obj:
        pickle.dump(all_results, file_obj)

    rows = []
    for k_idx in k_values:
        if k_idx not in all_results:
            continue
        record = all_results[k_idx]
        rows.append({
            "Horizon": record["label"],
            "Profile": record["profile"],
            "Accuracy": record["accuracy"],
            "Cohen κ": record["kappa"],
            "MCC": record["mcc"],
            "F1-Down": record["f1"][0],
            "F1-Stat": record["f1"][1],
            "F1-Up": record["f1"][2],
            "F1-Weighted": record["f1_w"],
            "Best Epoch": record["best_epoch"],
            "Best Val Loss": record["best_val_loss"],
            "Monitor": record["monitor"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "performance_summary.csv"), index=False, float_format="%.4f")
    print("\n" + "=" * 65)
    print("  DeepLOB Classification Training Complete — Performance Summary")
    print("=" * 65)
    print(df.to_string(index=False))
    print(f"\nAll results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
