#!/usr/bin/env python3
"""
train_deeplob.py — Standalone GPU training script for DeepLOB (FI-2010)
=======================================================================
Trains one DeepLOB model per prediction horizon (k = 1..5).
Results are saved to RESULT_DIR so the report notebook can load them.

Usage
-----
    python scripts/train_deeplob.py [--epochs 100] [--batch-size 32] [--lr 1e-3]

Outputs (all under results/)
--------
    models/deeplob_k{idx}.pt          — best-val model (full model object)
    results/preds_k{idx}.npz          — y_true, y_pred, y_probs on test set
    results/losses_k{idx}.npz         — train/val loss, val accuracy, early-stop metadata
    results/loss_k{idx}.png           — loss curve plot
    results/cm_k{idx}.png             — confusion matrix (counts + normalised)
    results/all_results.pkl           — dict with all metrics per horizon
    results/performance_summary.csv   — accuracy / κ / MCC / F1 table
"""
import argparse
import gc
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train DeepLOB on FI-2010 dataset")
    p.add_argument("--epochs",     type=int,   default=100,  help="Max training epochs (default: 100)")
    p.add_argument("--batch-size", type=int,   default=32,   help="Mini-batch size (default: 32)")
    p.add_argument("--lr",         type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--lookback",   type=int,   default=100,  help="LOB lookback window T (default: 100)")
    p.add_argument("--patience",   type=int,   default=20,   help="Early-stopping patience on val accuracy (default: 20)")
    p.add_argument("--min-epochs", type=int,   default=20,   help="Minimum epochs before early stopping (default: 20)")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Adam weight decay regularization (default: 1e-4)")
    p.add_argument("--dropout",    type=float, default=0.2,  help="Dropout before classifier head (default: 0.2)")
    p.add_argument("--force",      action="store_true",      help="Retrain even if results exist")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------
def load_data():
    """Load FI-2010 train / val / test splits."""
    dec_data  = np.loadtxt(os.path.join(DATA_DIR, "Train_Dst_NoAuction_DecPre_CF_7.txt"))
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
    dec_val   = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

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

def data_classification(X, Y, T):
    N, D = X.shape
    dataY = Y[T - 1:N]
    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = X[i - T:i, :]
    return dataX, dataY


class LOBDataset(data.Dataset):
    """FI-2010 LOB dataset for a given prediction horizon k."""
    def __init__(self, raw_data, k, num_classes=3, T=100):
        x = prepare_x(raw_data).astype(np.float32)
        y = get_label(raw_data)
        x, y = data_classification(x, y, T)
        y = y[:, k] - 1            # 0=down, 1=stat, 2=up

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)   # (N, 1, T, 40)
        self.y = torch.from_numpy(y)
        self.length = len(x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# DeepLOB model
# ---------------------------------------------------------------------------
class DeepLOB(nn.Module):
    """DeepLOB: CNN + Inception + LSTM for LOB mid-price prediction.

    Architecture follows Zhang et al. (2019):
      Input (B, 1, T, 40) → 3 Conv blocks → Inception → LSTM → FC(3)
    """
    def __init__(self, y_len=3, dropout=0.2):
        super().__init__()
        self.y_len = y_len

        # Block 1 — price-volume level extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,  32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        # Block 2 — level aggregation
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(), nn.BatchNorm2d(32),
        )
        # Block 3 — combine all 10 levels into 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )

        # Inception module (multi-scale temporal features)
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

        # LSTM for sequence modelling
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1  = nn.Linear(64, y_len)

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
        return self.fc1(x)   # logits; softmax is applied only for inference


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def batch_gd(model, criterion, optimizer, train_loader, val_loader,
             epochs, model_path, device, patience=20, min_epochs=20):
    """Training with validation monitoring, checkpointing, and early stopping."""
    train_losses = []
    val_losses   = []
    val_accs     = []
    best_val_loss  = np.inf
    best_val_acc   = -np.inf
    best_val_epoch = -1
    wait = 0

    for ep in tqdm(range(epochs), desc="Epochs"):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)

        model.eval()
        val_loss = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.int64)
                logits = model(inputs)
                val_loss.append(criterion(logits, targets).item())
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        val_loss = np.mean(val_loss)
        val_acc = val_correct / max(val_total, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        improved = (
            val_acc > best_val_acc + 1e-6 or
            (abs(val_acc - best_val_acc) <= 1e-6 and val_loss < best_val_loss - 1e-6)
        )
        if improved:
            torch.save(model, model_path)
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_val_epoch = ep
            wait = 0
            print(f"  [Epoch {ep+1}] Saved (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            wait += 1

        dt = datetime.now() - t0
        print(f"Epoch {ep+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Val acc: {val_acc:.4f} | Best ep: {best_val_epoch+1} | Δt: {dt}")

        if ep + 1 >= min_epochs and wait >= patience:
            print(f"Early stopping triggered at epoch {ep+1} (patience={patience}).")
            break

    return (
        np.array(train_losses, dtype=np.float32),
        np.array(val_losses, dtype=np.float32),
        np.array(val_accs, dtype=np.float32),
        best_val_epoch + 1,
        float(best_val_acc),
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_model(model_path, test_loader, device):
    """Load saved model and evaluate on test loader."""
    model = torch.load(model_path, map_location=device)
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, dtype=torch.float)
            probs  = torch.softmax(model(inputs), dim=1).cpu().numpy()
            preds  = probs.argmax(axis=1)
            y_true.extend(targets.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)
    return (np.array(y_true), np.array(y_pred), np.array(y_probs))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_loss(train_losses, val_losses, k_label, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train loss")
    ax.plot(val_losses,   label="Val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Training Curves — {k_label}")
    ax.legend(); fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_cm(y_true, y_pred, k_label, save_path):
    class_names = ["Down", "Stationary", "Up"]
    cm_val  = confusion_matrix(y_true, y_pred)
    cm_norm = cm_val.astype(float) / cm_val.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(cm_val,  display_labels=class_names).plot(ax=axes[0], colorbar=False)
    ConfusionMatrixDisplay(cm_norm, display_labels=class_names).plot(ax=axes[1], colorbar=False)
    axes[0].set_title(f"Confusion Matrix (counts) — {k_label}")
    axes[1].set_title(f"Confusion Matrix (row-normalised) — {k_label}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\n--- Loading FI-2010 data ---")
    dec_train, dec_val, dec_test = load_data()

    K_VALUES = [0, 1, 2, 3, 4]
    K_LABELS = ["k=1 (10 ev)", "k=2 (20 ev)", "k=3 (30 ev)", "k=4 (50 ev)", "k=5 (100 ev)"]

    all_results = {}

    for k_idx, k_label in zip(K_VALUES, K_LABELS):
        print(f"\n{'='*60}")
        print(f"Horizon {k_label}  (k_idx={k_idx})")
        print(f"{'-'*60}")

        pred_file   = os.path.join(RESULT_DIR, f"preds_k{k_idx}.npz")
        loss_file   = os.path.join(RESULT_DIR, f"losses_k{k_idx}.npz")
        model_path  = os.path.join(MODEL_DIR,  f"deeplob_k{k_idx}.pt")

        if not args.force and os.path.exists(pred_file) and os.path.exists(model_path) and os.path.exists(loss_file):
            print("  -> Already done, loading from disk.")
            pdata = np.load(pred_file)
            y_true, y_pred, y_probs = pdata["y_true"], pdata["y_pred"], pdata["y_probs"]
            loss_data = np.load(loss_file)
            train_losses = loss_data["train"]
            val_losses   = loss_data["val"]
            val_accs     = loss_data["val_acc"] if "val_acc" in loss_data else np.array([], dtype=np.float32)
            best_epoch   = int(loss_data["best_epoch"]) if "best_epoch" in loss_data else int(np.argmin(val_losses) + 1)
            best_val_acc = float(loss_data["best_val_acc"]) if "best_val_acc" in loss_data else float("nan")
        else:
            # Build datasets
            ds_train = LOBDataset(dec_train, k=k_idx, T=args.lookback)
            ds_val   = LOBDataset(dec_val,   k=k_idx, T=args.lookback)
            ds_test  = LOBDataset(dec_test,  k=k_idx, T=args.lookback)

            train_loader = data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                           num_workers=4, pin_memory=True)
            val_loader   = data.DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                                           num_workers=4, pin_memory=True)
            test_loader  = data.DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                                           num_workers=4, pin_memory=True)

            # Build & train model
            model     = DeepLOB(y_len=3, dropout=args.dropout).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            train_losses, val_losses, val_accs, best_epoch, best_val_acc = batch_gd(
                model, criterion, optimizer,
                train_loader, val_loader,
                args.epochs, model_path, device,
                patience=args.patience,
                min_epochs=args.min_epochs,
            )

            # Evaluate on test set
            y_true, y_pred, y_probs = evaluate_model(model_path, test_loader, device)

            # Save results
            np.savez_compressed(
                loss_file,
                train=train_losses,
                val=val_losses,
                val_acc=val_accs,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
            )
            np.savez_compressed(pred_file, y_true=y_true, y_pred=y_pred, y_probs=y_probs)

            del ds_train, ds_val, ds_test, train_loader, val_loader, test_loader, model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Metrics
        acc    = accuracy_score(y_true, y_pred)
        kappa  = cohen_kappa_score(y_true, y_pred)
        mcc    = matthews_corrcoef(y_true, y_pred)
        prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred,
                                 average=None, labels=[0, 1, 2])
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred,
                                      average="weighted")

        all_results[k_idx] = {
            "label": k_label, "train_losses": train_losses, "val_losses": val_losses,
            "val_accs": val_accs, "best_epoch": best_epoch, "best_val_acc": best_val_acc,
            "accuracy": acc, "kappa": kappa, "mcc": mcc,
            "precision": prec, "recall": rec, "f1": f1, "support": sup,
            "precision_w": prec_w, "recall_w": rec_w, "f1_w": f1_w,
        }

        print(classification_report(y_true, y_pred,
              target_names=["Down", "Stationary", "Up"], digits=4))

        # Plots
        plot_loss(train_losses, val_losses, k_label,
                  os.path.join(RESULT_DIR, f"loss_k{k_idx}.png"))
        plot_cm(y_true, y_pred, k_label,
                os.path.join(RESULT_DIR, f"cm_k{k_idx}.png"))

    # ---- Save summary ----
    with open(os.path.join(RESULT_DIR, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    rows = []
    for k_idx in K_VALUES:
        r = all_results[k_idx]
        rows.append({
            "Horizon": r["label"], "Accuracy": r["accuracy"],
            "Cohen κ": r["kappa"], "MCC": r["mcc"],
            "F1-Down": r["f1"][0], "F1-Stat": r["f1"][1], "F1-Up": r["f1"][2],
            "F1-Weighted": r["f1_w"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "performance_summary.csv"), index=False, float_format="%.4f")
    print("\n" + "="*65)
    print("  DeepLOB Training Complete — Performance Summary")
    print("="*65)
    print(df.to_string(index=False))
    print(f"\nAll results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
