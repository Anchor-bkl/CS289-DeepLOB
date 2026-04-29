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


def collect_explicit_cli_args(argv: list[str]) -> set[str]:
    explicit = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token[2:].split("=", 1)[0].strip()
        if option:
            explicit.add(option.replace("-", "_"))
    return explicit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "optiver_processed")
MODEL_DIR   = os.environ.get("OPTIVER_MODEL_DIR", os.path.join(BASE_DIR, "models"))
RESULT_DIR  = os.environ.get("OPTIVER_RESULT_DIR", os.path.join(BASE_DIR, "results", "optiver"))
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train DeepLOBLite on Optiver data")
    p.add_argument("--task-type", choices=["classification"], default="classification",
                   help="Primary Optiver objective. Optiver training is fixed to classification.")
    p.add_argument("--epochs",           type=int,   default=50,   help="Base training epochs")
    p.add_argument("--transfer-epochs",  type=int,   default=24,   help="Transfer fine-tune epochs")
    p.add_argument("--batch-size",       type=int,   default=64,   help="Mini-batch size")
    p.add_argument("--lr",               type=float, default=1e-4, help="Base learning rate")
    p.add_argument("--transfer-lr",      type=float, default=1e-5, help="Transfer fine-tune LR")
    p.add_argument("--weight-decay",     type=float, default=3e-4, help="Adam weight decay")
    p.add_argument("--dropout",          type=float, default=0.30, help="Dropout before classifier head")
    p.add_argument("--grad-clip",        type=float, default=1.0, help="Gradient clipping max norm; 0 disables")
    p.add_argument("--horizon-profile", choices=["legacy", "adaptive"], default="adaptive",
                   help="Use per-horizon tuned defaults before applying the shared training loop")
    p.add_argument("--balance-mode", choices=["loss", "sampler", "both", "none"], default="loss",
                   help="Class imbalance handling. 'loss' avoids double-correcting minority classes.")
    p.add_argument("--loss-type", choices=["ce", "focal"], default="focal",
                   help="Loss function. Focal loss can reduce majority-class collapse.")
    p.add_argument("--focal-gamma", type=float, default=1.5,
                   help="Focal loss gamma when --loss-type=focal")
    p.add_argument("--aux-reg-loss", choices=["none", "huber", "mse"], default="huber",
                   help="Auxiliary future log-return loss added on top of classification")
    p.add_argument("--aux-reg-weight", type=float, default=None,
                   help="Override auxiliary regression weight for all horizons")
    p.add_argument("--aux-reg-beta", type=float, default=1.0,
                   help="Huber delta for auxiliary regression loss")
    p.add_argument("--aux-reg-clip", type=float, default=5.0,
                   help="Clip standardized regression targets/predictions to +/- this value; <=0 disables")
    p.add_argument("--monitor", choices=["val_macro_f1", "val_kappa", "val_loss"], default="val_macro_f1",
                   help="Checkpoint / early-stop monitor for Optiver training")
    p.add_argument("--label-mode", choices=["original", "rolling-quantile-3class", "original-5class", "stock-quantile", "stock-quintile-middle", "rolling-quintile-5class"], default="original",
                   help="Use prepared original labels, rolling quantile 3-class labels, or legacy Optiver relabeling modes on load")
    p.add_argument("--quantile-stationary", type=float, default=0.20,
                   help="Stationary class fraction for --label-mode=stock-quantile or rolling-quantile-3class")
    p.add_argument("--rolling-quantile-window", type=int, default=20000,
                   help="Rolling historical event window for --label-mode=rolling-quantile-3class or rolling-quintile-5class")
    p.add_argument("--lr-plateau-patience", type=int, default=2,
                   help="ReduceLROnPlateau patience on validation loss")
    p.add_argument("--lr-plateau-factor", type=float, default=0.5,
                   help="ReduceLROnPlateau multiplicative factor")
    p.add_argument("--patience",         type=int,   default=8,    help="Early-stopping patience")
    p.add_argument("--min-epochs",       type=int,   default=8,    help="Minimum epochs before early stop")
    p.add_argument("--lookback",         type=int,   default=50,   help="LOB lookback T (events)")
    p.add_argument("--horizon",          type=int,   default=3,
                   help="Primary horizon index (0-based, defaults to the retained k=5 slice)")
    p.add_argument("--base-max-per-stock", type=int, default=3000,
                   help="Max sampled windows per train stock for base training; <=0 uses all valid windows")
    p.add_argument("--base-sample-mode", choices=["uniform", "time-id", "volatility"], default="time-id",
                   help="How base-training windows are sampled within each stock when a cap is active")
    p.add_argument("--base-sample-bins", type=int, default=8,
                   help="Number of bins used for volatility-stratified base sampling")
    p.add_argument("--base-stock-scope", choices=["train-only", "selected-transfer-only", "all-stocks"], default="selected-transfer-only",
                   help="Whether base training uses only source stocks, all non-selected stocks, or all stocks with reserved tail windows")
    p.add_argument("--base-val-mode", choices=["random", "temporal"], default="temporal",
                   help="How to form the base-training validation set (default: temporal)")
    p.add_argument("--base-val-frac", type=float, default=0.15,
                   help="Validation fraction used during base training (default: 0.15)")
    p.add_argument("--transfer-tail-frac", type=float, default=0.30,
                   help="When --base-stock-scope=all-stocks, reserve this final fraction of each selected transfer stock for zero-shot / fine-tune / OOS")
    p.add_argument("--transfer-max-samples", type=int, default=9000,
                   help="Max sampled windows per transfer stock; <=0 uses all valid windows")
    p.add_argument("--transfer-stock-selector", choices=["balanced", "manual"], default="balanced",
                   help="How transfer-learning study stocks are selected from the held-out pool")
    p.add_argument("--max-transfer-stocks", type=int, default=3,
                   help="Number of held-out stocks used for transfer-learning study")
    p.add_argument("--transfer-stock-ids", nargs="*", type=int, default=None,
                   help="Explicit retained held-out stock IDs to evaluate / fine-tune when using manual selection")
    p.add_argument("--num-workers", type=int, default=-1,
                   help="DataLoader workers (-1: auto, 0 on CPU / 4 on GPU)")
    p.add_argument("--transfer-mode", choices=["head", "lstm_conv3", "all", "auto"], default="auto",
                   help="Which layers to fine-tune on transfer stocks; 'auto' compares candidate modes on the fine-tune validation split")
    p.add_argument("--specific-epochs",  type=int,   default=36,   help="Specific-stock OOS epochs")
    p.add_argument("--force",            action="store_true",      help="Retrain even if saved")
    args = p.parse_args()
    args.explicit_cli = collect_explicit_cli_args(sys.argv[1:])
    return args

HORIZONS = [1, 2, 3, 5, 10]   # must match prepare_optiver.py defaults
CLASS_NAMES_3 = ["Down", "Stationary", "Up"]
CLASS_NAMES_5 = ["Strong Down", "Down", "Stationary", "Up", "Strong Up"]
ORIGINAL_LABEL_ALPHA = 0.002
ORIGINAL_LABEL_ROLL_STD_WIN = 200
_OPTIVER_LABEL_CACHE: dict[tuple, np.ndarray] = {}


def num_classes_for_label_mode(label_mode: str) -> int:
    return 5 if label_mode in {"original-5class", "rolling-quintile-5class"} else 3


def class_names_for_num_classes(num_classes: int) -> list[str]:
    if num_classes == 5:
        return list(CLASS_NAMES_5)
    return list(CLASS_NAMES_3)


def signal_values_for_num_classes(num_classes: int) -> np.ndarray:
    center = num_classes // 2
    return np.arange(num_classes, dtype=np.int64) - center


def resolve_training_config(args, horizon_k: int) -> dict:
    if args.task_type != "classification":
        raise ValueError("Optiver training is classification-only")
    cfg = {
        "task_type": "classification",
        "epochs": args.epochs,
        "transfer_epochs": args.transfer_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "transfer_lr": args.transfer_lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "grad_clip": args.grad_clip,
        "balance_mode": args.balance_mode,
        "loss_type": args.loss_type,
        "focal_gamma": args.focal_gamma,
        "aux_reg_loss": args.aux_reg_loss,
        "aux_reg_weight": 0.0,
        "aux_reg_beta": args.aux_reg_beta,
        "aux_reg_clip": args.aux_reg_clip,
        "monitor": args.monitor,
        "label_mode": args.label_mode,
        "quantile_stationary": args.quantile_stationary,
        "rolling_quantile_window": args.rolling_quantile_window,
        "lr_plateau_patience": args.lr_plateau_patience,
        "lr_plateau_factor": args.lr_plateau_factor,
        "patience": args.patience,
        "min_epochs": args.min_epochs,
        "lookback": args.lookback,
        "base_max_per_stock": args.base_max_per_stock,
        "base_sample_mode": args.base_sample_mode,
        "base_sample_bins": args.base_sample_bins,
        "base_stock_scope": args.base_stock_scope,
        "base_val_mode": args.base_val_mode,
        "base_val_frac": args.base_val_frac,
        "transfer_tail_frac": args.transfer_tail_frac,
        "transfer_max_samples": args.transfer_max_samples,
        "transfer_stock_selector": args.transfer_stock_selector,
        "max_transfer_stocks": args.max_transfer_stocks,
        "transfer_stock_ids": args.transfer_stock_ids,
        "transfer_mode": args.transfer_mode,
        "specific_epochs": args.specific_epochs,
    }
    if args.horizon_profile == "adaptive":
        adaptive = {
            1: {
                "lr": 3e-4,
                "transfer_lr": 1e-5,
                "transfer_mode": "head",
                "label_mode": "stock-quantile",
                "quantile_stationary": 0.28,
                "balance_mode": "both",
                "loss_type": "focal",
                "monitor": "val_kappa",
                "dropout": 0.15,
                "lr_plateau_patience": 3,
                "patience": 10,
                "min_epochs": 10,
                "aux_reg_weight": 0.0,
                "transfer_epochs": 16,
            },
            2: {
                "lr": 9e-5,
                "transfer_lr": 2e-5,
                "transfer_mode": "head",
                "label_mode": "stock-quantile",
                "quantile_stationary": 0.38,
                "balance_mode": "loss",
                "loss_type": "focal",
                "monitor": "val_kappa",
                "dropout": 0.30,
                "aux_reg_weight": 0.08,
            },
            3: {
                "lr": 1e-4,
                "transfer_lr": 1.5e-5,
                "transfer_mode": "head",
                "label_mode": "stock-quantile",
                "quantile_stationary": 0.36,
                "balance_mode": "loss",
                "loss_type": "focal",
                "monitor": "val_kappa",
                "dropout": 0.32,
                "aux_reg_weight": 0.10,
            },
            5: {
                "lr": 1.2e-4,
                "transfer_lr": 2e-5,
                "transfer_mode": "auto",
                "label_mode": "original",
                "quantile_stationary": 0.20,
                "balance_mode": "loss",
                "loss_type": "focal",
                "monitor": "val_macro_f1",
                "dropout": 0.30,
                "patience": 10,
                "min_epochs": 10,
                "aux_reg_weight": 0.08,
                "specific_epochs": 36,
                "rolling_quantile_window": 20000,
            },
            10: {
                "lr": 9e-5,
                "transfer_lr": 1e-5,
                "transfer_mode": "lstm_conv3",
                "label_mode": "stock-quantile",
                "quantile_stationary": 0.34,
                "balance_mode": "sampler",
                "loss_type": "focal",
                "monitor": "val_kappa",
                "dropout": 0.35,
                "patience": 10,
                "min_epochs": 8,
                "aux_reg_weight": 0.22,
                "specific_epochs": 40,
            },
        }
        cfg.update(adaptive.get(horizon_k, {}))
        for key in getattr(args, "explicit_cli", set()):
            if key in cfg and hasattr(args, key):
                cfg[key] = getattr(args, key)
    if cfg["task_type"] == "regression":
        if args.aux_reg_weight is not None:
            cfg["aux_reg_weight"] = args.aux_reg_weight
        if cfg["monitor"] in {"val_macro_f1", "val_kappa"}:
            cfg["monitor"] = "val_reg_corr"
        if cfg["aux_reg_loss"] == "none":
            cfg["aux_reg_loss"] = "huber"
        cfg["aux_reg_weight"] = 1.0
        cfg["balance_mode"] = "none"
    else:
        cfg["aux_reg_weight"] = 0.0
        cfg["aux_reg_loss"] = "none"
    cfg["base_max_per_stock"] = normalize_optional_limit(cfg["base_max_per_stock"])
    cfg["transfer_max_samples"] = normalize_optional_limit(cfg["transfer_max_samples"])
    cfg["transfer_tail_frac"] = float(np.clip(cfg["transfer_tail_frac"], 0.05, 0.95))
    cfg["rolling_quantile_window"] = max(500, int(cfg["rolling_quantile_window"]))
    cfg["num_classes"] = num_classes_for_label_mode(cfg["label_mode"])
    cfg["class_names"] = class_names_for_num_classes(cfg["num_classes"])
    return cfg


def uses_regression_targets(cfg: dict) -> bool:
    return cfg.get("task_type") == "regression"


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""
    def __init__(self, weight=None, gamma: float = 1.5):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def make_future_log_returns(mid: np.ndarray, horizon_events: int) -> np.ndarray:
    future_log_ret = np.full(len(mid), np.nan, dtype=np.float32)
    if horizon_events >= len(mid):
        return future_log_ret
    safe_mid = np.clip(mid.astype(np.float64, copy=False), 1e-10, None)
    future_log_ret[: len(mid) - horizon_events] = (
        np.log(safe_mid[horizon_events:]) - np.log(safe_mid[: len(mid) - horizon_events])
    ).astype(np.float32)
    return future_log_ret


def normalize_optional_limit(value: int | None) -> int | None:
    if value is None:
        return None
    return None if int(value) <= 0 else int(value)


def evenly_spaced_positions(length: int, num: int) -> np.ndarray:
    if num >= length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=num, dtype=int).astype(np.int64, copy=False)


def make_rolling_volatility(mid: np.ndarray, window: int) -> np.ndarray:
    safe_mid = np.clip(mid.astype(np.float64, copy=False), 1e-10, None)
    log_ret = np.diff(np.log(safe_mid), prepend=np.log(safe_mid[0]))
    min_periods = max(5, min(window, 20))
    roll = pd.Series(log_ret).rolling(window=window, min_periods=min_periods).std(ddof=0)
    return np.nan_to_num(roll.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def proportional_group_quotas(group_sizes: np.ndarray, max_samples: int) -> np.ndarray:
    total = int(group_sizes.sum())
    if max_samples >= total:
        return group_sizes.astype(np.int64, copy=True)

    raw = group_sizes.astype(np.float64) / max(total, 1) * max_samples
    quotas = np.floor(raw).astype(np.int64)
    quotas = np.minimum(quotas, group_sizes.astype(np.int64, copy=False))

    positive = group_sizes > 0
    if max_samples >= int(positive.sum()):
        need_one = positive & (quotas == 0)
        quotas[need_one] = 1

    while int(quotas.sum()) > max_samples:
        reducible = np.where(quotas > 0)[0]
        if len(reducible) == 0:
            break
        order = reducible[np.argsort(raw[reducible] - quotas[reducible])]
        quotas[order[0]] -= 1

    while int(quotas.sum()) < max_samples:
        capacity = group_sizes.astype(np.int64, copy=False) - quotas
        candidates = np.where(capacity > 0)[0]
        if len(candidates) == 0:
            break
        order = candidates[np.argsort(-(raw[candidates] - quotas[candidates]))]
        quotas[order[0]] += 1

    return quotas.astype(np.int64, copy=False)


def make_sampling_groups(
    valid_end: np.ndarray,
    sample_mode: str,
    time_id: np.ndarray | None,
    mid: np.ndarray | None,
    lookback: int,
    sample_bins: int,
) -> np.ndarray | None:
    if sample_mode == "time-id":
        if time_id is None:
            return None
        return time_id[valid_end - 1].astype(np.int64, copy=False)
    if sample_mode == "volatility":
        if mid is None:
            return None
        roll_vol = make_rolling_volatility(mid, window=lookback)
        vol_values = roll_vol[valid_end - 1]
        if len(vol_values) == 0:
            return None
        quantiles = np.linspace(0.0, 1.0, num=max(2, int(sample_bins)) + 1)
        edges = np.unique(np.quantile(vol_values, quantiles))
        if len(edges) <= 1:
            return None
        return np.digitize(vol_values, edges[1:-1], right=True).astype(np.int64, copy=False)
    return None


def sample_valid_windows(
    valid_end: np.ndarray,
    max_samples: int | None,
    sample_mode: str = "uniform",
    time_id: np.ndarray | None = None,
    mid: np.ndarray | None = None,
    lookback: int = 50,
    sample_bins: int = 8,
) -> np.ndarray:
    if max_samples is None or len(valid_end) <= max_samples:
        return valid_end.astype(np.int64, copy=False)

    if sample_mode == "uniform":
        return valid_end[evenly_spaced_positions(len(valid_end), max_samples)].astype(np.int64, copy=False)

    groups = make_sampling_groups(valid_end, sample_mode, time_id, mid, lookback, sample_bins)
    if groups is None:
        return valid_end[evenly_spaced_positions(len(valid_end), max_samples)].astype(np.int64, copy=False)

    _, inverse = np.unique(groups, return_inverse=True)
    group_positions = [np.flatnonzero(inverse == idx) for idx in range(int(inverse.max()) + 1)]
    group_sizes = np.array([len(pos) for pos in group_positions], dtype=np.int64)
    quotas = proportional_group_quotas(group_sizes, max_samples)

    selected = []
    for positions, quota in zip(group_positions, quotas):
        if quota <= 0:
            continue
        chosen = positions[evenly_spaced_positions(len(positions), int(quota))]
        selected.append(chosen)
    if not selected:
        return valid_end[evenly_spaced_positions(len(valid_end), max_samples)].astype(np.int64, copy=False)

    chosen_positions = np.sort(np.concatenate(selected))
    sampled = valid_end[chosen_positions]
    if len(sampled) > max_samples:
        sampled = sampled[evenly_spaced_positions(len(sampled), max_samples)]
    return sampled.astype(np.int64, copy=False)


def unpack_model_output(outputs):
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]
    return outputs, None


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        classification_loss,
        reg_weight=0.0,
        reg_loss="huber",
        reg_mean=0.0,
        reg_std=1.0,
        reg_beta=1.0,
        reg_clip=0.0,
    ):
        super().__init__()
        self.classification_loss = classification_loss
        self.reg_weight = float(reg_weight)
        self.reg_loss = reg_loss
        self.reg_mean = float(reg_mean)
        self.reg_std = max(float(reg_std), 1e-6)
        self.reg_beta = float(reg_beta)
        self.reg_clip = float(reg_clip)

    def forward(self, outputs, cls_targets, reg_targets=None):
        logits, reg_pred = unpack_model_output(outputs)
        cls_loss = self.classification_loss(logits, cls_targets)
        reg_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        if self.reg_weight > 0 and reg_targets is not None and reg_pred is not None:
            reg_targets = (reg_targets - self.reg_mean) / self.reg_std
            reg_pred = (reg_pred - self.reg_mean) / self.reg_std
            if self.reg_clip > 0:
                reg_targets = torch.clamp(reg_targets, -self.reg_clip, self.reg_clip)
                reg_pred = torch.clamp(reg_pred, -self.reg_clip, self.reg_clip)
            if self.reg_loss == "mse":
                reg_loss = nn.functional.mse_loss(reg_pred, reg_targets)
            else:
                reg_loss = nn.functional.huber_loss(reg_pred, reg_targets, delta=self.reg_beta)
        total_loss = cls_loss + self.reg_weight * reg_loss
        return {
            "total": total_loss,
            "classification": cls_loss,
            "regression": reg_loss,
        }


class RegressionOnlyLoss(nn.Module):
    def __init__(self, reg_loss="huber", reg_mean=0.0, reg_std=1.0, reg_beta=1.0, reg_clip=0.0):
        super().__init__()
        self.reg_loss = reg_loss
        self.reg_mean = float(reg_mean)
        self.reg_std = max(float(reg_std), 1e-6)
        self.reg_beta = float(reg_beta)
        self.reg_clip = float(reg_clip)

    def forward(self, outputs, cls_targets=None, reg_targets=None):
        reg_pred = outputs[1] if isinstance(outputs, tuple) else outputs
        if reg_targets is None:
            raise ValueError("Regression task requires continuous targets")
        reg_targets = (reg_targets - self.reg_mean) / self.reg_std
        reg_pred = (reg_pred - self.reg_mean) / self.reg_std
        if self.reg_clip > 0:
            reg_targets = torch.clamp(reg_targets, -self.reg_clip, self.reg_clip)
            reg_pred = torch.clamp(reg_pred, -self.reg_clip, self.reg_clip)
        if self.reg_loss == "mse":
            reg_loss = nn.functional.mse_loss(reg_pred, reg_targets)
        else:
            reg_loss = nn.functional.huber_loss(reg_pred, reg_targets, delta=self.reg_beta)
        zero = torch.zeros((), dtype=reg_loss.dtype, device=reg_loss.device)
        return {
            "total": reg_loss,
            "classification": zero,
            "regression": reg_loss,
        }

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
        sample_mode: str = "uniform",
        sample_bins: int = 8,
        time_id: np.ndarray | None = None,
        mid: np.ndarray | None = None,
        reg_target: np.ndarray | None = None,
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
        self.reg_target = (
            reg_target.astype(np.float32, copy=False)
            if reg_target is not None
            else None
        )

        valid_end = np.arange(T, len(X) + 1)
        valid_end = valid_end[self.y[valid_end - 1] >= 0]
        if self.reg_target is not None:
            valid_end = valid_end[np.isfinite(self.reg_target[valid_end - 1])]
        self.valid_end = sample_valid_windows(
            valid_end,
            max_samples=max_samples,
            sample_mode=sample_mode,
            time_id=self.time_id,
            mid=self.mid,
            lookback=T,
            sample_bins=sample_bins,
        )

    def __len__(self):
        return len(self.valid_end)

    def __getitem__(self, idx):
        end = int(self.valid_end[idx])
        x = self.X[end - self.T : end][np.newaxis, :, :]
        y = int(self.y[end - 1])
        if self.reg_target is not None:
            reg = float(self.reg_target[end - 1])
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), torch.tensor(reg, dtype=torch.float32)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def rebuild_quantile_labels(mid: np.ndarray, horizon_k: int, stationary_frac: float) -> np.ndarray:
    """Rebuild labels so the stationary band is controlled per stock/horizon."""
    stationary_frac = float(np.clip(stationary_frac, 0.05, 0.90))
    labels = np.full(len(mid), -1, dtype=np.int32)
    if horizon_k >= len(mid):
        return labels
    ret = (mid[horizon_k:] - mid[:-horizon_k]) / (mid[:-horizon_k] + 1e-10)
    threshold = float(np.quantile(np.abs(ret), stationary_frac))
    lbl = np.ones(len(ret), dtype=np.int32)
    lbl[ret > threshold] = 2
    lbl[ret < -threshold] = 0
    labels[: len(ret)] = lbl
    return labels


def rebuild_quintile_middle_labels(mid: np.ndarray, horizon_k: int) -> np.ndarray:
    """Split future price changes into five per-stock bins; only the middle bin is stationary."""
    labels = np.full(len(mid), -1, dtype=np.int32)
    if horizon_k >= len(mid):
        return labels
    price_change = mid[horizon_k:] - mid[:-horizon_k]
    quintile_edges = np.quantile(price_change, [0.2, 0.4, 0.6, 0.8])
    quintile_idx = np.digitize(price_change, quintile_edges, right=False)
    lbl = np.ones(len(price_change), dtype=np.int32)
    lbl[quintile_idx <= 1] = 0
    lbl[quintile_idx >= 3] = 2
    labels[: len(price_change)] = lbl
    return labels


def rebuild_rolling_quintile_5class_labels(mid: np.ndarray, horizon_k: int, rolling_window: int) -> np.ndarray:
    """Assign 5 classes from rolling historical quantiles of future k-step price changes."""
    labels = np.full(len(mid), -1, dtype=np.int32)
    if horizon_k >= len(mid):
        return labels

    rolling_window = max(500, int(rolling_window))
    min_history = min(rolling_window, max(200, rolling_window // 10))
    price_change = mid[horizon_k:] - mid[:-horizon_k]
    history = pd.Series(price_change, dtype=np.float64).shift(1)

    quantile_series = []
    for q in (0.2, 0.4, 0.6, 0.8):
        rolling_q = history.rolling(window=rolling_window, min_periods=min_history).quantile(q)
        expanding_q = history.expanding(min_periods=min_history).quantile(q)
        quantile_series.append(rolling_q.fillna(expanding_q).to_numpy(dtype=np.float64))

    q20, q40, q60, q80 = quantile_series
    valid = np.isfinite(q20) & np.isfinite(q40) & np.isfinite(q60) & np.isfinite(q80)
    if not np.any(valid):
        return labels

    valid_values = price_change[valid]
    valid_labels = np.zeros(valid_values.shape[0], dtype=np.int32)
    valid_labels[valid_values > q20[valid]] = 1
    valid_labels[valid_values > q40[valid]] = 2
    valid_labels[valid_values > q60[valid]] = 3
    valid_labels[valid_values > q80[valid]] = 4

    compact_labels = np.full(len(price_change), -1, dtype=np.int32)
    compact_labels[valid] = valid_labels
    labels[: len(price_change)] = compact_labels
    return labels


def rebuild_rolling_quantile_3class_labels(
    mid: np.ndarray,
    horizon_k: int,
    stationary_frac: float,
    rolling_window: int,
) -> np.ndarray:
    """Assign 3 classes from rolling historical quantiles of future k-step returns."""
    labels = np.full(len(mid), -1, dtype=np.int32)
    if horizon_k >= len(mid):
        return labels

    stationary_frac = float(np.clip(stationary_frac, 0.05, 0.90))
    rolling_window = max(500, int(rolling_window))
    min_history = min(rolling_window, max(200, rolling_window // 10))

    ret = (mid[horizon_k:] - mid[:-horizon_k]) / (mid[:-horizon_k] + 1e-10)
    history = pd.Series(np.abs(ret), dtype=np.float64).shift(1)
    rolling_q = history.rolling(window=rolling_window, min_periods=min_history).quantile(stationary_frac)
    expanding_q = history.expanding(min_periods=min_history).quantile(stationary_frac)
    threshold = rolling_q.fillna(expanding_q).to_numpy(dtype=np.float64)

    valid = np.isfinite(threshold)
    if not np.any(valid):
        return labels

    valid_ret = ret[valid]
    valid_threshold = threshold[valid]
    valid_labels = np.ones(valid_ret.shape[0], dtype=np.int32)
    valid_labels[valid_ret > valid_threshold] = 2
    valid_labels[valid_ret < -valid_threshold] = 0

    compact_labels = np.full(len(ret), -1, dtype=np.int32)
    compact_labels[valid] = valid_labels
    labels[: len(ret)] = compact_labels
    return labels


def rebuild_original_5class_labels(
    mid: np.ndarray,
    horizon_k: int,
    alpha: float = ORIGINAL_LABEL_ALPHA,
    roll_std_win: int = ORIGINAL_LABEL_ROLL_STD_WIN,
) -> np.ndarray:
    """Extend the original adaptive threshold labels into five symmetric bins."""
    labels = np.full(len(mid), -1, dtype=np.int32)
    if horizon_k >= len(mid):
        return labels

    ret = np.zeros(len(mid), dtype=np.float64)
    ret[: len(mid) - horizon_k] = (mid[horizon_k:] - mid[: len(mid) - horizon_k]) / (mid[: len(mid) - horizon_k] + 1e-10)

    threshold = alpha * pd.Series(ret).rolling(roll_std_win, min_periods=1).std(ddof=0).to_numpy(dtype=np.float64)
    threshold = np.where(np.isnan(threshold), 0.0, threshold)

    valid_ret = ret[: len(mid) - horizon_k]
    inner = threshold[: len(valid_ret)]
    outer = 2.0 * inner

    lbl = np.full(len(valid_ret), 2, dtype=np.int32)
    lbl[valid_ret > inner] = 3
    lbl[valid_ret > outer] = 4
    lbl[valid_ret < -inner] = 1
    lbl[valid_ret < -outer] = 0
    labels[: len(valid_ret)] = lbl
    return labels


def load_stock_dataset(
    stock_id: int,
    horizon_k: int,
    T: int,
    max_samples: int | None = None,
    sample_mode: str = "uniform",
    sample_bins: int = 8,
    label_mode: str = "original",
    quantile_stationary: float = 0.34,
    rolling_quantile_window: int = 20000,
    return_regression: bool = False,
) -> OptiverLOBDataset | None:
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
    cache_key = (stock_id, horizon_k, label_mode, round(float(quantile_stationary), 6), int(rolling_quantile_window))
    if cache_key in _OPTIVER_LABEL_CACHE:
        y = _OPTIVER_LABEL_CACHE[cache_key]
    else:
        if label_mode == "stock-quantile":
            y = rebuild_quantile_labels(npz["mid"].astype(np.float64), horizon_k, quantile_stationary)
        elif label_mode == "rolling-quantile-3class":
            y = rebuild_rolling_quantile_3class_labels(
                npz["mid"].astype(np.float64),
                horizon_k,
                quantile_stationary,
                rolling_quantile_window,
            )
        elif label_mode == "original-5class":
            y = rebuild_original_5class_labels(npz["mid"].astype(np.float64), horizon_k)
        elif label_mode == "stock-quintile-middle":
            y = rebuild_quintile_middle_labels(npz["mid"].astype(np.float64), horizon_k)
        elif label_mode == "rolling-quintile-5class":
            y = rebuild_rolling_quintile_5class_labels(npz["mid"].astype(np.float64), horizon_k, rolling_quantile_window)
        else:
            y = npz[key].astype(np.int32)
        _OPTIVER_LABEL_CACHE[cache_key] = y
    reg_target = None
    if return_regression and mid is not None:
        reg_target = make_future_log_returns(mid.astype(np.float64), horizon_k)
    ds = OptiverLOBDataset(
        X,
        y,
        T=T,
        max_samples=max_samples,
        sample_mode=sample_mode,
        sample_bins=sample_bins,
        time_id=time_id,
        mid=mid,
        reg_target=reg_target,
    )
    if len(ds) == 0:
        return None
    return ds


def dataset_prefix_subset(ds: data.Dataset, keep_frac: float, min_keep: int = 100):
    keep_frac = float(np.clip(keep_frac, 0.0, 1.0))
    if keep_frac >= 1.0:
        return ds
    n_total = len(ds)
    if n_total <= min_keep:
        return ds
    n_keep = max(min_keep, int(n_total * keep_frac))
    n_keep = min(n_keep, n_total)
    if n_keep >= n_total:
        return ds
    return data.Subset(ds, range(0, n_keep))


def dataset_tail_subset(ds: data.Dataset, tail_frac: float, min_keep: int = 200):
    tail_frac = float(np.clip(tail_frac, 0.0, 1.0))
    if tail_frac >= 1.0:
        return ds
    n_total = len(ds)
    if n_total <= min_keep:
        return ds
    n_keep = max(min_keep, int(n_total * tail_frac))
    n_keep = min(n_keep, n_total)
    if n_keep >= n_total:
        return ds
    return data.Subset(ds, range(n_total - n_keep, n_total))


def concat_datasets(stock_ids: list, horizon_k: int, T: int,
                    val_frac: float = 0.15, max_per_stock: int | None = None,
                    sample_mode: str = "uniform", sample_bins: int = 8,
                    val_mode: str = "temporal",
                    label_mode: str = "original", quantile_stationary: float = 0.34,
                    rolling_quantile_window: int = 20000,
                    return_regression: bool = False,
                    prefix_frac_by_stock: dict[int, float] | None = None) -> tuple:
    """Concatenate datasets from multiple stocks, split into train/val."""
    all_datasets = []
    train_parts = []
    val_parts = []
    skipped = []
    for sid in stock_ids:
        ds = load_stock_dataset(
            sid,
            horizon_k,
            T,
            max_samples=max_per_stock,
            sample_mode=sample_mode,
            sample_bins=sample_bins,
            label_mode=label_mode,
            quantile_stationary=quantile_stationary,
            rolling_quantile_window=rolling_quantile_window,
            return_regression=return_regression,
        )
        if ds is None or len(ds) < 100:
            skipped.append(sid)
            continue
        if prefix_frac_by_stock and sid in prefix_frac_by_stock:
            ds = dataset_prefix_subset(ds, prefix_frac_by_stock[sid])
            if ds is None or len(ds) < 100:
                skipped.append(sid)
                continue
        if val_mode == "temporal":
            n_val = max(100, int(len(ds) * val_frac))
            n_train = len(ds) - n_val
            if n_train < 100 or n_val <= 0:
                skipped.append(sid)
                continue
            train_parts.append(data.Subset(ds, range(0, n_train)))
            val_parts.append(data.Subset(ds, range(n_train, len(ds))))
            continue
        all_datasets.append(ds)

    if val_mode == "temporal":
        if not train_parts or not val_parts:
            return None, None
        train_ds = data.ConcatDataset(train_parts)
        val_ds = data.ConcatDataset(val_parts)
        n = len(train_ds) + len(val_ds)
        n_train = len(train_ds)
        n_val = len(val_ds)
        used_stocks = len(train_parts)
    else:
        if not all_datasets:
            return None, None

        combined = data.ConcatDataset(all_datasets)
        n        = len(combined)
        n_val    = max(100, int(n * val_frac))
        n_train  = n - n_val
        train_ds, val_ds = data.random_split(combined, [n_train, n_val],
                                             generator=torch.Generator().manual_seed(42))
        used_stocks = len(all_datasets)

    if skipped:
        print(f"  Skipped stocks (no data): {skipped}")
    print(f"  Combined: {n:,} samples ({n_train:,} train, {n_val:,} val) "
          f"from {used_stocks} stocks using {val_mode} validation")
    return train_ds, val_ds


def select_transfer_eval_ids(transfer_ids: list[int], max_transfer_stocks: int) -> list[int]:
    if max_transfer_stocks <= 0 or max_transfer_stocks >= len(transfer_ids):
        return transfer_ids
    pick = np.linspace(0, len(transfer_ids) - 1, num=max_transfer_stocks, dtype=int)
    return [transfer_ids[i] for i in pick]


def expected_label_probs(label_mode: str, quantile_stationary: float) -> np.ndarray:
    if label_mode == "rolling-quintile-5class":
        return np.full(5, 0.2, dtype=np.float64)
    if label_mode == "stock-quintile-middle":
        return np.array([0.4, 0.2, 0.4], dtype=np.float64)
    if label_mode in {"stock-quantile", "rolling-quantile-3class"}:
        stationary = float(np.clip(quantile_stationary, 0.05, 0.90))
        side = 0.5 * (1.0 - stationary)
        return np.array([side, stationary, side], dtype=np.float64)
    num_classes = num_classes_for_label_mode(label_mode)
    return np.full(num_classes, 1.0 / float(num_classes), dtype=np.float64)


def label_balance_summary(labels: np.ndarray, num_classes: int, target_probs: np.ndarray | None = None) -> dict:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.int64)
    total = int(counts.sum())
    if total <= 0:
        probs = np.zeros(num_classes, dtype=np.float64)
        entropy = 0.0
    else:
        probs = counts.astype(np.float64) / float(total)
        nonzero = probs > 0
        entropy = float(-(probs[nonzero] * np.log(probs[nonzero])).sum() / np.log(float(max(num_classes, 2))))
    if target_probs is None:
        target_probs = np.full(num_classes, 1.0 / float(num_classes), dtype=np.float64)
    target_probs = np.asarray(target_probs, dtype=np.float64)
    target_probs = target_probs / np.clip(target_probs.sum(), 1e-12, None)
    l1_gap = float(np.abs(probs - target_probs).sum())
    return {
        "counts": counts.tolist(),
        "probs": probs.tolist(),
        "target_probs": target_probs.tolist(),
        "entropy": entropy,
        "l1_gap": l1_gap,
        "n_samples": total,
    }


def rank_balanced_transfer_stocks(transfer_ids: list[int], horizon_k: int, cfg: dict) -> list[dict]:
    ranked = []
    target_probs = expected_label_probs(cfg["label_mode"], cfg["quantile_stationary"])
    for sid in transfer_ids:
        ds = load_stock_dataset(
            sid,
            horizon_k,
            cfg["lookback"],
            max_samples=None,
            label_mode=cfg["label_mode"],
            quantile_stationary=cfg["quantile_stationary"],
            rolling_quantile_window=cfg["rolling_quantile_window"],
            return_regression=False,
        )
        if ds is None or len(ds) < 200:
            continue
        summary = label_balance_summary(dataset_labels(ds), num_classes=cfg["num_classes"], target_probs=target_probs)
        ranked.append({"stock_id": sid, **summary})
    ranked.sort(key=lambda row: (row["l1_gap"], -row["entropy"], -row["n_samples"], row["stock_id"]))
    return ranked


def filter_transfer_stock_ids(transfer_ids: list[int], requested_ids: list[int] | None) -> list[int]:
    if not requested_ids:
        return transfer_ids
    transfer_set = set(transfer_ids)
    filtered = [stock_id for stock_id in requested_ids if stock_id in transfer_set]
    missing = sorted(set(requested_ids) - transfer_set)
    if missing:
        print(f"Ignoring non-held-out transfer stock IDs: {missing}")
    if not filtered:
        raise ValueError("No requested transfer stock IDs are present in the held-out split")
    return filtered


def choose_transfer_eval_ids(transfer_ids: list[int], horizon_k: int, cfg: dict) -> list[int]:
    if cfg["transfer_stock_selector"] == "manual":
        selected = filter_transfer_stock_ids(transfer_ids, cfg["transfer_stock_ids"])
        selected = select_transfer_eval_ids(selected, cfg["max_transfer_stocks"])
        print(f"Transfer-learning study stocks (manual): {selected}")
        return selected

    ranked = rank_balanced_transfer_stocks(transfer_ids, horizon_k, cfg)
    if not ranked:
        raise ValueError("Could not rank held-out stocks by label balance; no usable transfer stocks were found")
    n_keep = cfg["max_transfer_stocks"] if cfg["max_transfer_stocks"] > 0 else min(3, len(ranked))
    selected_rows = ranked[:n_keep]
    selected = [row["stock_id"] for row in selected_rows]
    print("Transfer stock balance ranking (lower gap is closer to the active label target):")
    for row in ranked:
        probs = ", ".join(f"{p:.3f}" for p in row["probs"])
        target = ", ".join(f"{p:.3f}" for p in row["target_probs"])
        print(
            f"  stock {row['stock_id']}: counts={row['counts']} probs=[{probs}] "
            f"target=[{target}] gap={row['l1_gap']:.4f} entropy={row['entropy']:.4f} n={row['n_samples']}"
        )
    print(f"Transfer-learning study stocks (balanced): {selected}")
    return selected


def build_base_stock_plan(
    train_ids: list[int],
    transfer_ids: list[int],
    transfer_eval_ids: list[int],
    cfg: dict,
) -> tuple[list[int], dict[int, float]]:
    if cfg["base_stock_scope"] == "train-only":
        return list(train_ids), {}

    if cfg["base_stock_scope"] == "selected-transfer-only":
        selected_transfer_set = set(transfer_eval_ids)
        base_stock_ids = list(train_ids) + [sid for sid in transfer_ids if sid not in selected_transfer_set]
        return base_stock_ids, {}

    base_stock_ids = list(train_ids)
    prefix_frac_by_stock: dict[int, float] = {}
    transfer_eval_set = set(transfer_eval_ids)
    for sid in transfer_ids:
        base_stock_ids.append(sid)
        if sid in transfer_eval_set:
            prefix_frac_by_stock[sid] = 1.0 - cfg["transfer_tail_frac"]
    return base_stock_ids, prefix_frac_by_stock


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


def dataset_regression_targets(ds: data.Dataset) -> np.ndarray:
    if isinstance(ds, OptiverLOBDataset):
        if ds.reg_target is None:
            return np.array([], dtype=np.float32)
        return ds.reg_target[ds.valid_end - 1].astype(np.float32, copy=False)
    if isinstance(ds, data.Subset):
        base_targets = dataset_regression_targets(ds.dataset)
        if base_targets.size == 0:
            return base_targets
        return base_targets[np.asarray(ds.indices, dtype=np.int64)]
    if isinstance(ds, data.ConcatDataset):
        chunks = [dataset_regression_targets(sub_ds) for sub_ds in ds.datasets]
        chunks = [chunk for chunk in chunks if chunk.size > 0]
        if not chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(chunks, axis=0)
    raise TypeError(f"Unsupported dataset type for regression target extraction: {type(ds)!r}")


def dataset_sample_end(ds: data.Dataset) -> np.ndarray:
    if isinstance(ds, OptiverLOBDataset):
        return ds.valid_end.astype(np.int64, copy=False)
    if isinstance(ds, data.Subset):
        base_end = dataset_sample_end(ds.dataset)
        return base_end[np.asarray(ds.indices, dtype=np.int64)]
    if isinstance(ds, data.ConcatDataset):
        return np.concatenate([dataset_sample_end(sub_ds) for sub_ds in ds.datasets], axis=0)
    raise TypeError(f"Unsupported dataset type for sample-end extraction: {type(ds)!r}")


def dataset_sample_time_id(ds: data.Dataset) -> np.ndarray:
    if isinstance(ds, OptiverLOBDataset):
        return ds.time_id[ds.valid_end - 1].astype(np.int64, copy=False)
    if isinstance(ds, data.Subset):
        base_time_id = dataset_sample_time_id(ds.dataset)
        return base_time_id[np.asarray(ds.indices, dtype=np.int64)]
    if isinstance(ds, data.ConcatDataset):
        return np.concatenate([dataset_sample_time_id(sub_ds) for sub_ds in ds.datasets], axis=0)
    raise TypeError(f"Unsupported dataset type for sample time-id extraction: {type(ds)!r}")


def make_regression_stats(ds: data.Dataset) -> dict | None:
    reg_targets = dataset_regression_targets(ds)
    if reg_targets.size == 0:
        return None
    reg_mean = float(np.mean(reg_targets))
    reg_std = float(np.std(reg_targets))
    return {"mean": reg_mean, "std": max(reg_std, 1e-6)}


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


def class_weight_for_mode(
    ds: data.Dataset,
    balance_mode: str,
    device: torch.device,
    num_classes: int,
) -> tuple[torch.Tensor | None, np.ndarray]:
    weights, counts = make_class_weights(ds, num_classes=num_classes)
    if balance_mode in {"loss", "both"}:
        return weights.to(device), counts
    return None, counts


def sampler_for_mode(ds: data.Dataset, balance_mode: str, num_classes: int):
    if balance_mode in {"sampler", "both"}:
        return make_weighted_sampler(ds, num_classes=num_classes)
    return None


def loader_shuffle_for_mode(balance_mode: str) -> bool:
    return balance_mode not in {"sampler", "both"}


def make_classification_criterion(loss_type: str, class_weights: torch.Tensor | None, focal_gamma: float):
    if loss_type == "focal":
        return FocalLoss(weight=class_weights, gamma=focal_gamma)
    return nn.CrossEntropyLoss(weight=class_weights)


def build_optiver_model(cfg: dict) -> "DeepLOBLite":
    return DeepLOBLite(
        y_len=cfg["num_classes"],
        dropout=cfg["dropout"],
        regression_head=uses_regression_targets(cfg),
        task_type=cfg["task_type"],
    )


def build_optiver_criterion(cfg: dict, class_weights: torch.Tensor | None, reg_stats: dict | None):
    if cfg["task_type"] == "regression":
        return RegressionOnlyLoss(
            reg_loss=cfg["aux_reg_loss"],
            reg_mean=0.0 if reg_stats is None else reg_stats["mean"],
            reg_std=1.0 if reg_stats is None else reg_stats["std"],
            reg_beta=cfg["aux_reg_beta"],
            reg_clip=cfg["aux_reg_clip"],
        )
    return MultiTaskLoss(
        make_classification_criterion(cfg["loss_type"], class_weights, cfg["focal_gamma"]),
        reg_weight=cfg["aux_reg_weight"],
        reg_loss=cfg["aux_reg_loss"],
        reg_mean=0.0 if reg_stats is None else reg_stats["mean"],
        reg_std=1.0 if reg_stats is None else reg_stats["std"],
        reg_beta=cfg["aux_reg_beta"],
        reg_clip=cfg["aux_reg_clip"],
    )


def evaluate_task_predictions(model, loader, device, task_type: str, num_classes: int, class_names: list[str]):
    y_true, y_pred, aux = evaluate(model, loader, device)
    if task_type == "regression":
        metrics = compute_regression_metrics(y_true, y_pred)
        summary = summarize_regression_predictions(y_pred)
    else:
        metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
        summary = summarize_prediction_distribution(y_pred, num_classes=num_classes, class_names=class_names)
    return y_true, y_pred, aux, metrics, summary


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
    Output: (B, 3) logits for classification, or (B,) return forecasts for regression.
    """
    def __init__(self, y_len: int = 3, dropout: float = 0.30, regression_head: bool = False,
                 task_type: str = "classification"):
        super().__init__()
        self.y_len = y_len
        self.task_type = task_type
        self.use_regression_head = regression_head or task_type == "regression"

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
        self.dropout = nn.Dropout(dropout)
        if task_type != "regression":
            self.fc1 = nn.Linear(64, y_len)
        if self.use_regression_head:
            self.reg_head = nn.Linear(64, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
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
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        if self.task_type == "regression" and hasattr(self, "reg_head"):
            return self.reg_head(x).squeeze(-1)
        logits = self.fc1(x)
        if self.use_regression_head and hasattr(self, "reg_head"):
            return logits, self.reg_head(x).squeeze(-1)
        return logits

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
        reg_params = list(self.reg_head.parameters()) if hasattr(self, "reg_head") else []
        fc_params = list(self.fc1.parameters()) if hasattr(self, "fc1") else []
        return list(self.lstm.parameters()) + fc_params + reg_params

    def get_lstm_conv3_params(self):
        """A modest transfer option: adapt late convolution, temporal layer, and head."""
        reg_params = list(self.reg_head.parameters()) if hasattr(self, "reg_head") else []
        fc_params = list(self.fc1.parameters()) if hasattr(self, "fc1") else []
        return list(self.conv3.parameters()) + list(self.lstm.parameters()) + fc_params + reg_params


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_loop(model, criterion, optimizer, train_loader, val_loader,
               epochs, save_path, device, desc="Epochs", patience=5, min_epochs=5,
               grad_clip=0.0, scheduler=None, monitor="val_macro_f1",
               task_type="classification", num_classes: int = 3):
    """Shared training loop for classification and regression Optiver tasks."""
    train_losses = []
    val_losses   = []
    train_cls_losses = []
    val_cls_losses = []
    train_reg_losses = []
    val_reg_losses = []
    val_reg_corrs = []
    val_macro_f1 = []
    val_kappas = []
    best_val_loss = np.inf
    best_macro_f1 = -np.inf
    best_kappa = -np.inf
    best_reg_corr = -np.inf
    wait = 0

    for ep in tqdm(range(epochs), desc=desc):
        model.train()
        t0 = datetime.now()
        tr_loss = []
        tr_cls_loss = []
        tr_reg_loss = []
        for batch in train_loader:
            if len(batch) == 3:
                inputs, targets, reg_targets = batch
                reg_targets = reg_targets.to(device, dtype=torch.float)
            else:
                inputs, targets = batch
                reg_targets = None
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            optimizer.zero_grad()
            loss_dict = criterion(model(inputs), targets, reg_targets)
            loss = loss_dict["total"]
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            tr_loss.append(loss.item())
            tr_cls_loss.append(loss_dict["classification"].item())
            tr_reg_loss.append(loss_dict["regression"].item())
        tr_loss = np.mean(tr_loss)
        tr_cls_loss = np.mean(tr_cls_loss)
        tr_reg_loss = np.mean(tr_reg_loss)

        model.eval()
        va_loss = []
        va_cls_loss = []
        va_reg_loss = []
        y_true = []
        y_pred = []
        reg_true = []
        reg_pred = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    inputs, targets, reg_targets = batch
                    reg_targets = reg_targets.to(device, dtype=torch.float)
                else:
                    inputs, targets = batch
                    reg_targets = None
                inputs  = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.long)
                outputs = model(inputs)
                loss_dict = criterion(outputs, targets, reg_targets)
                va_loss.append(loss_dict["total"].item())
                va_cls_loss.append(loss_dict["classification"].item())
                va_reg_loss.append(loss_dict["regression"].item())
                if task_type == "regression":
                    reg_outputs = outputs[1] if isinstance(outputs, tuple) else outputs
                    if reg_targets is None:
                        raise ValueError("Regression validation requires continuous targets")
                    reg_true.extend(reg_targets.cpu().numpy().tolist())
                    reg_pred.extend(reg_outputs.cpu().numpy().tolist())
                else:
                    logits, reg_outputs = unpack_model_output(outputs)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    y_pred.extend(preds.tolist())
                    y_true.extend(targets.cpu().numpy().tolist())
                    if reg_targets is not None and reg_outputs is not None:
                        reg_true.extend(reg_targets.cpu().numpy().tolist())
                        reg_pred.extend(reg_outputs.cpu().numpy().tolist())
        va_loss = np.mean(va_loss)
        va_cls_loss = np.mean(va_cls_loss)
        va_reg_loss = np.mean(va_reg_loss)
        reg_rmse = float("nan")
        reg_mae = float("nan")
        sign_acc = float("nan")
        if task_type == "regression":
            reg_true_arr = np.asarray(reg_true, dtype=np.float64)
            reg_pred_arr = np.asarray(reg_pred, dtype=np.float64)
            if reg_true_arr.size > 0:
                reg_mae = float(np.mean(np.abs(reg_pred_arr - reg_true_arr)))
                reg_rmse = float(np.sqrt(np.mean((reg_pred_arr - reg_true_arr) ** 2)))
                sign_acc = float(np.mean(np.sign(reg_true_arr) == np.sign(reg_pred_arr)))
            macro_f1 = float("nan")
            kappa = float("nan")
            if reg_true_arr.size > 1 and np.std(reg_true_arr) > 1e-12 and np.std(reg_pred_arr) > 1e-12:
                reg_corr = float(np.corrcoef(reg_true_arr, reg_pred_arr)[0, 1])
            else:
                reg_corr = float("nan")
        else:
            _, _, f1_vals, _ = precision_recall_fscore_support(
                y_true, y_pred, average=None, labels=list(range(num_classes)), zero_division=0
            )
            macro_f1 = float(np.mean(f1_vals))
            kappa = float(cohen_kappa_score(y_true, y_pred))
            if len(reg_true) > 1 and np.std(reg_true) > 1e-12 and np.std(reg_pred) > 1e-12:
                reg_corr = float(np.corrcoef(reg_true, reg_pred)[0, 1])
            else:
                reg_corr = float("nan")

        train_losses.append(float(tr_loss))
        val_losses.append(float(va_loss))
        train_cls_losses.append(float(tr_cls_loss))
        val_cls_losses.append(float(va_cls_loss))
        train_reg_losses.append(float(tr_reg_loss))
        val_reg_losses.append(float(va_reg_loss))
        val_reg_corrs.append(reg_corr)
        val_macro_f1.append(macro_f1)
        val_kappas.append(kappa)
        if scheduler is not None:
            scheduler.step(va_loss)

        if monitor == "val_loss":
            improved = (
                va_loss < best_val_loss - 1e-6 or
                (
                    abs(va_loss - best_val_loss) <= 1e-6 and (
                        reg_corr > best_reg_corr + 1e-6 if task_type == "regression"
                        else macro_f1 > best_macro_f1 + 1e-6
                    )
                )
            )
        elif monitor == "val_reg_corr":
            improved = (
                reg_corr > best_reg_corr + 1e-6 or
                (abs(reg_corr - best_reg_corr) <= 1e-6 and va_loss < best_val_loss - 1e-6)
            )
        elif monitor == "val_kappa":
            improved = (
                kappa > best_kappa + 1e-6 or
                (abs(kappa - best_kappa) <= 1e-6 and macro_f1 > best_macro_f1 + 1e-6)
            )
        else:
            improved = (
                macro_f1 > best_macro_f1 + 1e-6 or
                (abs(macro_f1 - best_macro_f1) <= 1e-6 and va_loss < best_val_loss - 1e-6)
            )
        if improved:
            torch.save(model.state_dict(), save_path)
            best_val_loss = float(va_loss)
            best_macro_f1 = macro_f1
            best_kappa = kappa
            best_reg_corr = reg_corr
            wait = 0
        else:
            wait += 1

        dt = datetime.now() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        reg_msg = ""
        if np.any(np.asarray(train_reg_losses) > 0):
            reg_corr_str = f"{reg_corr:.4f}" if np.isfinite(reg_corr) else "nan"
            reg_msg = (
                f" | Train reg {tr_reg_loss:.4f} | Val reg {va_reg_loss:.4f} "
                f"| Val reg corr {reg_corr_str}"
            )
        if task_type == "regression":
            best_reg_corr_str = f"{best_reg_corr:.4f}" if np.isfinite(best_reg_corr) else "nan"
            reg_rmse_str = f"{reg_rmse:.6f}" if np.isfinite(reg_rmse) else "nan"
            reg_mae_str = f"{reg_mae:.6f}" if np.isfinite(reg_mae) else "nan"
            sign_acc_str = f"{sign_acc:.4f}" if np.isfinite(sign_acc) else "nan"
            print(
                f"{desc} {ep+1}/{epochs} | Train {tr_loss:.4f} | Val {va_loss:.4f} | "
                f"Val corr {reg_corr_str} | Val RMSE {reg_rmse_str} | Val MAE {reg_mae_str} | "
                f"Sign acc {sign_acc_str} | Best corr {best_reg_corr_str}{reg_msg} | "
                f"LR {current_lr:.2e} | Δt {dt}"
            )
        else:
            print(f"{desc} {ep+1}/{epochs} | Train {tr_loss:.4f} | Val {va_loss:.4f} | "
                  f"Val macro-F1 {macro_f1:.4f} | Val κ {kappa:.4f} | "
                  f"Best macro-F1 {best_macro_f1:.4f} | Best κ {best_kappa:.4f}{reg_msg} | "
                  f"LR {current_lr:.2e} | Δt {dt}")

        if ep + 1 >= min_epochs and wait >= patience:
            print(f"Early stopping at epoch {ep+1} (patience={patience}).")
            break

    return (
        np.asarray(train_losses, dtype=np.float32),
        np.asarray(val_losses, dtype=np.float32),
        np.asarray(train_cls_losses, dtype=np.float32),
        np.asarray(val_cls_losses, dtype=np.float32),
        np.asarray(train_reg_losses, dtype=np.float32),
        np.asarray(val_reg_losses, dtype=np.float32),
        np.asarray(val_reg_corrs, dtype=np.float32),
        np.asarray(val_macro_f1, dtype=np.float32),
        np.asarray(val_kappas, dtype=np.float32),
    )


def evaluate(model, test_loader, device):
    model.eval()
    if getattr(model, "task_type", "classification") == "regression":
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(device, dtype=torch.float)
                if len(batch) < 3:
                    raise ValueError("Regression evaluation requires continuous targets")
                reg_targets = batch[2]
                outputs = model(inputs)
                preds = outputs[1] if isinstance(outputs, tuple) else outputs
                y_true.extend(reg_targets.numpy())
                y_pred.extend(preds.cpu().numpy())
        return np.array(y_true, dtype=np.float32), np.array(y_pred, dtype=np.float32), None
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            targets = batch[1]
            inputs = batch[0]
            inputs = inputs.to(device, dtype=torch.float)
            logits, _ = unpack_model_output(model(inputs))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            y_true.extend(targets.numpy())
            y_pred.extend(probs.argmax(axis=1))
            y_probs.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def compute_metrics(y_true, y_pred, num_classes: int = 3):
    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                      average=None, labels=list(range(num_classes)), zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(y_true, y_pred,
                        average="weighted", zero_division=0)
    return {"accuracy": acc, "kappa": kappa, "mcc": mcc, "f1": f1, "f1_w": f1_w}


def compute_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual))) if y_true.size else float("nan")
    rmse = float(np.sqrt(np.mean(residual ** 2))) if y_true.size else float("nan")
    sign_accuracy = float(np.mean(np.sign(y_true) == np.sign(y_pred))) if y_true.size else float("nan")
    if y_true.size > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    if y_true.size > 1 and np.std(y_true) > 1e-12:
        ss_res = float(np.sum(residual ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - ss_res / max(ss_tot, 1e-12))
    else:
        r2 = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "corr": corr,
        "r2": r2,
        "sign_accuracy": sign_accuracy,
        "mean_true": float(np.mean(y_true)) if y_true.size else float("nan"),
        "mean_pred": float(np.mean(y_pred)) if y_pred.size else float("nan"),
    }


def summarize_prediction_distribution(y_pred, num_classes: int = 3, class_names: list[str] | None = None) -> dict:
    y_pred = np.asarray(y_pred, dtype=np.int64)
    counts = np.bincount(y_pred, minlength=num_classes)
    total = int(counts.sum())
    shares = counts.astype(np.float64) / max(total, 1)
    nonzero = shares > 0
    if class_names is None:
        class_names = class_names_for_num_classes(num_classes)
    normalized_entropy = 0.0
    if np.any(nonzero):
        normalized_entropy = float(
            -np.sum(shares[nonzero] * np.log(shares[nonzero])) / np.log(float(max(num_classes, 2)))
        )
    dominant_class = int(np.argmax(counts)) if total else 0
    dominant_share = float(shares[dominant_class]) if total else 0.0
    unique_pred_classes = int(np.count_nonzero(counts))
    single_class_collapse = bool(total > 0 and unique_pred_classes == 1)
    return {
        "class_counts": counts.astype(int).tolist(),
        "class_shares": shares.tolist(),
        "dominant_class": dominant_class,
        "dominant_label": class_names[dominant_class] if dominant_class < len(class_names) else str(dominant_class),
        "dominant_share": dominant_share,
        "unique_pred_classes": unique_pred_classes,
        "single_class_collapse": single_class_collapse,
        "near_single_class": bool(single_class_collapse or dominant_share >= 0.85),
        "normalized_entropy": normalized_entropy,
    }


def summarize_regression_predictions(y_pred, atol: float = 1e-6) -> dict:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.size == 0:
        return {
            "mean_pred": float("nan"),
            "std_pred": float("nan"),
            "positive_share": float("nan"),
            "negative_share": float("nan"),
            "near_zero_share": float("nan"),
        }
    return {
        "mean_pred": float(np.mean(y_pred)),
        "std_pred": float(np.std(y_pred)),
        "positive_share": float(np.mean(y_pred > atol)),
        "negative_share": float(np.mean(y_pred < -atol)),
        "near_zero_share": float(np.mean(np.abs(y_pred) <= atol)),
    }


def transfer_mode_candidates(requested_mode: str, horizon_k: int) -> list[str]:
    if requested_mode != "auto":
        return [requested_mode]
    return ["head", "lstm_conv3", "all"]


def set_transfer_trainable(model: nn.Module, transfer_mode: str):
    for param in model.parameters():
        param.requires_grad = False
    if transfer_mode == "all":
        transfer_params = list(model.parameters())
    elif transfer_mode == "lstm_conv3":
        transfer_params = model.get_lstm_conv3_params()
    else:
        transfer_params = model.get_head_params()
    for param in transfer_params:
        param.requires_grad = True
    return transfer_params


def _finite_max(values, default: float = -np.inf) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.max(arr))


def _finite_argmax_epoch(values, default_epoch: int = 1) -> int:
    arr = np.asarray(values, dtype=np.float64)
    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return int(default_epoch)
    best_local = finite_idx[int(np.argmax(arr[finite_idx]))]
    return int(best_local) + 1


def best_monitor_value(monitor: str, val_losses, val_macro_f1, val_kappas, val_reg_corrs=None) -> float:
    if monitor == "val_loss":
        return -float(np.min(val_losses))
    if monitor == "val_kappa":
        return _finite_max(val_kappas)
    if monitor == "val_reg_corr":
        return _finite_max([] if val_reg_corrs is None else val_reg_corrs)
    return _finite_max(val_macro_f1)

def best_monitor_epoch(monitor: str, val_losses, val_macro_f1, val_kappas, val_reg_corrs=None) -> int:
    if monitor == "val_loss":
        return int(np.argmin(val_losses)) + 1
    if monitor == "val_kappa":
        return _finite_argmax_epoch(val_kappas)
    if monitor == "val_reg_corr":
        return _finite_argmax_epoch([] if val_reg_corrs is None else val_reg_corrs)
    return _finite_argmax_epoch(val_macro_f1)


def loader_kwargs(num_workers: int, use_pin_memory: bool) -> dict:
    kwargs = {"num_workers": num_workers, "pin_memory": use_pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def monitor_label(monitor: str) -> str:
    return {
        "val_loss": "Val loss",
        "val_macro_f1": "Val macro-F1",
        "val_kappa": "Val kappa",
        "val_reg_corr": "Val corr",
    }.get(monitor, monitor)


def monitor_series(monitor: str, val_losses, val_macro_f1=None, val_kappas=None, val_reg_corrs=None):
    if monitor == "val_loss":
        return np.asarray(val_losses, dtype=np.float64), monitor_label(monitor)
    if monitor == "val_kappa":
        return np.asarray([] if val_kappas is None else val_kappas, dtype=np.float64), monitor_label(monitor)
    if monitor == "val_reg_corr":
        return np.asarray([] if val_reg_corrs is None else val_reg_corrs, dtype=np.float64), monitor_label(monitor)
    return np.asarray([] if val_macro_f1 is None else val_macro_f1, dtype=np.float64), monitor_label(monitor)


def plot_loss_curve(train_losses, val_losses, title, save_path, monitor="val_loss",
                    val_macro_f1=None, val_kappas=None, val_reg_corrs=None):
    train_losses = np.asarray(train_losses, dtype=np.float64)
    val_losses = np.asarray(val_losses, dtype=np.float64)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(epochs, train_losses, label="Train loss", color="#1d4ed8", lw=2.0)
    ax.plot(epochs, val_losses, label="Val loss", color="#dc2626", lw=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.2, axis="y")

    handles, labels = ax.get_legend_handles_labels()
    monitor_values, monitor_name = monitor_series(
        monitor,
        val_losses,
        val_macro_f1=val_macro_f1,
        val_kappas=val_kappas,
        val_reg_corrs=val_reg_corrs,
    )
    if len(monitor_values) == len(epochs):
        best_epoch = best_monitor_epoch(monitor, val_losses, val_macro_f1, val_kappas, val_reg_corrs)
        if 1 <= best_epoch <= len(epochs):
            ax.axvline(best_epoch, color="#6b7280", linestyle=":", linewidth=1.2, alpha=0.9)
        if monitor == "val_loss":
            if 1 <= best_epoch <= len(epochs) and np.isfinite(val_losses[best_epoch - 1]):
                ax.scatter(
                    [best_epoch],
                    [val_losses[best_epoch - 1]],
                    color="#991b1b",
                    s=36,
                    zorder=5,
                    label=f"Best {monitor_name}",
                )
                handles, labels = ax.get_legend_handles_labels()
        else:
            ax2 = ax.twinx()
            ax2.plot(epochs, monitor_values, label=monitor_name, color="#d97706", lw=2.0)
            ax2.set_ylabel(monitor_name)
            if 1 <= best_epoch <= len(epochs) and np.isfinite(monitor_values[best_epoch - 1]):
                ax2.scatter(
                    [best_epoch],
                    [monitor_values[best_epoch - 1]],
                    color="#92400e",
                    s=36,
                    zorder=5,
                    label=f"Best {monitor_name}",
                )
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles.extend(handles2)
            labels.extend(labels2)

    ax.legend(handles, labels, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_transfer_comparison(results_before, results_after, save_path, num_classes: int):
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
    ax.legend(); ax.axhline(1.0 / float(num_classes), color="gray", linestyle="--", label="random")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_transfer_regimes(base_results, transfer_results, specific_results, save_path, num_classes: int):
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
    ax.axhline(1.0 / float(num_classes), color="gray", linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_cm_grid(results_list, save_path, class_names: list[str], title="Confusion Matrices"):
    n = len(results_list)
    if n == 0:
        return
    cols = min(5, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]
    display_labels = [name.replace(" ", "\n") for name in class_names]
    label_ids = list(range(len(class_names)))
    for i, res in enumerate(results_list):
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=label_ids)
        row_sums = np.clip(cm.sum(axis=1, keepdims=True), 1.0, None)
        cmn = cm.astype(float) / row_sums
        ConfusionMatrixDisplay(cmn, display_labels=display_labels).plot(
            ax=axes[i], colorbar=False)
        axes[i].set_title(res.get("label", f"s{i}"), fontsize=9)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(save_path, dpi=100); plt.close(fig)


def plot_regression_scatter_grid(results_list, save_path, title="Predicted vs Actual Log Returns"):
    n = len(results_list)
    if n == 0:
        return
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]
    for i, res in enumerate(results_list):
        y_true = np.asarray(res["y_true"], dtype=np.float64)
        y_pred = np.asarray(res["y_pred"], dtype=np.float64)
        ax = axes[i]
        ax.scatter(y_true, y_pred, s=8, alpha=0.25, color="tab:blue")
        lo = float(min(np.min(y_true), np.min(y_pred))) if y_true.size else -1.0
        hi = float(max(np.max(y_true), np.max(y_pred))) if y_true.size else 1.0
        ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1.0)
        metrics = res.get("metrics", {})
        corr = metrics.get("corr")
        rmse = metrics.get("rmse")
        corr_str = f"corr={corr:.3f}" if corr is not None and np.isfinite(corr) else "corr=nan"
        rmse_str = f"rmse={rmse:.4f}" if rmse is not None and np.isfinite(rmse) else "rmse=nan"
        ax.set_title(f"{res.get('label', f's{i}')}\n{corr_str}, {rmse_str}", fontsize=9)
        ax.set_xlabel("True log return")
        ax.set_ylabel("Predicted log return")
        ax.grid(alpha=0.2)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_transfer_comparison_regression(results_before, results_after, save_path):
    stocks = [r["stock_id"] for r in results_before]
    corr_before = [r["metrics"]["corr"] for r in results_before]
    corr_after = [r["metrics"]["corr"] for r in results_after]
    sign_before = [r["metrics"]["sign_accuracy"] for r in results_before]
    sign_after = [r["metrics"]["sign_accuracy"] for r in results_after]

    x = np.arange(len(stocks))
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(stocks) * 0.7), 8), sharex=True)
    axes[0].bar(x - 0.2, corr_before, 0.4, label="Base model (frozen)")
    axes[0].bar(x + 0.2, corr_after, 0.4, label="Fine-tuned (transfer)")
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Pearson corr")
    axes[0].set_title("Transfer Learning: Base vs Fine-tuned (return correlation)")
    axes[0].legend()
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].bar(x - 0.2, sign_before, 0.4, label="Base model (frozen)")
    axes[1].bar(x + 0.2, sign_after, 0.4, label="Fine-tuned (transfer)")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="random sign")
    axes[1].set_ylabel("Directional accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"s{s}" for s in stocks], rotation=45, fontsize=8)
    axes[1].legend()
    axes[1].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_transfer_regimes_regression(base_results, transfer_results, specific_results, save_path):
    base_corr = {r["stock_id"]: r["metrics"]["corr"] for r in base_results}
    transfer_corr = {r["stock_id"]: r["after"]["corr"] for r in transfer_results}
    specific_corr = {r["stock_id"]: r["metrics"]["corr"] for r in specific_results}
    base_sign = {r["stock_id"]: r["metrics"]["sign_accuracy"] for r in base_results}
    transfer_sign = {r["stock_id"]: r["after"]["sign_accuracy"] for r in transfer_results}
    specific_sign = {r["stock_id"]: r["metrics"]["sign_accuracy"] for r in specific_results}
    stocks = sorted(set(base_corr) & set(transfer_corr) & set(specific_corr))
    if not stocks:
        return

    x = np.arange(len(stocks))
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(stocks) * 0.7), 8), sharex=True)
    axes[0].bar(x - width, [base_corr[s] for s in stocks], width, label="Universal zero-shot")
    axes[0].bar(x, [transfer_corr[s] for s in stocks], width, label="Fine-tuned transfer")
    axes[0].bar(x + width, [specific_corr[s] for s in stocks], width, label="Specific-stock OOS")
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Pearson corr")
    axes[0].set_title("Transfer Regimes on Held-out Stocks (return correlation)")
    axes[0].legend()
    axes[0].grid(alpha=0.2, axis="y")

    axes[1].bar(x - width, [base_sign[s] for s in stocks], width, label="Universal zero-shot")
    axes[1].bar(x, [transfer_sign[s] for s in stocks], width, label="Fine-tuned transfer")
    axes[1].bar(x + width, [specific_sign[s] for s in stocks], width, label="Specific-stock OOS")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Directional accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"s{s}" for s in stocks], rotation=45, fontsize=8)
    axes[1].grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


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
    print(f"Training profile: {args.horizon_profile}")

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
    if "split_mode" in split or "norm_mode" in split:
        print(
            f"Prepared split metadata: split_mode={split.get('split_mode', 'unknown')} "
            f"holdout={split.get('num_transfer_stocks', len(transfer_ids))} "
            f"norm_mode={split.get('norm_mode', 'unknown')} "
            f"norm_clip={split.get('norm_clip', 'n/a')}"
        )

    horizon_k = HORIZONS[args.horizon]
    cfg = resolve_training_config(args, horizon_k)
    transfer_eval_ids = choose_transfer_eval_ids(transfer_ids, horizon_k, cfg)
    base_stock_ids, base_prefix_frac_by_stock = build_base_stock_plan(
        train_ids, transfer_ids, transfer_eval_ids, cfg
    )

    T = cfg["lookback"]
    tag = f"k{horizon_k}"
    print(
        f"Resolved config for k={horizon_k}: lr={cfg['lr']} transfer_lr={cfg['transfer_lr']} "
        f"dropout={cfg['dropout']} balance={cfg['balance_mode']} loss={cfg['loss_type']} "
        f"label_mode={cfg['label_mode']} task_type={cfg['task_type']} aux_reg_weight={cfg['aux_reg_weight']} "
        f"transfer_mode={cfg['transfer_mode']} transfer_stock_selector={cfg['transfer_stock_selector']} "
        f"base_sample_mode={cfg['base_sample_mode']} "
        f"base_stock_scope={cfg['base_stock_scope']} transfer_tail_frac={cfg['transfer_tail_frac']:.2f}"
    )
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
        print(f"  Building combined dataset from {len(base_stock_ids)} base-training stocks...")
        train_ds, val_ds = concat_datasets(
            base_stock_ids,
            horizon_k,
            T,
            val_frac=cfg["base_val_frac"],
            max_per_stock=cfg["base_max_per_stock"],
            sample_mode=cfg["base_sample_mode"],
            sample_bins=cfg["base_sample_bins"],
            val_mode=cfg["base_val_mode"],
            label_mode=cfg["label_mode"],
            quantile_stationary=cfg["quantile_stationary"],
            return_regression=uses_regression_targets(cfg),
            prefix_frac_by_stock=base_prefix_frac_by_stock,
        )
        if train_ds is None:
            print("ERROR: No training data found. Check data/optiver_processed/.")
            sys.exit(1)

        train_loader = data.DataLoader(
            train_ds,
            batch_size=cfg["batch_size"],
            sampler=sampler_for_mode(train_ds, cfg["balance_mode"], cfg["num_classes"]),
            shuffle=loader_shuffle_for_mode(cfg["balance_mode"]),
            **loader_kwargs(num_workers, use_pin_memory),
        )
        val_loader = data.DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            **loader_kwargs(num_workers, use_pin_memory),
        )

        model = build_optiver_model(cfg).to(device)
        class_weights, class_counts = class_weight_for_mode(train_ds, cfg["balance_mode"], device, cfg["num_classes"])
        reg_stats = make_regression_stats(train_ds)
        print(
            f"  Base train class counts: {class_counts.tolist()}  "
            f"loss_weights: {class_weights.detach().cpu().tolist() if class_weights is not None else 'disabled'}"
        )
        criterion = build_optiver_criterion(cfg, class_weights, reg_stats)
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg["lr_plateau_factor"],
            patience=cfg["lr_plateau_patience"],
            min_lr=1e-6,
        )

        tr_losses, va_losses, tr_cls_losses, va_cls_losses, tr_reg_losses, va_reg_losses, va_reg_corr, va_macro_f1, va_kappa = train_loop(
            model,
            criterion,
            optimizer,
            train_loader,
            val_loader,
            cfg["epochs"],
            base_tmp_path,
            device,
            desc="Base training",
            patience=cfg["patience"],
            min_epochs=cfg["min_epochs"],
            grad_clip=cfg["grad_clip"],
            scheduler=scheduler,
            monitor=cfg["monitor"],
            task_type=cfg["task_type"],
            num_classes=cfg["num_classes"],
        )

        # Load best checkpoint
        model.load_state_dict(torch.load(base_tmp_path, map_location=device))
        torch.save(model, base_path)
        os.replace(base_tmp_path, base_state_path)

        plot_loss_curve(
            tr_losses,
            va_losses,
            f"Base Training Loss (k={horizon_k}, monitor={cfg['monitor']})",
            os.path.join(RESULT_DIR, f"base_loss_{tag}.png"),
            monitor=cfg["monitor"],
            val_macro_f1=va_macro_f1,
            val_kappas=va_kappa,
            val_reg_corrs=va_reg_corr,
        )

        # Evaluate on held-out transfer stocks
        print("\n  Evaluating base model on transfer stocks (zero-shot)...")
        base_metrics = {
            "per_stock": [],
            "horizon_k": horizon_k,
            "task_type": cfg["task_type"],
            "num_classes": cfg["num_classes"],
            "class_names": cfg["class_names"],
            "train_losses": tr_losses,
            "val_losses": va_losses,
            "train_cls_losses": tr_cls_losses,
            "val_cls_losses": va_cls_losses,
            "train_reg_losses": tr_reg_losses,
            "val_reg_losses": va_reg_losses,
            "val_reg_corr": va_reg_corr,
            "val_macro_f1": va_macro_f1,
            "val_kappa": va_kappa,
            "balance_mode": cfg["balance_mode"],
            "loss_type": cfg["loss_type"],
            "monitor": cfg["monitor"],
            "label_mode": cfg["label_mode"],
            "quantile_stationary": cfg["quantile_stationary"],
            "dropout": cfg["dropout"],
            "aux_reg_loss": cfg["aux_reg_loss"],
            "aux_reg_weight": cfg["aux_reg_weight"],
            "base_max_per_stock": cfg["base_max_per_stock"],
            "base_sample_mode": cfg["base_sample_mode"],
            "base_sample_bins": cfg["base_sample_bins"],
            "base_val_mode": cfg["base_val_mode"],
            "base_val_frac": cfg["base_val_frac"],
            "transfer_max_samples": cfg["transfer_max_samples"],
            "split_metadata": split,
        }
        diagnostic_results = []
        for sid in transfer_eval_ids:
            ds = load_stock_dataset(
                sid,
                horizon_k,
                T,
                max_samples=cfg["transfer_max_samples"],
                label_mode=cfg["label_mode"],
                quantile_stationary=cfg["quantile_stationary"],
                return_regression=uses_regression_targets(cfg),
            )
            if ds is not None and cfg["base_stock_scope"] == "all-stocks":
                ds = dataset_tail_subset(ds, cfg["transfer_tail_frac"])
            if ds is None or len(ds) < 50:
                continue
            loader = data.DataLoader(
                ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )
            y_true, y_pred, y_aux, m, pred_summary = evaluate_task_predictions(
                model, loader, device, cfg["task_type"], cfg["num_classes"], cfg["class_names"]
            )
            time_id = dataset_sample_time_id(ds)
            sample_end = dataset_sample_end(ds).copy()
            record = {
                "stock_id": sid,
                "metrics": m,
                "y_true": y_true,
                "y_pred": y_pred,
                "time_id": time_id,
                "sample_end": sample_end,
                "prediction_summary": pred_summary,
            }
            if y_aux is not None:
                record["y_probs"] = y_aux
            base_metrics["per_stock"].append(record)
            diagnostic_results.append({
                "label": f"stock {sid}",
                "y_true": y_true,
                "y_pred": y_pred,
                "metrics": m,
            })
            if cfg["task_type"] == "regression":
                print(
                    f"    stock {sid}: corr={m['corr']:.4f}  rmse={m['rmse']:.6f}  "
                    f"sign_acc={m['sign_accuracy']:.4f}"
                )
            else:
                print(
                    f"    stock {sid}: acc={m['accuracy']:.4f}  κ={m['kappa']:.4f}  "
                    f"dominant={pred_summary['dominant_label']} ({pred_summary['dominant_share']:.3f})"
                )

        with open(base_metrics_path, "wb") as f:
            pickle.dump(base_metrics, f)

        if diagnostic_results:
            if cfg["task_type"] == "regression":
                plot_regression_scatter_grid(
                    diagnostic_results,
                    os.path.join(RESULT_DIR, f"cm_base_{tag}.png"),
                    title="Base Model on Transfer Stocks (zero-shot regression)",
                )
            else:
                plot_cm_grid(
                    diagnostic_results,
                    os.path.join(RESULT_DIR, f"cm_base_{tag}.png"),
                    cfg["class_names"],
                    title="Base Model on Transfer Stocks (zero-shot)",
                )

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
        transfer_results = {
            "per_stock": [],
            "horizon_k": horizon_k,
            "task_type": cfg["task_type"],
            "num_classes": cfg["num_classes"],
            "class_names": cfg["class_names"],
        }
        results_before   = []
        results_after    = []

        for sid in transfer_eval_ids:
            ds = load_stock_dataset(
                sid,
                horizon_k,
                T,
                max_samples=cfg["transfer_max_samples"],
                label_mode=cfg["label_mode"],
                quantile_stationary=cfg["quantile_stationary"],
                return_regression=uses_regression_targets(cfg),
            )
            if ds is not None and cfg["base_stock_scope"] == "all-stocks":
                ds = dataset_tail_subset(ds, cfg["transfer_tail_frac"])
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
                sampler=sampler_for_mode(ft_train, cfg["balance_mode"], cfg["num_classes"]),
                shuffle=loader_shuffle_for_mode(cfg["balance_mode"]),
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
                batch_size=cfg["batch_size"],
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )

            base_model = torch.load(base_path, map_location=device)

            # Evaluate BEFORE fine-tuning
            y_true_b, y_pred_b, y_aux_b, m_before, base_prediction_summary = evaluate_task_predictions(
                base_model, test_loader, device, cfg["task_type"], cfg["num_classes"], cfg["class_names"]
            )
            results_before.append({"stock_id": sid, "metrics": m_before, "prediction_summary": base_prediction_summary})
            if cfg["task_type"] == "regression":
                print(
                    f"  stock {sid} zero-shot: corr={m_before['corr']:.4f}  rmse={m_before['rmse']:.6f}  "
                    f"sign_acc={m_before['sign_accuracy']:.4f}"
                )
            else:
                print(
                    f"  stock {sid} zero-shot: acc={m_before['accuracy']:.4f}  κ={m_before['kappa']:.4f}  "
                    f"dominant={base_prediction_summary['dominant_label']} ({base_prediction_summary['dominant_share']:.3f})"
                )
            del base_model

            ft_class_weights, ft_class_counts = class_weight_for_mode(
                ft_train, cfg["balance_mode"], device, cfg["num_classes"]
            )
            ft_reg_stats = make_regression_stats(ft_train)
            print(
                f"  stock {sid} fine-tune class counts: {ft_class_counts.tolist()}  "
                f"loss_weights: {ft_class_weights.detach().cpu().tolist() if ft_class_weights is not None else 'disabled'}"
            )
            requested_transfer_mode = cfg["transfer_mode"]
            candidate_modes = transfer_mode_candidates(requested_transfer_mode, horizon_k)
            mode_rank = {mode: rank for rank, mode in enumerate(candidate_modes)}
            candidate_summaries = []
            best_candidate = None
            best_state_path = None
            ft_model_path = os.path.join(MODEL_DIR, f"optiver_transfer_s{sid}_{tag}_model.pt")
            final_state_path = os.path.join(MODEL_DIR, f"optiver_transfer_s{sid}_{tag}.pt")

            for candidate_mode in candidate_modes:
                model = torch.load(base_path, map_location=device)
                set_transfer_trainable(model, candidate_mode)
                criterion = build_optiver_criterion(cfg, ft_class_weights, ft_reg_stats)
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=cfg["transfer_lr"],
                    weight_decay=cfg["weight_decay"],
                )
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=cfg["lr_plateau_factor"],
                    patience=cfg["lr_plateau_patience"],
                    min_lr=1e-6,
                )
                candidate_state_path = os.path.join(MODEL_DIR, f"optiver_transfer_s{sid}_{tag}_{candidate_mode}.tmp")
                ft_tr_losses_c, ft_va_losses_c, ft_tr_cls_losses_c, ft_va_cls_losses_c, ft_tr_reg_losses_c, ft_va_reg_losses_c, ft_va_reg_corr_c, ft_va_macro_f1_c, ft_va_kappa_c = train_loop(
                    model,
                    criterion,
                    optimizer,
                    ft_loader,
                    ftv_loader,
                    cfg["transfer_epochs"],
                    candidate_state_path,
                    device,
                    desc=f"TL stock {sid} [{candidate_mode}]",
                    patience=cfg["patience"],
                    min_epochs=min(cfg["min_epochs"], 3),
                    grad_clip=cfg["grad_clip"],
                    scheduler=scheduler,
                    monitor=cfg["monitor"],
                    task_type=cfg["task_type"],
                    num_classes=cfg["num_classes"],
                )

                model.load_state_dict(torch.load(candidate_state_path, map_location=device))
                y_true_c, y_pred_c, _, m_candidate, candidate_prediction_summary = evaluate_task_predictions(
                    model, test_loader, device, cfg["task_type"], cfg["num_classes"], cfg["class_names"]
                )
                selection_score = best_monitor_value(
                    cfg["monitor"], ft_va_losses_c, ft_va_macro_f1_c, ft_va_kappa_c, ft_va_reg_corr_c
                )
                candidate_summary = {
                    "transfer_mode": candidate_mode,
                    "selection_score": selection_score,
                    "best_epoch": best_monitor_epoch(
                        cfg["monitor"], ft_va_losses_c, ft_va_macro_f1_c, ft_va_kappa_c, ft_va_reg_corr_c
                    ),
                    "best_val_loss": float(np.min(ft_va_losses_c)),
                    "best_val_macro_f1": _finite_max(ft_va_macro_f1_c),
                    "best_val_kappa": _finite_max(ft_va_kappa_c),
                    "best_val_reg_corr": _finite_max(ft_va_reg_corr_c),
                    "test_metrics": m_candidate,
                    "prediction_summary": candidate_prediction_summary,
                }
                candidate_summaries.append(candidate_summary)
                if cfg["task_type"] == "regression":
                    print(
                        f"    mode={candidate_mode}: select_score={selection_score:.4f}  "
                        f"test_corr={m_candidate['corr']:.4f}  test_rmse={m_candidate['rmse']:.6f}  "
                        f"sign_acc={m_candidate['sign_accuracy']:.4f}"
                    )
                else:
                    print(
                        f"    mode={candidate_mode}: select_score={selection_score:.4f}  "
                        f"test_acc={m_candidate['accuracy']:.4f}  "
                        f"dominant={candidate_prediction_summary['dominant_label']} ({candidate_prediction_summary['dominant_share']:.3f})"
                    )

                is_better = (
                    best_candidate is None
                    or selection_score > best_candidate["selection_score"] + 1e-12
                    or (
                        np.isclose(selection_score, best_candidate["selection_score"])
                        and mode_rank[candidate_mode] < mode_rank[best_candidate["transfer_mode"]]
                    )
                )
                if is_better:
                    if best_state_path and os.path.exists(best_state_path):
                        os.remove(best_state_path)
                    best_candidate = {
                        **candidate_summary,
                        "train_losses": ft_tr_losses_c,
                        "val_losses": ft_va_losses_c,
                        "train_cls_losses": ft_tr_cls_losses_c,
                        "val_cls_losses": ft_va_cls_losses_c,
                        "train_reg_losses": ft_tr_reg_losses_c,
                        "val_reg_losses": ft_va_reg_losses_c,
                        "val_reg_corr": ft_va_reg_corr_c,
                        "val_macro_f1": ft_va_macro_f1_c,
                        "val_kappa": ft_va_kappa_c,
                    }
                    best_state_path = candidate_state_path
                else:
                    if os.path.exists(candidate_state_path):
                        os.remove(candidate_state_path)

                del model
                gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if best_candidate is None or best_state_path is None:
                print(f"  stock {sid}: no successful transfer candidate")
                continue

            if os.path.exists(final_state_path):
                os.remove(final_state_path)
            os.replace(best_state_path, final_state_path)

            model = torch.load(base_path, map_location=device)
            model.load_state_dict(torch.load(final_state_path, map_location=device))
            torch.save(model, ft_model_path)

            # Evaluate AFTER fine-tuning with the selected mode.
            y_true_a, y_pred_a, y_aux_a, m_after, after_prediction_summary = evaluate_task_predictions(
                model, test_loader, device, cfg["task_type"], cfg["num_classes"], cfg["class_names"]
            )
            results_after.append({"stock_id": sid, "metrics": m_after, "prediction_summary": after_prediction_summary})

            sample_end = dataset_sample_end(ds)
            sample_time_id = dataset_sample_time_id(ds)
            test_end = sample_end[n_ft:n_total].copy()
            test_time_id = sample_time_id[n_ft:n_total].copy()

            if cfg["task_type"] == "regression":
                print(
                    f"  stock {sid}: corr before={m_before['corr']:.4f} → after={m_after['corr']:.4f} "
                    f"(Δ={m_after['corr']-m_before['corr']:+.4f})  selected_mode={best_candidate['transfer_mode']}"
                )
            else:
                print(f"  stock {sid}: "
                      f"acc before={m_before['accuracy']:.4f} → after={m_after['accuracy']:.4f} "
                      f"(Δ={m_after['accuracy']-m_before['accuracy']:+.4f})  "
                      f"selected_mode={best_candidate['transfer_mode']}")

            transfer_record = {
                "stock_id": sid,
                "before": m_before,
                "after":  m_after,
                "y_true": y_true_a,
                "y_pred_before": y_pred_b,
                "y_pred_after": y_pred_a,
                "time_id": test_time_id,
                "sample_end": test_end,
                "train_losses": best_candidate["train_losses"],
                "val_losses": best_candidate["val_losses"],
                "train_cls_losses": best_candidate["train_cls_losses"],
                "val_cls_losses": best_candidate["val_cls_losses"],
                "train_reg_losses": best_candidate["train_reg_losses"],
                "val_reg_losses": best_candidate["val_reg_losses"],
                "val_reg_corr": best_candidate["val_reg_corr"],
                "val_macro_f1": best_candidate["val_macro_f1"],
                "val_kappa": best_candidate["val_kappa"],
                "balance_mode": cfg["balance_mode"],
                "loss_type": cfg["loss_type"],
                "monitor": cfg["monitor"],
                "label_mode": cfg["label_mode"],
                "quantile_stationary": cfg["quantile_stationary"],
                "transfer_mode": best_candidate["transfer_mode"],
                "requested_transfer_mode": requested_transfer_mode,
                "selected_transfer_mode": best_candidate["transfer_mode"],
                "candidate_modes": candidate_summaries,
                "base_prediction_summary": base_prediction_summary,
                "after_prediction_summary": after_prediction_summary,
                "aux_reg_loss": cfg["aux_reg_loss"],
                "aux_reg_weight": cfg["aux_reg_weight"],
            }
            if y_aux_b is not None:
                transfer_record["y_probs_before"] = y_aux_b
            if y_aux_a is not None:
                transfer_record["y_probs_after"] = y_aux_a
            if cfg["task_type"] != "regression":
                transfer_record["base_collapse"] = base_prediction_summary
                transfer_record["after_collapse"] = after_prediction_summary
            transfer_results["per_stock"].append(transfer_record)

            del model, test_ds
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        with open(transfer_path, "wb") as f:
            pickle.dump(transfer_results, f)

        if results_before and results_after:
            if cfg["task_type"] == "regression":
                plot_transfer_comparison_regression(
                    results_before, results_after,
                    os.path.join(RESULT_DIR, f"transfer_comparison_{tag}.png"))
            else:
                plot_transfer_comparison(
                    results_before, results_after,
                    os.path.join(RESULT_DIR, f"transfer_comparison_{tag}.png"),
                    cfg["num_classes"],
                )

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
        specific_results = {
            "per_stock": [],
            "horizon_k": horizon_k,
            "task_type": cfg["task_type"],
            "num_classes": cfg["num_classes"],
            "class_names": cfg["class_names"],
        }
        for sid in transfer_eval_ids:
            ds = load_stock_dataset(
                sid,
                horizon_k,
                T,
                max_samples=cfg["transfer_max_samples"],
                label_mode=cfg["label_mode"],
                quantile_stationary=cfg["quantile_stationary"],
                return_regression=uses_regression_targets(cfg),
            )
            if ds is not None and cfg["base_stock_scope"] == "all-stocks":
                ds = dataset_tail_subset(ds, cfg["transfer_tail_frac"])
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
                batch_size=min(cfg["batch_size"], len(train_ds)),
                sampler=sampler_for_mode(train_ds, cfg["balance_mode"], cfg["num_classes"]),
                shuffle=loader_shuffle_for_mode(cfg["balance_mode"]),
                **loader_kwargs(num_workers, use_pin_memory),
            )
            val_loader = data.DataLoader(
                val_ds,
                batch_size=min(cfg["batch_size"], len(val_ds)),
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )
            test_loader = data.DataLoader(
                test_ds,
                batch_size=cfg["batch_size"],
                shuffle=False,
                **loader_kwargs(num_workers, use_pin_memory),
            )

            model = build_optiver_model(cfg).to(device)
            sp_class_weights, sp_class_counts = class_weight_for_mode(
                train_ds, cfg["balance_mode"], device, cfg["num_classes"]
            )
            sp_reg_stats = make_regression_stats(train_ds)
            print(
                f"  stock {sid} specific-train class counts: {sp_class_counts.tolist()}  "
                f"loss_weights: {sp_class_weights.detach().cpu().tolist() if sp_class_weights is not None else 'disabled'}"
            )
            criterion = build_optiver_criterion(cfg, sp_class_weights, sp_reg_stats)
            optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg["lr_plateau_factor"],
                patience=cfg["lr_plateau_patience"],
                min_lr=1e-6,
            )
            save_path = os.path.join(MODEL_DIR, f"optiver_specific_s{sid}_{tag}.pt")

            sp_tr_losses, sp_va_losses, sp_tr_cls_losses, sp_va_cls_losses, sp_tr_reg_losses, sp_va_reg_losses, sp_va_reg_corr, sp_va_macro_f1, sp_va_kappa = train_loop(
                model,
                criterion,
                optimizer,
                train_loader,
                val_loader,
                cfg["specific_epochs"],
                save_path,
                device,
                desc=f"Specific stock {sid}",
                patience=cfg["patience"],
                min_epochs=cfg["min_epochs"],
                grad_clip=cfg["grad_clip"],
                scheduler=scheduler,
                monitor=cfg["monitor"],
                task_type=cfg["task_type"],
                num_classes=cfg["num_classes"],
            )

            model.load_state_dict(torch.load(save_path, map_location=device))
            y_true, y_pred, y_aux, metrics, prediction_summary = evaluate_task_predictions(
                model, test_loader, device, cfg["task_type"], cfg["num_classes"], cfg["class_names"]
            )
            sample_end = dataset_sample_end(ds)
            sample_time_id = dataset_sample_time_id(ds)
            test_end = sample_end[n_prefix:].copy()
            test_time_id = sample_time_id[n_prefix:].copy()

            record = {
                "stock_id": sid,
                "metrics": metrics,
                "y_true": y_true,
                "y_pred": y_pred,
                "time_id": test_time_id,
                "sample_end": test_end,
                "train_losses": sp_tr_losses,
                "val_losses": sp_va_losses,
                "train_cls_losses": sp_tr_cls_losses,
                "val_cls_losses": sp_va_cls_losses,
                "train_reg_losses": sp_tr_reg_losses,
                "val_reg_losses": sp_va_reg_losses,
                "val_reg_corr": sp_va_reg_corr,
                "val_macro_f1": sp_va_macro_f1,
                "val_kappa": sp_va_kappa,
                "balance_mode": cfg["balance_mode"],
                "loss_type": cfg["loss_type"],
                "monitor": cfg["monitor"],
                "label_mode": cfg["label_mode"],
                "quantile_stationary": cfg["quantile_stationary"],
                "aux_reg_loss": cfg["aux_reg_loss"],
                "aux_reg_weight": cfg["aux_reg_weight"],
                "prediction_summary": prediction_summary,
            }
            if y_aux is not None:
                record["y_probs"] = y_aux
            specific_results["per_stock"].append(record)
            if cfg["task_type"] == "regression":
                print(
                    f"  stock {sid}: specific-stock corr={metrics['corr']:.4f}  "
                    f"rmse={metrics['rmse']:.6f}  sign_acc={metrics['sign_accuracy']:.4f}"
                )
            else:
                print(f"  stock {sid}: specific-stock acc={metrics['accuracy']:.4f}  κ={metrics['kappa']:.4f}")

            del model, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

        with open(specific_path, "wb") as f:
            pickle.dump(specific_results, f)

    if cfg["task_type"] == "regression":
        plot_transfer_regimes_regression(
            base_metrics.get("per_stock", []),
            transfer_results.get("per_stock", []),
            specific_results.get("per_stock", []),
            os.path.join(RESULT_DIR, f"transfer_regimes_{tag}.png"),
        )
    else:
        plot_transfer_regimes(
            base_metrics.get("per_stock", []),
            transfer_results.get("per_stock", []),
            specific_results.get("per_stock", []),
            os.path.join(RESULT_DIR, f"transfer_regimes_{tag}.png"),
            cfg["num_classes"],
        )

    # =========================================================
    # Print summary
    # =========================================================
    print(f"\n{'='*60}")
    print("  Optiver Transfer Learning Summary")
    print(f"{'='*60}")
    rows = []
    for r in transfer_results["per_stock"]:
        if cfg["task_type"] == "regression":
            rows.append({
                "Stock": r["stock_id"],
                "Corr Before": f"{r['before']['corr']:.4f}",
                "Corr After":  f"{r['after']['corr']:.4f}",
                "Δ Corr":      f"{r['after']['corr']-r['before']['corr']:+.4f}",
                "RMSE After":  f"{r['after']['rmse']:.6f}",
                "Sign Acc":    f"{r['after']['sign_accuracy']:.4f}",
            })
        else:
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
        if cfg["task_type"] == "regression":
            specific_rows.append({
                "Stock": r["stock_id"],
                "Specific OOS Corr": f"{r['metrics']['corr']:.4f}",
                "Specific OOS RMSE": f"{r['metrics']['rmse']:.6f}",
                "Specific Sign Acc": f"{r['metrics']['sign_accuracy']:.4f}",
            })
        else:
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
