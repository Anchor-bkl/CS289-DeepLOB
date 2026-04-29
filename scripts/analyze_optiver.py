#!/usr/bin/env python3
"""
analyze_optiver.py — Paper-style transfer-learning analysis for Optiver
======================================================================
Builds notebook-ready figures analogous to DeepLOB paper Figure 6/7/8/9:
  Figure 6  — per-session accuracy boxplots on held-out transfer stocks
  Figure 7  — normalized per-session profit boxplots + t-statistics
  Figure 8  — cumulative normalized profit curves by stock and horizon
  Figure 9  — LIME-style local surrogate explanation before/after transfer
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch

try:
    from train_optiver import (
        DATA_DIR,
        MODEL_DIR,
        RESULT_DIR,
        DeepLOBLite,
        class_names_for_num_classes,
        signal_values_for_num_classes,
        unpack_model_output,
    )
except ModuleNotFoundError:
    from scripts.train_optiver import (
        DATA_DIR,
        MODEL_DIR,
        RESULT_DIR,
        DeepLOBLite,
        class_names_for_num_classes,
        signal_values_for_num_classes,
        unpack_model_output,
    )


FEATURE_LABELS = [
    "ask_p1", "ask_v1", "bid_p1", "bid_v1",
    "ask_p2", "ask_v2", "bid_p2", "bid_v2",
]
CLASS_LABELS = class_names_for_num_classes(3)
MONITOR_LABELS = {
    "val_loss": "Val loss",
    "val_macro_f1": "Val macro-F1",
    "val_kappa": "Val kappa",
    "val_reg_corr": "Val corr",
}


def monitor_label(monitor: str) -> str:
    return MONITOR_LABELS.get(monitor, monitor)


def monitor_series_from_item(item: dict):
    monitor = item.get("monitor", "val_macro_f1")
    if monitor == "val_loss":
        values = item.get("val_losses", [])
    elif monitor == "val_kappa":
        values = item.get("val_kappa", [])
    elif monitor == "val_reg_corr":
        values = item.get("val_reg_corr", [])
    else:
        values = item.get("val_macro_f1", [])
    return monitor, np.asarray(values, dtype=np.float64), monitor_label(monitor)


def best_item_monitor_epoch(item: dict) -> int:
    monitor, values, _ = monitor_series_from_item(item)
    if len(values) == 0:
        return 0
    if monitor == "val_loss":
        return int(np.argmin(values)) + 1
    finite_idx = np.flatnonzero(np.isfinite(values))
    if len(finite_idx) == 0:
        return 0
    best_local = finite_idx[int(np.argmax(values[finite_idx]))]
    return int(best_local) + 1


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def infer_task_type(transfer: dict) -> str:
    task_type = transfer.get("task_type")
    if task_type in {"classification", "regression"}:
        return task_type
    per_stock = transfer.get("per_stock", [])
    if not per_stock:
        return "classification"
    sample = per_stock[0]
    y_true = np.asarray(sample.get("y_true", []))
    if np.issubdtype(y_true.dtype, np.floating) and "y_probs_before" not in sample:
        return "regression"
    return "classification"


def analysis_task_type(all_frames: dict[int, list[dict]]) -> str:
    for items in all_frames.values():
        if items:
            return items[0].get("task_type", "classification")
    return "classification"


def infer_num_classes(transfer: dict) -> int:
    num_classes = transfer.get("num_classes")
    if isinstance(num_classes, (int, np.integer)) and int(num_classes) > 0:
        return int(num_classes)
    class_names = transfer.get("class_names")
    if isinstance(class_names, (list, tuple)) and len(class_names) > 0:
        return len(class_names)
    max_label = -1
    for rec in transfer.get("per_stock", []):
        for key in ("y_true", "y_pred_before", "y_pred_after", "y_pred"):
            if key not in rec:
                continue
            values = np.asarray(rec[key])
            if values.size == 0 or not np.issubdtype(values.dtype, np.integer):
                continue
            max_label = max(max_label, int(values.max()))
    return max(max_label + 1, 3)


def infer_class_names(transfer: dict) -> list[str]:
    class_names = transfer.get("class_names")
    if isinstance(class_names, (list, tuple)) and len(class_names) > 0:
        return [str(name) for name in class_names]
    return class_names_for_num_classes(infer_num_classes(transfer))


def one_sample_tstat(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    std = values.std(ddof=1)
    if np.isclose(std, 0.0):
        return 0.0
    return float(values.mean() / (std / np.sqrt(len(values))))


def parse_args():
    p = argparse.ArgumentParser(description="Analyze Optiver transfer-learning results")
    p.add_argument("--lookback", type=int, default=50, help="Lookback window T used in training")
    p.add_argument("--horizons", nargs="+", type=int, default=[5],
                   help="Event horizons to summarize (defaults to the retained k=5 Optiver slice)")
    p.add_argument("--num-lime-samples", type=int, default=400,
                   help="Perturbation samples for the LIME-style surrogate")
    p.add_argument("--lime-time-bins", type=int, default=10, help="Temporal segments for LIME")
    p.add_argument("--lime-feature-bins", type=int, default=4, help="Feature-group segments for LIME")
    return p.parse_args()


def load_transfer_results(horizon_k: int) -> dict:
    path = os.path.join(RESULT_DIR, f"transfer_metrics_k{horizon_k}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing transfer metrics: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def stock_npz(stock_id: int):
    path = os.path.join(DATA_DIR, f"stock_{stock_id}_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing processed stock file: {path}")
    return np.load(path)


def load_model_from_state(state_path: str):
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Missing model state: {state_path}")
    state = torch.load(state_path, map_location="cpu")
    regression_head = any(key.startswith("reg_head.") for key in state.keys())
    has_cls_head = any(key.startswith("fc1.") for key in state.keys())
    task_type = "regression" if regression_head and not has_cls_head else "classification"
    y_len = int(state["fc1.weight"].shape[0]) if has_cls_head and "fc1.weight" in state else 3
    model = DeepLOBLite(y_len=y_len, regression_head=regression_head, task_type=task_type)
    model.load_state_dict(state)
    model.eval()
    return model


def build_session_frame(stock_id: int, horizon_k: int, sample_end: np.ndarray,
                        y_true: np.ndarray, y_pred: np.ndarray, signal_values: np.ndarray) -> pd.DataFrame:
    npz = stock_npz(stock_id)
    mid = npz["mid"].astype(np.float64)
    time_id = npz["time_id"].astype(np.int64)

    t0 = sample_end - 1
    t1 = t0 + horizon_k
    realized_return = (mid[t1] - mid[t0]) / (mid[t0] + 1e-10)
    signal = np.asarray(signal_values, dtype=np.float64)[y_pred.astype(np.int64)]

    frame = pd.DataFrame({
        "time_id": time_id[t0],
        "correct": (y_true == y_pred).astype(np.float64),
        "profit_raw": signal * realized_return,
        "abs_return": np.abs(realized_return),
    })
    grouped = frame.groupby("time_id", sort=True).agg(
        accuracy=("correct", "mean"),
        profit_raw=("profit_raw", "sum"),
        abs_return=("abs_return", "sum"),
        n=("correct", "size"),
    ).reset_index()
    grouped["profit_norm"] = grouped["profit_raw"] / (grouped["abs_return"] + 1e-8)
    grouped["stock_id"] = stock_id
    grouped["horizon_k"] = horizon_k
    return grouped


def build_session_frame_regression(stock_id: int, horizon_k: int, time_id: np.ndarray,
                                   y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    time_id = np.asarray(time_id, dtype=np.int64)
    signal = np.sign(y_pred)

    frame = pd.DataFrame({
        "time_id": time_id,
        "correct": (np.sign(y_true) == signal).astype(np.float64),
        "profit_raw": signal * y_true,
        "abs_return": np.abs(y_true),
        "sq_error": (y_true - y_pred) ** 2,
    })
    grouped = frame.groupby("time_id", sort=True).agg(
        accuracy=("correct", "mean"),
        profit_raw=("profit_raw", "sum"),
        abs_return=("abs_return", "sum"),
        mse=("sq_error", "mean"),
        n=("correct", "size"),
    ).reset_index()
    grouped["profit_norm"] = grouped["profit_raw"] / (grouped["abs_return"] + 1e-8)
    grouped["rmse"] = np.sqrt(grouped["mse"])
    grouped["stock_id"] = stock_id
    grouped["horizon_k"] = horizon_k
    return grouped


def collect_frames(transfer: dict, horizon_k: int):
    task_type = infer_task_type(transfer)
    class_names = infer_class_names(transfer)
    num_classes = len(class_names)
    signal_values = signal_values_for_num_classes(num_classes) if task_type != "regression" else None
    rows = []
    for rec in transfer["per_stock"]:
        stock_id = rec["stock_id"]
        if task_type == "regression":
            before_df = build_session_frame_regression(
                stock_id,
                horizon_k,
                np.asarray(rec["time_id"]),
                np.asarray(rec["y_true"]),
                np.asarray(rec["y_pred_before"]),
            )
            after_df = build_session_frame_regression(
                stock_id,
                horizon_k,
                np.asarray(rec["time_id"]),
                np.asarray(rec["y_true"]),
                np.asarray(rec["y_pred_after"]),
            )
        else:
            before_df = build_session_frame(
                stock_id, horizon_k, np.asarray(rec["sample_end"]),
                np.asarray(rec["y_true"]), np.asarray(rec["y_pred_before"]), signal_values
            )
            after_df = build_session_frame(
                stock_id, horizon_k, np.asarray(rec["sample_end"]),
                np.asarray(rec["y_true"]), np.asarray(rec["y_pred_after"]), signal_values
            )
        rows.append({
            "stock_id": stock_id,
            "task_type": task_type,
            "num_classes": num_classes,
            "class_names": class_names,
            "before_df": before_df,
            "after_df": after_df,
            "before_metrics": rec["before"],
            "after_metrics": rec["after"],
            "base_collapse": rec.get("base_collapse", {}),
            "after_collapse": rec.get("after_collapse", {}),
            "requested_transfer_mode": rec.get("requested_transfer_mode", rec.get("transfer_mode")),
            "selected_transfer_mode": rec.get("selected_transfer_mode", rec.get("transfer_mode")),
            "candidate_modes": rec.get("candidate_modes", []),
            "train_losses": rec.get("train_losses", []),
            "val_losses": rec.get("val_losses", []),
            "train_cls_losses": rec.get("train_cls_losses", []),
            "val_cls_losses": rec.get("val_cls_losses", []),
            "train_reg_losses": rec.get("train_reg_losses", []),
            "val_reg_losses": rec.get("val_reg_losses", []),
            "val_reg_corr": np.asarray(rec.get("val_reg_corr", []), dtype=np.float64),
            "val_macro_f1": np.asarray(rec.get("val_macro_f1", []), dtype=np.float64),
            "val_kappa": np.asarray(rec.get("val_kappa", []), dtype=np.float64),
            "monitor": rec.get("monitor", "val_macro_f1"),
            "y_true": np.asarray(rec["y_true"]),
            "y_pred_before": np.asarray(rec["y_pred_before"]),
            "y_pred_after": np.asarray(rec["y_pred_after"]),
            "y_probs_before": np.asarray(rec["y_probs_before"]),
            "y_probs_after": np.asarray(rec["y_probs_after"]),
            "sample_end": np.asarray(rec["sample_end"]),
        })
    return rows


def choose_transfer_loss_case(items: list[dict]):
    best = None
    best_key = None
    for item in items:
        train_losses = item.get("train_losses", [])
        val_losses = item.get("val_losses", [])
        if len(train_losses) == 0 or len(val_losses) == 0:
            continue
        if item.get("task_type") == "regression":
            delta = item["after_metrics"]["corr"] - item["before_metrics"]["corr"]
            key = (
                delta,
                item["after_metrics"]["corr"],
                item["after_metrics"]["sign_accuracy"],
                -int(item["stock_id"]),
            )
        else:
            delta = item["after_metrics"]["accuracy"] - item["before_metrics"]["accuracy"]
            key = (
                delta,
                item["after_metrics"]["accuracy"],
                -float(item.get("after_collapse", {}).get("dominant_share", 1.0)),
                -int(item["stock_id"]),
            )
        if best is None or key > best_key:
            best = item
            best_key = key
    return best


def plot_transfer_loss_examples(all_frames: dict[int, list[dict]]):
    task_type = analysis_task_type(all_frames)
    rows = []
    paths = {}
    for horizon_k, items in all_frames.items():
        item = choose_transfer_loss_case(items)
        if item is None:
            continue
        train_losses = np.asarray(item.get("train_losses", []), dtype=np.float64)
        val_losses = np.asarray(item.get("val_losses", []), dtype=np.float64)
        train_cls = np.asarray(item.get("train_cls_losses", []), dtype=np.float64)
        val_cls = np.asarray(item.get("val_cls_losses", []), dtype=np.float64)
        train_reg = np.asarray(item.get("train_reg_losses", []), dtype=np.float64)
        val_reg = np.asarray(item.get("val_reg_losses", []), dtype=np.float64)
        monitor, monitor_values, monitor_name = monitor_series_from_item(item)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train total", lw=2)
        ax.plot(epochs, val_losses, label="Val total", lw=2)
        if len(train_cls) == len(epochs) and len(val_cls) == len(epochs):
            ax.plot(epochs, train_cls, label="Train cls", lw=1.5, linestyle="--", alpha=0.8)
            ax.plot(epochs, val_cls, label="Val cls", lw=1.5, linestyle="--", alpha=0.8)
        if len(train_reg) == len(epochs) and len(val_reg) == len(epochs) and (np.any(train_reg) or np.any(val_reg)):
            ax.plot(epochs, train_reg, label="Train reg", lw=1.2, linestyle=":", alpha=0.9)
            ax.plot(epochs, val_reg, label="Val reg", lw=1.2, linestyle=":", alpha=0.9)

        stock_id = item["stock_id"]
        selected_mode = item.get("selected_transfer_mode")
        if task_type == "regression":
            delta = item["after_metrics"]["corr"] - item["before_metrics"]["corr"]
            ax.set_title(
                f"Representative Transfer Curves (k={horizon_k}, stock {stock_id}, mode={selected_mode}, monitor={monitor}, Δcorr={delta:+.3f})"
            )
        else:
            delta = item["after_metrics"]["accuracy"] - item["before_metrics"]["accuracy"]
            ax.set_title(
                f"Representative Transfer Curves (k={horizon_k}, stock {stock_id}, mode={selected_mode}, monitor={monitor}, Δacc={delta:+.3f})"
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        if len(monitor_values) == len(epochs):
            best_epoch = best_item_monitor_epoch(item)
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
        out = os.path.join(RESULT_DIR, f"transfer_loss_selected_k{horizon_k}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)

        paths[f"transfer_loss_k{horizon_k}"] = out
        row = {
            "horizon_k": horizon_k,
            "stock_id": stock_id,
            "task_type": task_type,
            "selected_transfer_mode": selected_mode,
            "path": out,
        }
        if task_type == "regression":
            row.update({
                "corr_before": item["before_metrics"]["corr"],
                "corr_after": item["after_metrics"]["corr"],
                "corr_delta": delta,
            })
        else:
            row.update({
                "accuracy_before": item["before_metrics"]["accuracy"],
                "accuracy_after": item["after_metrics"]["accuracy"],
                "accuracy_delta": delta,
            })
        rows.append(row)
    return paths, rows


def grouped_boxplots(ax, data_by_horizon: dict[int, list[tuple[int, np.ndarray]]], ylabel: str, title: str):
    horizons = list(data_by_horizon.keys())
    stocks = [sid for sid, _ in data_by_horizon[horizons[0]]]
    x = np.arange(len(stocks), dtype=float)
    offsets = np.linspace(-0.25, 0.25, num=len(horizons))
    colors = plt.cm.Set2(np.linspace(0, 1, len(horizons)))

    for color, offset, horizon_k in zip(colors, offsets, horizons):
        series = [vals for _, vals in data_by_horizon[horizon_k]]
        pos = x + offset
        bp = ax.boxplot(
            series,
            positions=pos,
            widths=0.18,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        for median in bp["medians"]:
            median.set_color("black")
        ax.plot([], [], color=color, lw=8, label=f"k={horizon_k}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"s{sid}" for sid in stocks], rotation=25)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2, axis="y")


def plot_figure6(all_frames: dict[int, list[dict]]):
    task_type = analysis_task_type(all_frames)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    data_before = {}
    data_after = {}
    for horizon_k, items in all_frames.items():
        data_before[horizon_k] = [(item["stock_id"], item["before_df"]["accuracy"].to_numpy()) for item in items]
        data_after[horizon_k] = [(item["stock_id"], item["after_df"]["accuracy"].to_numpy()) for item in items]

    grouped_boxplots(
        axes[0],
        data_before,
        ylabel="Per-session directional accuracy" if task_type == "regression" else "Per-session accuracy",
        title=(
            "Figure 6 analogue — Zero-shot directional accuracy on held-out transfer stocks"
            if task_type == "regression"
            else "Figure 6 analogue — Zero-shot accuracy on held-out transfer stocks"
        ),
    )
    grouped_boxplots(
        axes[1],
        data_after,
        ylabel="Per-session directional accuracy" if task_type == "regression" else "Per-session accuracy",
        title=(
            "Figure 6 analogue — Directional accuracy after transfer learning"
            if task_type == "regression"
            else "Figure 6 analogue — Accuracy after transfer learning"
        ),
    )
    axes[1].set_xlabel("Held-out transfer stocks")
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "figure6_transfer_accuracy.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_figure7(all_frames: dict[int, list[dict]]):
    task_type = analysis_task_type(all_frames)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex="col")
    stocks = [item["stock_id"] for item in next(iter(all_frames.values()))]
    horizons = list(all_frames.keys())
    x = np.arange(len(stocks), dtype=float)
    offsets = np.linspace(-0.25, 0.25, num=len(horizons))
    colors = plt.cm.Set2(np.linspace(0, 1, len(horizons)))

    def profit_data(which: str):
        out = {}
        for horizon_k, items in all_frames.items():
            out[horizon_k] = [
                (item["stock_id"], item[f"{which}_df"]["profit_norm"].to_numpy())
                for item in items
            ]
        return out

    grouped_boxplots(
        axes[0, 0],
        profit_data("before"),
        ylabel="Normalized session profit",
        title=(
            "Figure 7 analogue — Zero-shot normalized profits from sign forecasts"
            if task_type == "regression"
            else "Figure 7 analogue — Zero-shot normalized profits"
        ),
    )
    grouped_boxplots(
        axes[1, 0],
        profit_data("after"),
        ylabel="Normalized session profit",
        title=(
            "Figure 7 analogue — Transfer-learning normalized profits from sign forecasts"
            if task_type == "regression"
            else "Figure 7 analogue — Transfer-learning normalized profits"
        ),
    )

    for row, which in enumerate(["before", "after"]):
        ax = axes[row, 1]
        for color, offset, horizon_k in zip(colors, offsets, horizons):
            tstats = []
            for item in all_frames[horizon_k]:
                profits = item[f"{which}_df"]["profit_norm"].to_numpy()
                tstats.append(one_sample_tstat(profits))
            ax.bar(x + offset, tstats, width=0.18, color=color, alpha=0.85, label=f"k={horizon_k}")
        ax.axhline(1.645, color="gray", linestyle="--", lw=1, label="10% one-sided")
        ax.axhline(1.96, color="black", linestyle=":", lw=1, label="5% one-sided")
        ax.set_title(
            (
                "Zero-shot profit t-statistics from sign forecasts"
                if which == "before" else "Transfer-learning profit t-statistics from sign forecasts"
            ) if task_type == "regression" else (
                "Zero-shot profit t-statistics" if which == "before"
                else "Transfer-learning profit t-statistics"
            )
        )
        ax.set_ylabel("t-statistic")
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{sid}" for sid in stocks], rotation=25)
        ax.grid(alpha=0.2, axis="y")
        ax.legend(loc="best", fontsize=8)

    axes[1, 0].set_xlabel("Held-out transfer stocks")
    axes[1, 1].set_xlabel("Held-out transfer stocks")
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "figure7_transfer_profit_tstats.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_figure8(all_frames: dict[int, list[dict]]):
    task_type = analysis_task_type(all_frames)
    stocks = [item["stock_id"] for item in next(iter(all_frames.values()))]
    fig, axes = plt.subplots(2, len(stocks), figsize=(4 * len(stocks), 8), sharey="row")
    if len(stocks) == 1:
        axes = np.array(axes).reshape(2, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_frames)))

    for col, stock_id in enumerate(stocks):
        for row, which in enumerate(["before", "after"]):
            ax = axes[row, col]
            for color, horizon_k in zip(colors, all_frames.keys()):
                item = next(it for it in all_frames[horizon_k] if it["stock_id"] == stock_id)
                session_df = item[f"{which}_df"].sort_values("time_id")
                cum_profit = session_df["profit_norm"].cumsum().to_numpy()
                ax.plot(session_df["time_id"], cum_profit, lw=2, color=color, label=f"k={horizon_k}")
            ax.set_title(
                f"s{stock_id} — zero-shot" if row == 0 else f"s{stock_id} — transfer",
                fontsize=10,
            )
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("Cumulative normalized profit")
            if row == 1:
                ax.set_xlabel("time_id")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(
        "Figure 8 analogue — Cumulative normalized profits from sign forecasts"
        if task_type == "regression"
        else "Figure 8 analogue — Cumulative normalized profits on transfer stocks",
        y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "figure8_transfer_cum_profit.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def choose_lime_case(all_frames: dict[int, list[dict]]):
    task_type = analysis_task_type(all_frames)
    best = None
    for horizon_k, items in all_frames.items():
        for item in items:
            if task_type == "regression":
                delta = item["after_metrics"]["corr"] - item["before_metrics"]["corr"]
                candidate = (delta, item["after_metrics"]["corr"], horizon_k, item["stock_id"])
            else:
                delta = item["after_metrics"]["accuracy"] - item["before_metrics"]["accuracy"]
                candidate = (delta, horizon_k, item["stock_id"])
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("No transfer-learning results found for LIME case selection")

    if task_type == "regression":
        _, _, horizon_k, stock_id = best
    else:
        _, horizon_k, stock_id = best
    item = next(it for it in all_frames[horizon_k] if it["stock_id"] == stock_id)
    if task_type == "regression":
        err_before = np.abs(item["y_true"] - item["y_pred_before"])
        err_after = np.abs(item["y_true"] - item["y_pred_after"])
        improvement = err_before - err_after
        idx = int(np.argmax(improvement))
    else:
        margin = item["y_probs_after"].max(axis=1) - item["y_probs_before"].max(axis=1)
        candidates = np.where(item["y_pred_after"] == item["y_true"])[0]
        idx = int(candidates[np.argmax(margin[candidates])]) if len(candidates) else int(np.argmax(margin))
    return horizon_k, stock_id, item, idx


def load_window(stock_id: int, sample_end: int, lookback: int) -> np.ndarray:
    npz = stock_npz(stock_id)
    X = npz["X"].astype(np.float32)
    return X[sample_end - lookback: sample_end].copy()


def lime_heatmap(model, x0: np.ndarray, target_class: int, num_samples: int,
                 time_bins: int, feature_bins: int, task_type: str = "classification") -> np.ndarray:
    model.eval()
    T, F = x0.shape
    t_groups = np.array_split(np.arange(T), time_bins)
    f_groups = np.array_split(np.arange(F), feature_bins)
    segments = [(tg, fg) for tg in t_groups for fg in f_groups]
    n_segments = len(segments)
    masks = np.random.binomial(1, 0.75, size=(num_samples, n_segments)).astype(np.float32)
    masks[0, :] = 1.0

    inputs = np.repeat(x0[None, :, :], num_samples, axis=0)
    for i, mask in enumerate(masks):
        for seg_idx, keep in enumerate(mask):
            if keep == 1.0:
                continue
            tg, fg = segments[seg_idx]
            inputs[i][np.ix_(tg, fg)] = 0.0

    responses = []
    with torch.no_grad():
        batch = torch.from_numpy(inputs[:, None, :, :]).float()
        for start in range(0, len(batch), 64):
            outputs = model(batch[start:start + 64])
            if getattr(model, "task_type", task_type) == "regression":
                preds = outputs[1] if isinstance(outputs, tuple) else outputs
                responses.append(preds.cpu().numpy())
            else:
                logits, _ = unpack_model_output(outputs)
                responses.append(torch.softmax(logits, dim=1)[:, target_class].cpu().numpy())
    responses = np.concatenate(responses)

    distances = np.sqrt(((1.0 - masks) ** 2).sum(axis=1))
    kernel_width = 0.75 * np.sqrt(n_segments)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2 + 1e-8))

    surrogate = Ridge(alpha=1.0)
    surrogate.fit(masks, responses, sample_weight=weights)
    coefs = surrogate.coef_

    heatmap = np.zeros((T, F), dtype=np.float32)
    for coef, (tg, fg) in zip(coefs, segments):
        heatmap[np.ix_(tg, fg)] = coef
    return heatmap


def plot_figure9(all_frames: dict[int, list[dict]], lookback: int, args):
    task_type = analysis_task_type(all_frames)
    horizon_k, stock_id, item, idx = choose_lime_case(all_frames)
    sample_end = int(item["sample_end"][idx])
    x0 = load_window(stock_id, sample_end, lookback)
    target_class = int(item["y_true"][idx]) if task_type != "regression" else 0
    class_names = item.get("class_names", CLASS_LABELS)

    base_model = load_model_from_state(
        os.path.join(MODEL_DIR, f"optiver_base_k{horizon_k}_state.pt"),
    )
    ft_model = load_model_from_state(
        os.path.join(MODEL_DIR, f"optiver_transfer_s{stock_id}_k{horizon_k}.pt"),
    )

    heat_before = lime_heatmap(
        base_model, x0, target_class, args.num_lime_samples,
        args.lime_time_bins, args.lime_feature_bins,
        task_type=task_type,
    )
    heat_after = lime_heatmap(
        ft_model, x0, target_class, args.num_lime_samples,
        args.lime_time_bins, args.lime_feature_bins,
        task_type=task_type,
    )

    vmax = max(np.abs(heat_before).max(), np.abs(heat_after).max(), 1e-6)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].imshow(x0.T, aspect="auto", origin="lower", cmap="viridis")
    if task_type == "regression":
        target_value = float(item["y_true"][idx])
        axes[0].set_title(
            f"Figure 9 analogue — Input window (stock {stock_id}, k={horizon_k}, true log return={target_value:+.5f})"
        )
    else:
        axes[0].set_title(
            f"Figure 9 analogue — Input window (stock {stock_id}, k={horizon_k}, class={class_names[target_class] if target_class < len(class_names) else target_class})"
        )
    axes[1].imshow(heat_before.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Base model local importance (pre-transfer)")
    axes[2].imshow(heat_after.T, aspect="auto", origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Fine-tuned model local importance (post-transfer)")
    for ax in axes:
        ax.set_yticks(np.arange(len(FEATURE_LABELS)))
        ax.set_yticklabels(FEATURE_LABELS)
        ax.set_ylabel("LOB features")
    axes[2].set_xlabel("Lookback event index")
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "figure9_transfer_lime.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)

    return out, {
        "task_type": task_type,
        "stock_id": stock_id,
        "horizon_k": horizon_k,
        "sample_index": idx,
        **({"true_class": class_names[target_class] if target_class < len(class_names) else str(target_class)} if task_type != "regression" else {"true_log_return": float(item["y_true"][idx])}),
    }


def build_summary(all_frames: dict[int, list[dict]], figure_paths: dict, lime_case: dict,
                  representative_transfer_losses: list[dict]):
    task_type = analysis_task_type(all_frames)
    rows = []
    strategy_rows = []
    for horizon_k, items in all_frames.items():
        for item in items:
            before_profit = float(item["before_df"]["profit_norm"].mean())
            after_profit = float(item["after_df"]["profit_norm"].mean())
            row = {
                "stock_id": item["stock_id"],
                "horizon_k": horizon_k,
                "task_type": task_type,
                "profit_before_mean": before_profit,
                "profit_after_mean": after_profit,
                "profit_delta": after_profit - before_profit,
                "requested_transfer_mode": item.get("requested_transfer_mode"),
                "selected_transfer_mode": item.get("selected_transfer_mode"),
            }
            if task_type == "regression":
                row.update({
                    "corr_before": item["before_metrics"]["corr"],
                    "corr_after": item["after_metrics"]["corr"],
                    "corr_delta": item["after_metrics"]["corr"] - item["before_metrics"]["corr"],
                    "rmse_before": item["before_metrics"]["rmse"],
                    "rmse_after": item["after_metrics"]["rmse"],
                    "sign_accuracy_before": item["before_metrics"]["sign_accuracy"],
                    "sign_accuracy_after": item["after_metrics"]["sign_accuracy"],
                })
            else:
                base_collapse = item.get("base_collapse", {})
                after_collapse = item.get("after_collapse", {})
                row.update({
                    "accuracy_before": item["before_metrics"]["accuracy"],
                    "accuracy_after": item["after_metrics"]["accuracy"],
                    "accuracy_delta": item["after_metrics"]["accuracy"] - item["before_metrics"]["accuracy"],
                    "base_single_class_collapse": base_collapse.get("single_class_collapse"),
                    "base_dominant_label": base_collapse.get("dominant_label"),
                    "base_dominant_share": base_collapse.get("dominant_share"),
                    "transfer_single_class_collapse": after_collapse.get("single_class_collapse"),
                    "transfer_dominant_label": after_collapse.get("dominant_label"),
                    "transfer_dominant_share": after_collapse.get("dominant_share"),
                })
            rows.append(row)
            strategy_rows.append({
                "stock_id": item["stock_id"],
                "horizon_k": horizon_k,
                "task_type": task_type,
                "requested_transfer_mode": item.get("requested_transfer_mode"),
                "selected_transfer_mode": item.get("selected_transfer_mode"),
                "candidate_modes": item.get("candidate_modes", []),
            })

    summary = {
        "task_type": task_type,
        "figures": figure_paths,
        "lime_case": lime_case,
        "per_stock_horizon": rows,
        "strategy_diagnostics": strategy_rows,
        "representative_transfer_losses": representative_transfer_losses,
    }
    summary_json = json_safe(summary)
    with open(os.path.join(RESULT_DIR, "transfer_analysis_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)
    return summary_json


def main():
    args = parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_transfers = {h: load_transfer_results(h) for h in args.horizons}
    task_types = {infer_task_type(transfer) for transfer in all_transfers.values()}
    if len(task_types) > 1:
        raise RuntimeError(f"Mixed Optiver task types are not supported in one analysis run: {sorted(task_types)}")
    task_type = next(iter(task_types)) if task_types else "classification"
    all_frames = {h: collect_frames(all_transfers[h], h) for h in args.horizons}
    figure_paths = {
        "figure6": plot_figure6(all_frames),
        "figure7": plot_figure7(all_frames),
        "figure8": plot_figure8(all_frames),
    }
    transfer_loss_paths, representative_transfer_losses = plot_transfer_loss_examples(all_frames)
    figure_paths.update(transfer_loss_paths)
    figure9_path, lime_case = plot_figure9(all_frames, args.lookback, args)
    figure_paths["figure9"] = figure9_path
    summary = build_summary(all_frames, figure_paths, lime_case, representative_transfer_losses)

    print("Saved Optiver analysis artifacts:")
    for name, path in figure_paths.items():
        print(f"  {name}: {path}")
    if task_type == "regression":
        print(
            f"LIME case: stock {lime_case['stock_id']} | k={lime_case['horizon_k']} | "
            f"true log return={lime_case['true_log_return']:+.5f}"
        )
    else:
        print(f"LIME case: stock {lime_case['stock_id']} | k={lime_case['horizon_k']} | class={lime_case['true_class']}")
    print(f"Summary JSON: {os.path.join(RESULT_DIR, 'transfer_analysis_summary.json')}")
    print(f"Rows summarized: {len(summary['per_stock_horizon'])}")


if __name__ == "__main__":
    main()
