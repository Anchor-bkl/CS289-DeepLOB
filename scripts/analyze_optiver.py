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

from train_optiver import DATA_DIR, MODEL_DIR, RESULT_DIR, DeepLOBLite


FEATURE_LABELS = [
    "ask_p1", "ask_v1", "bid_p1", "bid_v1",
    "ask_p2", "ask_v2", "bid_p2", "bid_v2",
]
CLASS_LABELS = ["Down", "Stationary", "Up"]


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
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10],
                   help="Event horizons to summarize")
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
    model = DeepLOBLite(y_len=3)
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()
    return model


def build_session_frame(stock_id: int, horizon_k: int, sample_end: np.ndarray,
                        y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    npz = stock_npz(stock_id)
    mid = npz["mid"].astype(np.float64)
    time_id = npz["time_id"].astype(np.int64)

    t0 = sample_end - 1
    t1 = t0 + horizon_k
    realized_return = (mid[t1] - mid[t0]) / (mid[t0] + 1e-10)
    signal = y_pred.astype(np.int64) - 1

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


def collect_frames(horizon_k: int):
    transfer = load_transfer_results(horizon_k)
    rows = []
    for rec in transfer["per_stock"]:
        stock_id = rec["stock_id"]
        before_df = build_session_frame(
            stock_id, horizon_k, np.asarray(rec["sample_end"]),
            np.asarray(rec["y_true"]), np.asarray(rec["y_pred_before"])
        )
        after_df = build_session_frame(
            stock_id, horizon_k, np.asarray(rec["sample_end"]),
            np.asarray(rec["y_true"]), np.asarray(rec["y_pred_after"])
        )
        rows.append({
            "stock_id": stock_id,
            "before_df": before_df,
            "after_df": after_df,
            "before_metrics": rec["before"],
            "after_metrics": rec["after"],
            "y_true": np.asarray(rec["y_true"]),
            "y_pred_before": np.asarray(rec["y_pred_before"]),
            "y_pred_after": np.asarray(rec["y_pred_after"]),
            "y_probs_before": np.asarray(rec["y_probs_before"]),
            "y_probs_after": np.asarray(rec["y_probs_after"]),
            "sample_end": np.asarray(rec["sample_end"]),
        })
    return rows


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
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    data_before = {}
    data_after = {}
    for horizon_k, items in all_frames.items():
        data_before[horizon_k] = [(item["stock_id"], item["before_df"]["accuracy"].to_numpy()) for item in items]
        data_after[horizon_k] = [(item["stock_id"], item["after_df"]["accuracy"].to_numpy()) for item in items]

    grouped_boxplots(
        axes[0],
        data_before,
        ylabel="Per-session accuracy",
        title="Figure 6 analogue — Zero-shot accuracy on held-out transfer stocks",
    )
    grouped_boxplots(
        axes[1],
        data_after,
        ylabel="Per-session accuracy",
        title="Figure 6 analogue — Accuracy after transfer learning",
    )
    axes[1].set_xlabel("Held-out transfer stocks")
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "figure6_transfer_accuracy.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_figure7(all_frames: dict[int, list[dict]]):
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
        title="Figure 7 analogue — Zero-shot normalized profits",
    )
    grouped_boxplots(
        axes[1, 0],
        profit_data("after"),
        ylabel="Normalized session profit",
        title="Figure 7 analogue — Transfer-learning normalized profits",
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
            "Zero-shot profit t-statistics" if which == "before"
            else "Transfer-learning profit t-statistics"
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
    fig.suptitle("Figure 8 analogue — Cumulative normalized profits on transfer stocks", y=1.02)
    fig.tight_layout()
    out = os.path.join(RESULT_DIR, "figure8_transfer_cum_profit.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def choose_lime_case(all_frames: dict[int, list[dict]]):
    best = None
    for horizon_k, items in all_frames.items():
        for item in items:
            delta = item["after_metrics"]["accuracy"] - item["before_metrics"]["accuracy"]
            candidate = (delta, horizon_k, item["stock_id"])
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise RuntimeError("No transfer-learning results found for LIME case selection")

    _, horizon_k, stock_id = best
    item = next(it for it in all_frames[horizon_k] if it["stock_id"] == stock_id)
    margin = item["y_probs_after"].max(axis=1) - item["y_probs_before"].max(axis=1)
    candidates = np.where(item["y_pred_after"] == item["y_true"])[0]
    idx = int(candidates[np.argmax(margin[candidates])]) if len(candidates) else int(np.argmax(margin))
    return horizon_k, stock_id, item, idx


def load_window(stock_id: int, sample_end: int, lookback: int) -> np.ndarray:
    npz = stock_npz(stock_id)
    X = npz["X"].astype(np.float32)
    return X[sample_end - lookback: sample_end].copy()


def lime_heatmap(model, x0: np.ndarray, target_class: int, num_samples: int,
                 time_bins: int, feature_bins: int) -> np.ndarray:
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

    probs = []
    with torch.no_grad():
        batch = torch.from_numpy(inputs[:, None, :, :]).float()
        for start in range(0, len(batch), 64):
            logits = model(batch[start:start + 64])
            probs.append(torch.softmax(logits, dim=1)[:, target_class].cpu().numpy())
    probs = np.concatenate(probs)

    distances = np.sqrt(((1.0 - masks) ** 2).sum(axis=1))
    kernel_width = 0.75 * np.sqrt(n_segments)
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2 + 1e-8))

    surrogate = Ridge(alpha=1.0)
    surrogate.fit(masks, probs, sample_weight=weights)
    coefs = surrogate.coef_

    heatmap = np.zeros((T, F), dtype=np.float32)
    for coef, (tg, fg) in zip(coefs, segments):
        heatmap[np.ix_(tg, fg)] = coef
    return heatmap


def plot_figure9(all_frames: dict[int, list[dict]], lookback: int, args):
    horizon_k, stock_id, item, idx = choose_lime_case(all_frames)
    sample_end = int(item["sample_end"][idx])
    x0 = load_window(stock_id, sample_end, lookback)
    target_class = int(item["y_true"][idx])

    base_model = load_model_from_state(
        os.path.join(MODEL_DIR, f"optiver_base_k{horizon_k}_state.pt"),
    )
    ft_model = load_model_from_state(
        os.path.join(MODEL_DIR, f"optiver_transfer_s{stock_id}_k{horizon_k}.pt"),
    )

    heat_before = lime_heatmap(
        base_model, x0, target_class, args.num_lime_samples,
        args.lime_time_bins, args.lime_feature_bins,
    )
    heat_after = lime_heatmap(
        ft_model, x0, target_class, args.num_lime_samples,
        args.lime_time_bins, args.lime_feature_bins,
    )

    vmax = max(np.abs(heat_before).max(), np.abs(heat_after).max(), 1e-6)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].imshow(x0.T, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title(
        f"Figure 9 analogue — Input window (stock {stock_id}, k={horizon_k}, class={CLASS_LABELS[target_class]})"
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
        "stock_id": stock_id,
        "horizon_k": horizon_k,
        "sample_index": idx,
        "true_class": CLASS_LABELS[target_class],
    }


def build_summary(all_frames: dict[int, list[dict]], figure_paths: dict, lime_case: dict):
    rows = []
    for horizon_k, items in all_frames.items():
        for item in items:
            before_acc = item["before_metrics"]["accuracy"]
            after_acc = item["after_metrics"]["accuracy"]
            before_profit = float(item["before_df"]["profit_norm"].mean())
            after_profit = float(item["after_df"]["profit_norm"].mean())
            rows.append({
                "stock_id": item["stock_id"],
                "horizon_k": horizon_k,
                "accuracy_before": before_acc,
                "accuracy_after": after_acc,
                "accuracy_delta": after_acc - before_acc,
                "profit_before_mean": before_profit,
                "profit_after_mean": after_profit,
                "profit_delta": after_profit - before_profit,
            })

    summary = {
        "figures": figure_paths,
        "lime_case": lime_case,
        "per_stock_horizon": rows,
    }
    with open(os.path.join(RESULT_DIR, "transfer_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)

    all_frames = {h: collect_frames(h) for h in args.horizons}
    figure_paths = {
        "figure6": plot_figure6(all_frames),
        "figure7": plot_figure7(all_frames),
        "figure8": plot_figure8(all_frames),
    }
    figure9_path, lime_case = plot_figure9(all_frames, args.lookback, args)
    figure_paths["figure9"] = figure9_path
    summary = build_summary(all_frames, figure_paths, lime_case)

    print("Saved Optiver analysis artifacts:")
    for name, path in figure_paths.items():
        print(f"  {name}: {path}")
    print(f"LIME case: stock {lime_case['stock_id']} | k={lime_case['horizon_k']} | class={lime_case['true_class']}")
    print(f"Summary JSON: {os.path.join(RESULT_DIR, 'transfer_analysis_summary.json')}")
    print(f"Rows summarized: {len(summary['per_stock_horizon'])}")


if __name__ == "__main__":
    main()
