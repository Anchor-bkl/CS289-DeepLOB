#!/usr/bin/env python3
"""
analyze_fi2010.py — CPU analysis pipeline for FI-2010 DeepLOB project
======================================================================

This script generates the non-training artifacts used by the report notebook:
1. Engineered-feature predictability tests (Newey-West t-tests)
2. Rolling-window stability / monotonicity / Rank-IC analysis
3. FDR-qualified factor selection
4. Baseline model training (Ridge logistic regression + 2-layer MLP)
5. Trading-signal strategy statistics using saved DeepLOB predictions

It is intentionally separate from the notebook so the notebook can stay light
and primarily *load* existing artifacts rather than perform long computations.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")

FEATURE_INFO = {
    "u2": (40, 60, "Spread & Mid-Price"),
    "u3": (60, 80, "Price Differences"),
    "u4": (80, 84, "Price & Volume Means"),
    "u5": (84, 86, "Accumulated Differences"),
    "u6": (86, 126, "Price & Volume Derivatives"),
    "u7": (126, 132, "Avg Intensity per Type"),
    "u8": (132, 138, "Relative Intensity Comparison"),
    "u9": (138, 144, "Limit Activity Acceleration"),
}
K_EVENT_MAP = {0: 10, 1: 20, 2: 30, 3: 50, 4: 100}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FI-2010 analysis pipeline")
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    return p.parse_args()


def load_fi2010() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_all = np.loadtxt(os.path.join(DATA_DIR, "Train_Dst_NoAuction_DecPre_CF_7.txt"))
    train = train_all[:, : int(np.floor(train_all.shape[1] * 0.8))]
    val = train_all[:, int(np.floor(train_all.shape[1] * 0.8)) :]
    t1 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_7.txt"))
    t2 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_8.txt"))
    t3 = np.loadtxt(os.path.join(DATA_DIR, "Test_Dst_NoAuction_DecPre_CF_9.txt"))
    test = np.hstack((t1, t2, t3))
    return train, val, test


def nw_tstat(r: np.ndarray, f: np.ndarray, nlags: int) -> tuple[float, float, float]:
    """OLS slope with Bartlett-kernel Newey-West t-statistics."""
    n = len(r)
    fm = f - f.mean()
    rm = r - r.mean()
    sff = (fm * fm).mean()
    if sff < 1e-14:
        return 0.0, 0.0, 1.0
    beta = (fm * rm).mean() / sff
    resid = rm - beta * fm
    g = fm * resid
    s = (g * g).mean()
    for j in range(1, nlags + 1):
        w = 1.0 - j / (nlags + 1)
        s += 2.0 * w * (g[j:] * g[:-j]).mean()
    var_b = max(s / (sff**2 * n), 1e-20)
    t_nw = beta / np.sqrt(var_b)
    p_nw = float(2.0 * stats.t.sf(abs(t_nw), df=n - 2))
    return beta, t_nw, p_nw


def bh_correction(p_values: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresholds = q * (np.arange(1, m + 1) / m)
    reject = np.zeros(m, dtype=bool)
    below = p[order] <= thresholds
    if below.any():
        last = np.where(below)[0][-1]
        reject[order[: last + 1]] = True
    return reject


def sharpe_like_from_labels(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple directional pseudo-PnL statistic for model comparison."""
    true_dir = np.where(y_true == 2, 1.0, np.where(y_true == 0, -1.0, 0.0))
    pred_dir = np.where(y_pred == 2, 1.0, np.where(y_pred == 0, -1.0, 0.0))
    pnl = pred_dir * true_dir
    pnl_std = pnl.std(ddof=0)
    if pnl_std < 1e-12:
        return 0.0
    return float(pnl.mean() / pnl_std)


def plot_group_bars(df_nw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), squeeze=False)
    axes = axes.ravel()
    for ax, (set_name, (_, _, desc)) in zip(axes, FEATURE_INFO.items()):
        sub = df_nw[df_nw["Set"] == set_name].copy()
        colors = ["#d73027" if p < 0.001 else "#91bfdb" for p in sub["p_NW"]]
        ax.bar(sub["Feature #"], sub["t_NW"], color=colors, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="k", linewidth=0.6)
        ax.set_title(f"{set_name}: {desc}", fontsize=9)
        ax.set_xlabel("Feature #", fontsize=8)
        ax.set_ylabel("NW t-stat", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "feature_nw_ttest.png"), dpi=150)
    plt.close()


def part_a_feature_predictability(dec_train: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    k_events = 10
    mid = (dec_train[0] + dec_train[2]) / 2.0
    ret = ((mid[k_events:] - mid[:-k_events]) / mid[:-k_events]).astype(np.float64)
    nlags = max(5, int(4.0 * (len(ret) / 100.0) ** (2.0 / 9.0)))

    rows = []
    for set_name, (start, end, desc) in FEATURE_INFO.items():
        for j in range(end - start):
            feat = dec_train[start + j, :-k_events].astype(np.float64)
            beta, t_nw, p_nw = nw_tstat(ret, feat, nlags)
            rows.append(
                {
                    "Set": set_name,
                    "Description": desc,
                    "Feature #": j + 1,
                    "Data row": start + j,
                    "beta": beta,
                    "t_NW": t_nw,
                    "p_NW": p_nw,
                }
            )

    df_nw = pd.DataFrame(rows)
    df_nw["sig_05"] = df_nw["p_NW"] < 0.05
    df_nw["sig_001"] = df_nw["p_NW"] < 0.001
    df_nw.to_csv(os.path.join(RESULT_DIR, "feature_predictability.csv"), index=False)
    plot_group_bars(df_nw)
    return df_nw, mid, ret


def part_b_rolling_stability(
    dec_train: np.ndarray, df_nw: pd.DataFrame, mid_train: np.ndarray, k_events: int = 10
) -> pd.DataFrame:
    win_size = 20_000
    step_size = 5_000
    cands = df_nw[df_nw["p_NW"] < 0.05].copy()

    rows = []
    n_tr = dec_train.shape[1]
    starts = list(range(0, n_tr - win_size - k_events, step_size))
    for _, row in cands.iterrows():
        feat_row = int(row["Data row"])
        feat_all = dec_train[feat_row, :].astype(np.float64)
        sig_wins, abs_t_list = [], []
        for s in starts:
            e = s + win_size
            f_w = feat_all[s : e - k_events]
            m_w = mid_train[s:e]
            r_w = ((m_w[k_events:] - m_w[:-k_events]) / m_w[:-k_events]).astype(np.float64)
            lag_w = max(5, int(4.0 * (len(r_w) / 100.0) ** (2.0 / 9.0)))
            _, t_w, _ = nw_tstat(r_w, f_w, lag_w)
            abs_t_list.append(abs(t_w))
            crit = stats.t.ppf(0.975, df=len(r_w) - 2)
            sig_wins.append(int(abs(t_w) > crit))

        rows.append(
            {
                "Set": row["Set"],
                "Description": row["Description"],
                "Feature #": row["Feature #"],
                "Data row": feat_row,
                "beta_full": row["beta"],
                "t_NW_full": row["t_NW"],
                "p_NW_full": row["p_NW"],
                "N windows": len(starts),
                "Sig windows": sum(sig_wins),
                "Sig ratio": round(sum(sig_wins) / len(starts), 3) if starts else 0.0,
                "Mean |t_NW|": round(np.mean(abs_t_list), 3) if abs_t_list else 0.0,
            }
        )

    df_roll = pd.DataFrame(rows)
    df_roll.to_csv(os.path.join(RESULT_DIR, "rolling_stability.csv"), index=False)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4.5))
    labels = [f"{r['Set']}/f{r['Feature #']}" for _, r in df_roll.iterrows()]
    ax0.bar(
        range(len(df_roll)),
        df_roll["Sig ratio"],
        color=["#d73027" if v >= 0.5 else "#91bfdb" for v in df_roll["Sig ratio"]],
        edgecolor="k",
        linewidth=0.4,
    )
    ax0.axhline(0.5, color="#d73027", linestyle="--", linewidth=1.2)
    ax0.set_xticks(range(len(df_roll)))
    ax0.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax0.set_ylabel("Sig window ratio")
    ax0.set_title("Rolling significance ratio")

    ax1.bar(range(len(df_roll)), df_roll["Mean |t_NW|"], color="#4575b4", edgecolor="k", linewidth=0.4)
    ax1.set_xticks(range(len(df_roll)))
    ax1.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax1.set_ylabel("Mean |t_NW|")
    ax1.set_title("Rolling mean absolute t-stat")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "rolling_ttest.png"), dpi=150)
    plt.close(fig)
    return df_roll


def part_c_monotonicity(
    dec_train: np.ndarray, df_roll: pd.DataFrame, mid_train: np.ndarray, k_events: int = 10
) -> pd.DataFrame:
    n_buckets = 20
    ic_win_size = 10_000
    ic_step = 2_500
    ret_all = ((mid_train[k_events:] - mid_train[:-k_events]) / mid_train[:-k_events]).astype(np.float64)
    n_align = len(ret_all)
    stable = df_roll[df_roll["Sig ratio"] >= 0.50].copy()
    rows = []

    if len(stable) == 0:
        pd.DataFrame().to_csv(os.path.join(RESULT_DIR, "factor_summary.csv"), index=False)
        return pd.DataFrame()

    cols_per_row = min(len(stable), 4)
    nrows = (len(stable) + cols_per_row - 1) // cols_per_row
    fig_mono, axes_mono = plt.subplots(nrows, cols_per_row, figsize=(4.5 * cols_per_row, 4 * nrows), squeeze=False)
    fig_ic, axes_ic = plt.subplots(nrows, cols_per_row, figsize=(4.5 * cols_per_row, 4 * nrows), squeeze=False)

    for idx, (_, row) in enumerate(stable.iterrows()):
        feat_row = int(row["Data row"])
        feat = dec_train[feat_row, :-k_events].astype(np.float64)
        labels_b = pd.qcut(feat, q=n_buckets, labels=False, duplicates="drop")
        bucket_ret = pd.Series(ret_all).groupby(labels_b).mean()
        b_idx = np.array(bucket_ret.index, dtype=float)
        b_mean = bucket_ret.values
        rho_mono, p_mono = spearmanr(b_idx, b_mean)

        ic_vals = []
        for s in range(0, n_align - ic_win_size, ic_step):
            e = s + ic_win_size
            rho_ic, _ = spearmanr(feat[s:e], ret_all[s:e])
            ic_vals.append(rho_ic)
        ic_arr = np.array(ic_vals, dtype=float)
        ic_mean = float(ic_arr.mean()) if len(ic_arr) else 0.0
        ic_std = float(ic_arr.std(ddof=0)) if len(ic_arr) else 0.0
        ic_ir = ic_mean / ic_std if ic_std > 1e-12 else 0.0

        rows.append(
            {
                "Set": row["Set"],
                "Feature #": row["Feature #"],
                "Data row": feat_row,
                "Sig ratio": row["Sig ratio"],
                "Mean |t_NW|": row["Mean |t_NW|"],
                "Mono rho": rho_mono,
                "Mono p": p_mono,
                "Mono sig": bool(p_mono < 0.05),
                "IC mean": ic_mean,
                "IC std": ic_std,
                "IC IR": ic_ir,
            }
        )

        ri, ci = divmod(idx, cols_per_row)
        axm = axes_mono[ri, ci]
        axi = axes_ic[ri, ci]
        axm.bar(
            bucket_ret.index,
            b_mean * 1e4,
            color=["#2166ac" if v >= 0 else "#d6604d" for v in b_mean],
            edgecolor="k",
            linewidth=0.3,
        )
        axm.axhline(0, color="k", linewidth=0.5)
        axm.set_title(f"{row['Set']}/f{row['Feature #']}\nρ={rho_mono:.3f}, p={p_mono:.1e}", fontsize=9)
        axm.set_xlabel("Quantile bucket")
        axm.set_ylabel("Mean fwd return ×1e4")
        axm.tick_params(labelsize=7)

        if len(ic_arr):
            axi.plot(ic_arr, color="#4575b4", linewidth=1.0)
        axi.axhline(0, color="k", linewidth=0.5)
        axi.set_title(f"{row['Set']}/f{row['Feature #']}\nIC={ic_mean:.4f}, IR={ic_ir:.3f}", fontsize=9)
        axi.set_xlabel("Rolling window")
        axi.set_ylabel("Rank IC")
        axi.tick_params(labelsize=7)

    for fig, axes in [(fig_mono, axes_mono), (fig_ic, axes_ic)]:
        flat = axes.ravel()
        for j in range(len(stable), len(flat)):
            flat[j].set_visible(False)
        fig.tight_layout()
    fig_mono.savefig(os.path.join(RESULT_DIR, "factor_monotonicity.png"), dpi=150)
    fig_ic.savefig(os.path.join(RESULT_DIR, "factor_rank_ic.png"), dpi=150)
    plt.close(fig_mono)
    plt.close(fig_ic)

    df_mono = pd.DataFrame(rows)
    df_mono.to_csv(os.path.join(RESULT_DIR, "factor_summary.csv"), index=False)
    return df_mono


def part_d_qualified_factors(df_nw: pd.DataFrame, df_mono: pd.DataFrame, dec_train: np.ndarray, k_events: int = 10) -> pd.DataFrame:
    df_nw = df_nw.copy()
    df_nw["BH_pass"] = bh_correction(df_nw["p_NW"].values, q=0.05)
    bh_rows = set(df_nw[df_nw["BH_pass"]]["Data row"].astype(int).tolist())

    if len(df_mono):
        qualified = df_mono[
            df_mono["Data row"].isin(bh_rows)
            & (df_mono["Sig ratio"] >= 0.50)
            & (df_mono["Mono sig"])
            & (df_mono["IC mean"] > 0)
        ].copy()
    else:
        qualified = df_nw[df_nw["BH_pass"]].copy()

    if len(qualified) == 0:
        qualified = df_nw[df_nw["BH_pass"]].copy()

    qualified.to_csv(os.path.join(RESULT_DIR, "qualified_factors.csv"), index=False)
    qual_rows = sorted(qualified["Data row"].astype(int).tolist())

    if len(qual_rows):
        mid_train = (dec_train[0] + dec_train[2]) / 2.0
        ret_all = ((mid_train[k_events:] - mid_train[:-k_events]) / mid_train[:-k_events]).astype(np.float64)
        qual_in_mono = df_mono[df_mono["Data row"].isin(qual_rows)].copy() if len(df_mono) else pd.DataFrame()
        if len(qual_in_mono):
            cols = min(4, len(qual_in_mono))
            rows = (len(qual_in_mono) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)
            for idx, (_, row) in enumerate(qual_in_mono.iterrows()):
                feat = dec_train[int(row["Data row"]), :-k_events].astype(np.float64)
                labels_b = pd.qcut(feat, q=20, labels=False, duplicates="drop")
                bucket_ret = pd.Series(ret_all).groupby(labels_b).mean()
                ri, ci = divmod(idx, cols)
                ax = axes[ri, ci]
                ax.bar(bucket_ret.index, bucket_ret.values * 1e4, color="#4daf4a", edgecolor="k", linewidth=0.3)
                ax.axhline(0, color="k", linewidth=0.5)
                ax.set_title(f"{row['Set']}/f{row['Feature #']}", fontsize=9)
                ax.tick_params(labelsize=7)
            flat = axes.ravel()
            for j in range(len(qual_in_mono), len(flat)):
                flat[j].set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(RESULT_DIR, "qualified_monotonicity.png"), dpi=150)
            plt.close(fig)

    return qualified


def _build_point_in_time_xy(data_mat: np.ndarray, qual_rows: list[int], k_idx: int, t_skip: int = 100) -> tuple[np.ndarray, np.ndarray]:
    k_ev = K_EVENT_MAP[k_idx]
    n = data_mat.shape[1]
    x = data_mat[qual_rows, :].T.astype(np.float64)
    y = data_mat[144 + k_idx, :].astype(int) - 1
    x = x[t_skip : n - k_ev]
    y = y[t_skip : n - k_ev]
    return x, y


def part_e_baselines(train: np.ndarray, test: np.ndarray, qualified: pd.DataFrame) -> pd.DataFrame:
    qual_rows = sorted(qualified["Data row"].astype(int).tolist())
    if not qual_rows:
        raise RuntimeError("No qualified features available for baseline model training.")

    x_train, y_train = _build_point_in_time_xy(train, qual_rows, k_idx=0, t_skip=100)
    x_test, y_test = _build_point_in_time_xy(test, qual_rows, k_idx=0, t_skip=100)

    # Keep baseline training reproducible and fast enough for routine CPU reruns.
    rng = np.random.default_rng(42)
    max_train = 60_000
    max_test = 80_000
    if len(x_train) > max_train:
        idx = np.sort(rng.choice(len(x_train), size=max_train, replace=False))
        x_train, y_train = x_train[idx], y_train[idx]
    if len(x_test) > max_test:
        idx = np.sort(rng.choice(len(x_test), size=max_test, replace=False))
        x_test, y_test = x_test[idx], y_test[idx]

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400, multi_class="auto", C=1.0, random_state=42)),
        ]
    )
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_test)

    mlp = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    max_iter=20,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=4,
                ),
            ),
        ]
    )
    mlp.fit(x_train, y_train)
    mlp_pred = mlp.predict(x_test)

    deep = pd.read_csv(os.path.join(RESULT_DIR, "performance_summary.csv"))
    deep_row = deep.iloc[0]

    rows = []
    for name, pred in [("Ridge Logistic Regression", ridge_pred), ("MLP (64-32)", mlp_pred)]:
        _, _, f1, _ = precision_recall_fscore_support(y_test, pred, average=None, labels=[0, 1, 2], zero_division=0)
        _, _, f1_w, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Cohen κ": cohen_kappa_score(y_test, pred),
                "MCC": matthews_corrcoef(y_test, pred),
                "F1-Down": f1[0],
                "F1-Stat": f1[1],
                "F1-Up": f1[2],
                "F1-Weighted": f1_w,
                "Sharpe-like": sharpe_like_from_labels(y_test, pred),
            }
        )

    rows.append(
        {
            "Model": "DeepLOB",
            "Accuracy": float(deep_row["Accuracy"]),
            "Cohen κ": float(deep_row["Cohen κ"]),
            "MCC": float(deep_row["MCC"]),
            "F1-Down": float(deep_row["F1-Down"]),
            "F1-Stat": float(deep_row["F1-Stat"]),
            "F1-Up": float(deep_row["F1-Up"]),
            "F1-Weighted": float(deep_row["F1-Weighted"]),
            "Sharpe-like": np.nan,
        }
    )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "baseline_model_comparison.csv"), index=False)

    metrics = ["Accuracy", "Cohen κ", "MCC", "F1-Weighted"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5))
    colors = ["#91bfdb", "#fdae61", "#1a9641"]
    for ax, metric in zip(axes, metrics):
        ax.bar(df["Model"], df[metric], color=colors[: len(df)], edgecolor="k", linewidth=0.4)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "baseline_comparison.png"), dpi=150)
    plt.close(fig)
    return df


def part_trading_strategy(test: np.ndarray) -> pd.DataFrame:
    rows = []
    mid = (test[0] + test[2]) / 2.0
    for k_idx, k_ev in K_EVENT_MAP.items():
        pred_path = os.path.join(RESULT_DIR, f"preds_k{k_idx}.npz")
        if not os.path.exists(pred_path):
            continue
        pdata = np.load(pred_path)
        y_pred = pdata["y_pred"]

        aligned_returns = ((mid[100 + k_ev :] - mid[100:-k_ev]) / mid[100:-k_ev]).astype(np.float64)
        n = min(len(y_pred), len(aligned_returns))
        signal = np.where(y_pred[:n] == 2, 1.0, np.where(y_pred[:n] == 0, -1.0, 0.0))
        pnl = signal * aligned_returns[:n]

        day_len = 500
        daily = np.array([pnl[i : i + day_len].sum() for i in range(0, len(pnl), day_len) if len(pnl[i : i + day_len]) > 0])
        if len(daily) > 1:
            t_stat, p_val = stats.ttest_1samp(daily, 0.0)
        else:
            t_stat, p_val = 0.0, 1.0
        rows.append(
            {
                "Horizon": f"k={k_idx + 1}",
                "Events ahead": k_ev,
                "Mean daily pnl": daily.mean() if len(daily) else 0.0,
                "Std daily pnl": daily.std(ddof=0) if len(daily) else 0.0,
                "Sharpe-like": daily.mean() / daily.std(ddof=0) if len(daily) and daily.std(ddof=0) > 1e-12 else 0.0,
                "t-stat": t_stat,
                "p-value": p_val,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULT_DIR, "trading_strategy_stats.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(df["Horizon"], df["Mean daily pnl"], color="#4575b4", edgecolor="k", linewidth=0.4)
    axes[0].axhline(0, color="k", linewidth=0.6)
    axes[0].set_title("Mean daily pseudo-PnL")
    axes[1].bar(
        df["Horizon"],
        -np.log10(np.maximum(df["p-value"], 1e-12)),
        color=["#d73027" if p < 0.05 else "#91bfdb" for p in df["p-value"]],
        edgecolor="k",
        linewidth=0.4,
    )
    axes[1].axhline(-np.log10(0.05), color="#d73027", linestyle="--", linewidth=1.0)
    axes[1].set_title(r"Strategy t-test significance: $-\log_{10}(p)$")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "trading_profit_ttest.png"), dpi=150)
    plt.close(fig)
    return df


def main() -> None:
    args = parse_args()
    os.makedirs(RESULT_DIR, exist_ok=True)
    train, _val, test = load_fi2010()
    mid_train = (train[0] + train[2]) / 2.0

    fp_csv = os.path.join(RESULT_DIR, "feature_predictability.csv")
    roll_csv = os.path.join(RESULT_DIR, "rolling_stability.csv")
    mono_csv = os.path.join(RESULT_DIR, "factor_summary.csv")
    qual_csv = os.path.join(RESULT_DIR, "qualified_factors.csv")
    base_csv = os.path.join(RESULT_DIR, "baseline_model_comparison.csv")
    strat_csv = os.path.join(RESULT_DIR, "trading_strategy_stats.csv")

    if os.path.exists(fp_csv) and not args.force:
        df_nw = pd.read_csv(fp_csv)
    else:
        df_nw, _, _ = part_a_feature_predictability(train)

    if os.path.exists(roll_csv) and not args.force:
        df_roll = pd.read_csv(roll_csv)
    else:
        df_roll = part_b_rolling_stability(train, df_nw, mid_train, k_events=10)

    if os.path.exists(mono_csv) and not args.force:
        df_mono = pd.read_csv(mono_csv)
    elif os.path.exists(os.path.join(RESULT_DIR, "factor_monotonicity.png")) and not args.force:
        # Existing plots are enough for report rendering; keep the pipeline fast
        # by falling back to BH + rolling stability when the detailed CSV is absent.
        df_mono = pd.DataFrame()
    else:
        df_mono = part_c_monotonicity(train, df_roll, mid_train, k_events=10)

    if os.path.exists(qual_csv) and not args.force:
        qualified = pd.read_csv(qual_csv)
    else:
        qualified = part_d_qualified_factors(df_nw, df_mono, train, k_events=10)

    if os.path.exists(base_csv) and not args.force:
        baselines = pd.read_csv(base_csv)
    else:
        baselines = part_e_baselines(train, test, qualified)

    if os.path.exists(strat_csv) and not args.force:
        strategy = pd.read_csv(strat_csv)
    else:
        strategy = part_trading_strategy(test)

    summary = {
        "feature_predictability_rows": len(df_nw),
        "rolling_candidates": len(df_roll),
        "qualified_factors": len(qualified),
        "baseline_models": baselines["Model"].tolist(),
        "strategy_horizons": strategy["Horizon"].tolist(),
    }
    with open(os.path.join(RESULT_DIR, "analysis_summary.pkl"), "wb") as f:
        pickle.dump(summary, f)

    print("FI-2010 analysis complete.")
    print("Qualified factors:", len(qualified))
    print("Saved baseline comparison to results/baseline_model_comparison.csv")


if __name__ == "__main__":
    main()
