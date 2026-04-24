#!/usr/bin/env python3
"""
prepare_optiver.py — Preprocess Optiver LOB data for DeepLOB training
======================================================================
Reads the Optiver Realized Volatility Prediction dataset (2-level order
book snapshots), constructs 3-class mid-price direction labels, applies
causal normalization, and saves per-stock LOB arrays as .npz files.

Data layout after preprocessing (saved to data/optiver_processed/):
  stock_{id}_data.npz   — keys:
      X                (T_events × 8) causally normalized LOB features
      mid              (T_events,)     raw mid-prices
      time_id          (T_events,)     Optiver auction bucket ID
      seconds_in_bucket(T_events,)     within-bucket timestamp
      y_k              (T_events,)     labels for each horizon

Feature arrangement (8 features per event) — follows FI-2010 interleaved layout:
  ask_price1, ask_size1, bid_price1, bid_size1,   ← level 1 (ask first, then bid)
  ask_price2, ask_size2, bid_price2, bid_size2    ← level 2

  Each adjacent pair (price, size) is merged by Conv Block 1's (1×2) stride-2
  convolution, which then captures price+size information per side per level.
  This matches Zhang et al. (2019):  {p_ask^(i), v_ask^(i), p_bid^(i), v_bid^(i)}.

Label construction:
  mid_price = (ask_price1 + bid_price1) / 2
  k-step return r_t = (mid_t+k - mid_t) / mid_t
  Threshold θ = α × rolling_std(r_t, window=200)  [stock-specific]
  label =  0 (Down)       if r_t < -θ
           1 (Stationary) if |r_t| ≤ θ
           2 (Up)         if r_t >  θ

Normalization:
  By default, each stock is normalized causally with statistics from the
  previous 5 `time_id` buckets. This follows the spirit of the LSE setup in
  DeepLOB, where rolling standardization uses only past information.
  A faster event-wise rolling normalization is also available as an option.

Train/Transfer split:
  112 stocks total.
  Train stocks  : stock IDs 0..79   (80 stocks)
  Transfer stocks: stock IDs 80..111 (32 stocks)
  The split is deterministic (sorted stock_id) so it is reproducible.

Usage
-----
    python scripts/prepare_optiver.py [--zip PATH] [--out-dir PATH] [--alpha 0.002]
"""
import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Preprocess Optiver LOB data")
    p.add_argument("--zip",     default=os.path.join(BASE_DIR, "optiver-realized-volatility-prediction.zip"),
                   help="Path to Optiver competition zip file")
    p.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data", "optiver_processed"),
                   help="Output directory for processed .npz files")
    p.add_argument("--alpha",   type=float, default=0.002,
                   help="Threshold multiplier α for label construction (default: 0.002)")
    p.add_argument("--roll-norm", type=int, default=100,
                   help="Rolling event window for event-wise normalization (default: 100)")
    p.add_argument("--norm-mode", choices=["time-id", "event"], default="time-id",
                   help="Causal normalization mode (default: time-id)")
    p.add_argument("--norm-time-window", type=int, default=5,
                   help="Number of past time_id buckets used for causal normalization (default: 5)")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 5, 10],
                   help="Prediction horizons in events (default: 1 2 3 5 10)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing per-stock .npz files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature arrangement — mirrors FI-2010 interleaved layout
#
# FI-2010 per-level ordering (Zhang et al. 2019):
#   {p_ask^(i), v_ask^(i), p_bid^(i), v_bid^(i)}  for i = 1..10
#
# This lets Conv Block 1 (kernel (1,2), stride 2) merge price+size pairs
# WITHIN the same side at each level, not across sides:
#   pair 0 → (ask_price1, ask_size1)   ask  side, level 1
#   pair 1 → (bid_price1, bid_size1)   bid  side, level 1
#   pair 2 → (ask_price2, ask_size2)   ask  side, level 2
#   pair 3 → (bid_price2, bid_size2)   bid  side, level 2
# ---------------------------------------------------------------------------
LOB_FEATURE_COLS = [
    "ask_price1", "ask_size1", "bid_price1", "bid_size1",   # level 1
    "ask_price2", "ask_size2", "bid_price2", "bid_size2",   # level 2
]

# Column indices for mid-price computation (after re-ordering above)
_IDX_ASK_P1 = 0   # ask_price1
_IDX_BID_P1 = 2   # bid_price1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling Z-score normalisation along axis 0 (event axis).

    Uses a causal (backward-looking) window so there is no lookahead bias.
    The first `window` events use an expanding window (min_periods=1).

    Vectorised via pandas for speed (~100× faster than a Python loop).
    """
    df   = pd.DataFrame(arr)
    roll = df.rolling(window=window, min_periods=1)
    mu   = roll.mean().values
    std  = roll.std(ddof=0).values                         # population std, no NaN at t=0
    std  = np.where(std < 1e-8, 1.0, std)                  # avoid div-by-zero
    return ((arr - mu) / std).astype(np.float32)


def causal_time_id_zscore(df_book: pd.DataFrame, feature_cols: list[str], window: int) -> np.ndarray:
    """Normalize each time_id bucket using only previous buckets' statistics.

    This is a bucket-level approximation to the paper's rolling daily
    normalization: all events within a bucket share the same causal mean/std
    estimated from the previous `window` buckets of the same stock.
    """
    grouped = df_book.groupby("time_id", sort=True)[feature_cols]
    bucket_mean = grouped.mean()
    bucket_std = grouped.std(ddof=0).replace(0.0, np.nan)
    bucket_second = grouped.apply(lambda g: (g ** 2).mean())

    hist_mean = bucket_mean.rolling(window=window, min_periods=1).mean().shift(1)
    hist_second = bucket_second.rolling(window=window, min_periods=1).mean().shift(1)
    hist_var = (hist_second - hist_mean.pow(2)).clip(lower=1e-8)
    hist_std = hist_var.pow(0.5)

    first_mean = bucket_mean.iloc[0]
    first_std = bucket_std.iloc[0].fillna(1.0)
    hist_mean = hist_mean.fillna(first_mean)
    hist_std = hist_std.fillna(first_std).replace(0.0, 1.0)

    norm_stats = pd.concat(
        [hist_mean.add_prefix("mu_"), hist_std.add_prefix("std_")],
        axis=1,
    ).reset_index()
    df_norm = df_book[["time_id"]].merge(norm_stats, on="time_id", how="left")

    mu = df_norm[[f"mu_{c}" for c in feature_cols]].to_numpy(dtype=np.float64)
    std = df_norm[[f"std_{c}" for c in feature_cols]].to_numpy(dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    arr = df_book[feature_cols].to_numpy(dtype=np.float64)
    return ((arr - mu) / std).astype(np.float32)


def make_labels(mid: np.ndarray, horizons: list, alpha: float,
                roll_std_win: int = 200) -> dict:
    """Build 3-class direction labels for each horizon.

    Parameters
    ----------
    mid : 1-D array of mid-prices
    horizons : list of int — look-ahead event counts
    alpha : threshold multiplier (fraction of rolling std)
    roll_std_win : window for computing adaptive threshold

    Returns
    -------
    dict mapping horizon → 1-D int array of labels (0/1/2)
    """
    N = len(mid)
    labels = {}
    for k in horizons:
        if k >= N:
            continue
        ret = np.zeros(N, dtype=np.float64)
        ret[: N - k] = (mid[k:] - mid[: N - k]) / (mid[: N - k] + 1e-10)

        # Adaptive threshold: alpha × rolling std of returns (vectorised)
        threshold = alpha * pd.Series(ret).rolling(roll_std_win, min_periods=1).std(ddof=0).values
        threshold = np.where(np.isnan(threshold), 0.0, threshold)

        lbl = np.ones(N, dtype=np.int32)                # default: Stationary
        lbl[ret >  threshold] = 2                        # Up
        lbl[ret < -threshold] = 0                        # Down
        lbl[N - k :] = -1                                # no valid label at end
        labels[k] = lbl
    return labels


def process_stock(df_book: pd.DataFrame, horizons: list, alpha: float,
                  roll_norm: int, norm_mode: str, norm_time_window: int) -> dict | None:
    """Process a single stock's book data into normalised LOB arrays.

    Parameters
    ----------
    df_book : DataFrame with columns bid_price1, ask_price1, ...
    horizons, alpha, roll_norm : see prepare_args

    Returns
    -------
    dict with keys "X", "mid", "time_id", "seconds_in_bucket", "y_{k}" for
    each horizon, or None if too few rows.
    """
    # Sort by time to get chronological LOB snapshots
    if "time_id" in df_book.columns and "seconds_in_bucket" in df_book.columns:
        df_book = df_book.sort_values(["time_id", "seconds_in_bucket"]).reset_index(drop=True)

    # Check required columns
    for col in LOB_FEATURE_COLS:
        if col not in df_book.columns:
            return None

    if len(df_book) < max(horizons) + 200:
        return None

    # Raw feature matrix (events × 8)
    X_raw = df_book[LOB_FEATURE_COLS].values.astype(np.float64)

    # Handle NaN/inf (forward-fill then zero-fill)
    df_x = pd.DataFrame(X_raw, columns=LOB_FEATURE_COLS).ffill().fillna(0.0)
    X_raw = df_x.values.astype(np.float64)

    # Mid-price for labelling (use raw prices before normalisation)
    # Indices follow LOB_FEATURE_COLS: ask_price1=col 0, bid_price1=col 2
    mid = (X_raw[:, _IDX_ASK_P1] + X_raw[:, _IDX_BID_P1]) / 2.0

    # Causal normalization
    if norm_mode == "time-id" and "time_id" in df_book.columns:
        X_norm = causal_time_id_zscore(df_book, LOB_FEATURE_COLS, window=norm_time_window)
    else:
        X_norm = rolling_zscore(X_raw, window=roll_norm).astype(np.float32)

    # Labels for each horizon
    label_dict = make_labels(mid, horizons, alpha=alpha)

    result = {
        "X": X_norm,
        "mid": mid.astype(np.float32),
        "time_id": df_book["time_id"].to_numpy(dtype=np.int32) if "time_id" in df_book.columns else np.zeros(len(df_book), dtype=np.int32),
        "seconds_in_bucket": (
            df_book["seconds_in_bucket"].to_numpy(dtype=np.int32)
            if "seconds_in_bucket" in df_book.columns
            else np.arange(len(df_book), dtype=np.int32)
        ),
    }
    for k, lbl in label_dict.items():
        result[f"y_{k}"] = lbl.astype(np.int8)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Opening zip: {args.zip}")
    with zipfile.ZipFile(args.zip, "r") as zf:
        # Discover all book_train stock IDs
        stock_files = {}
        for name in zf.namelist():
            if name.startswith("book_train.parquet/stock_id=") and name.endswith(".parquet"):
                parts = name.split("/")
                sid   = int(parts[1].replace("stock_id=", ""))
                stock_files.setdefault(sid, []).append(name)

        stock_ids = sorted(stock_files.keys())
        print(f"Found {len(stock_ids)} stocks: {stock_ids[:5]} ... {stock_ids[-5:]}")

        # Split into train / transfer
        n_train    = 80
        train_ids  = stock_ids[:n_train]
        transfer_ids = stock_ids[n_train:]
        print(f"Train stocks  : {len(train_ids)}  ({train_ids[0]}..{train_ids[-1]})")
        print(f"Transfer stocks: {len(transfer_ids)} ({transfer_ids[0]}..{transfer_ids[-1]})")

        # Save split metadata
        split = {"train": train_ids, "transfer": transfer_ids}
        import json
        with open(os.path.join(args.out_dir, "stock_split.json"), "w") as f:
            json.dump(split, f, indent=2)

        # Process each stock
        for i, sid in enumerate(stock_ids):
            out_path = os.path.join(args.out_dir, f"stock_{sid}_data.npz")
            if os.path.exists(out_path) and not args.force:
                print(f"  [{i+1}/{len(stock_ids)}] stock {sid}: skip (exists)")
                continue

            # Load all parquet parts for this stock
            frames = []
            for fname in stock_files[sid]:
                with zf.open(fname) as fobj:
                    frames.append(pd.read_parquet(io.BytesIO(fobj.read())))
            df_book = pd.concat(frames, ignore_index=True)

            result = process_stock(
                df_book,
                horizons=args.horizons,
                alpha=args.alpha,
                roll_norm=args.roll_norm,
                norm_mode=args.norm_mode,
                norm_time_window=args.norm_time_window,
            )
            if result is None:
                print(f"  [{i+1}/{len(stock_ids)}] stock {sid}: skip (insufficient data)")
                continue

            tmp_out_path = out_path + ".tmp.npz"
            np.savez_compressed(tmp_out_path, **result)
            os.replace(tmp_out_path, out_path)

            n_events = result["X"].shape[0]
            # Label stats for primary horizon
            k0 = args.horizons[0]
            lbl = result[f"y_{k0}"]
            valid = lbl[lbl >= 0]
            dist  = {0: (valid==0).sum(), 1: (valid==1).sum(), 2: (valid==2).sum()}
            split_tag = "TRAIN" if sid in train_ids else "TRANSFER"
            print(f"  [{i+1}/{len(stock_ids)}] stock {sid} [{split_tag}]: "
                  f"{n_events:,} events | "
                  f"k={k0}: D={dist[0]} S={dist[1]} U={dist[2]} "
                  f"(bal={min(dist.values())/max(max(dist.values()),1):.2f})")

    print(f"\nDone. Processed data saved to {args.out_dir}")

    # Print summary stats
    files = [f for f in os.listdir(args.out_dir) if f.endswith("_data.npz")]
    print(f"Total files: {len(files)}")


if __name__ == "__main__":
    main()
