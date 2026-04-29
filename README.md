# DeepLOB Project — FI-2010 Reproduction + Optiver Transfer Learning

This repository contains a **cluster-oriented, report-style DeepLOB project** with two linked workflows:

1. **FI-2010 main workflow**  
   Reproduces the DeepLOB CNN + Inception + LSTM pipeline on FI-2010, then extends it with feature testing, factor qualification, baseline models, and strategy-style evaluation.

2. **Optiver application workflow**  
   Adapts DeepLOB to the 2-level Optiver order book, studies causal normalization, zero-shot cross-stock transfer, fine-tuning, and same-stock out-of-sample generalization.

The design principle is:

- **scripts + SLURM jobs** do heavy computation,
- **notebooks** load saved artifacts and present a clean project report.

The final report scope is now intentionally narrower than the full experiment history:

- **FI report slice**: only `k=3 / 4 / 5` (`30 / 50 / 100` events)
- **Optiver report slice**: only `k=5` on held-out stocks `5, 17, 42, 108, 120`
- the cleaned project keeps only these retained configurations in the active tree

---

## 1. Repository map

```text
DeepLOB/
├── data/
│   ├── Train_Dst_NoAuction_DecPre_CF_7.txt
│   ├── Test_Dst_NoAuction_DecPre_CF_7.txt
│   ├── Test_Dst_NoAuction_DecPre_CF_8.txt
│   ├── Test_Dst_NoAuction_DecPre_CF_9.txt
│   └── optiver_processed/                  # generated Optiver per-stock .npz files
├── logs/                                   # SLURM stdout/stderr + monitoring logs
├── models/
│   ├── deeplob_k*.pt
│   ├── optiver_base_k*.pt
│   ├── optiver_base_k*_state.pt
│   ├── optiver_transfer_s*_k*.pt
│   ├── optiver_transfer_s*_k*_model.pt
│   └── optiver_specific_s*_k*.pt
├── results/
│   ├── *.png / *.csv / *.npz / *.pkl       # FI-2010 generated artifacts
│   └── optiver/                            # final retained Optiver artifacts
├── scripts/
│   ├── train_deeplob.py
│   ├── analyze_fi2010.py
│   ├── prepare_optiver.py
│   ├── train_optiver.py
│   └── analyze_optiver.py
├── run_deeplob_pytorch.ipynb               # FI-2010 final report notebook
├── run_optiver.ipynb                       # Optiver final report notebook
├── submit_deeplob.sh                       # FI-2010 end-to-end training + analysis job
└── submit_optiver.sh                       # Optiver end-to-end GPU workflow
```

---

## 2. File dependency graph

### FI-2010 chain

```text
FI raw txt files
   ↓
scripts/train_deeplob.py
   ↓
models/deeplob_k*.pt
results/losses_k*.npz
results/preds_k*.npz
results/loss_k*.png
results/cm_k*.png
results/performance_summary.csv
results/all_results.pkl
   ↓
scripts/analyze_fi2010.py
   ↓
feature / factor / baseline / strategy artifacts in results/
   ↓
run_deeplob_pytorch.ipynb
```

### Optiver chain

```text
optiver-realized-volatility-prediction.zip
   ↓
scripts/prepare_optiver.py
   ↓
data/optiver_processed/stock_*.npz
data/optiver_processed/stock_split.json
   ↓
scripts/train_optiver.py
   ↓
models/optiver_base_*.pt
models/optiver_transfer_*.pt
models/optiver_specific_*.pt
results/optiver/base_metrics_*.pkl
results/optiver/transfer_metrics_*.pkl
results/optiver/specific_metrics_*.pkl
results/optiver/*.png
   ↓
scripts/analyze_optiver.py
   ↓
results/optiver/figure6~9_*.png
results/optiver/transfer_analysis_summary.json
   ↓
run_optiver.ipynb
```

---

## 3. Cluster environment and Python dependencies

This project targets **PSC Bridges-2**.

### Modules

```bash
module purge
module load AI/pytorch_23.02-1.13.1-py3
module load gcc/13.3.1-p20240614
```

### User site-packages

Scripts assume user packages are available under:

```bash
/ocean/projects/mth250011p/xxiao7/pyuser/lib/python3.10/site-packages
```

Important packages used by the workflows:

- `torch`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `statsmodels`
- `tqdm`
- `pyarrow` (required for Optiver parquet reading)

### Runtime behavior

- GPU jobs are submitted with SLURM shell wrappers.
- Long-running training is **not** done interactively in notebooks.
- Notebooks are intended to be the **final readable report surface**.

---

## 4. Workflow A — FI-2010 main report

### What this workflow does

1. Train the retained DeepLOB horizons only: `k=3 / 4 / 5`.
2. Save loss curves, confusion matrices, predictions, and summary metrics.
3. Run feature testing, qualified-factor selection, baseline manual-factor models, and trading-style analysis in the same job.
4. Present everything in `run_deeplob_pytorch.ipynb`.

### Entry points

#### Step A1 — GPU training

```bash
cd /ocean/projects/mth250011p/xxiao7/DeepLOB
sbatch submit_deeplob.sh
```

This launches:

```bash
python3 scripts/train_deeplob.py
python3 scripts/analyze_fi2010.py
```

### FI technical details

- **Input**: FI-2010 LOB tensor with shape `(B, 1, T, 40)`.
- **Feature order** follows the original DeepLOB layout.
- **Model**: 3 convolution blocks + inception module + LSTM + FC classifier,
   with an optional auxiliary regression head that predicts same-horizon future
   log return from the shared representation.
- **Training policy used in the final report**:
  - batch size `32`
  - lookback `100`
  - max epochs `100`
  - `--horizon-profile adaptive`, which assigns separate learning-rate,
    regularization, patience, and monitor choices to `k_idx=0..4`
   - an auxiliary Huber loss on future log return is enabled by default in the
      adaptive profile to regularize short horizons whose validation loss is most
      jagged
  - short horizons use macro-F1/MCC-oriented checkpointing to reduce
    stationary-class collapse visible in their confusion matrices
  - longer horizons use validation-loss checkpointing with stronger
    regularization because their loss curves overfit earlier
  - the 100-event horizon (`k_idx=4`, shown as `k=5`) uses:
    - learning rate `7e-4`
    - weight decay `3e-4`
    - dropout `0.30`
    - label smoothing `0.02`
    - gradient clipping `1.0`
    - checkpoint / early stopping monitored by validation loss

### Why the exact paper optimizer was not kept verbatim

The direct `Adam + lr=0.01` paper-style setting caused majority-class collapse in this environment: validation accuracy froze at the stationary-class prior and validation loss exploded. The current setup preserves the **paper architecture and experimental logic** but uses a **stabilized optimizer configuration** that trains reliably on this codebase.

### FI analysis outputs

`scripts/analyze_fi2010.py` depends on the training outputs and builds:

- engineered-feature Newey-West predictability tests,
- rolling stability and monotonicity checks,
- FDR-qualified factor selection,
- baseline models,
- simple strategy-style statistics.

Representative outputs:

- `results/feature_predictability.csv`
- `results/rolling_stability.csv`
- `results/qualified_factors.csv`
- `results/baseline_model_comparison.csv`
- `results/trading_strategy_stats.csv`
- related `png` visualizations

#### Step A2 — notebook report

Open and run:

```text
run_deeplob_pytorch.ipynb
```

This notebook is the single FI-2010 report notebook. It should be run after `submit_deeplob.sh` finishes.

---

## 5. Workflow B — Optiver + transfer learning

### What this workflow does

1. Read the Optiver zip dataset.
2. Build per-stock normalized `.npz` files.
3. Train DeepLOBLite on source stocks.
4. Evaluate zero-shot on unseen stocks.
5. Fine-tune on held-out stocks.
6. Train same-stock temporal out-of-sample models.
7. Optionally generate paper-style transfer figures and notebook-ready summaries.

### Entry point

```bash
cd /ocean/projects/mth250011p/xxiao7/DeepLOB
sbatch submit_optiver.sh
sbatch --export=ALL,OPTIVER_LABEL_MODE=rolling-quantile-3class submit_optiver.sh
```

This SLURM wrapper runs two phases by default and a third optional phase when `OPTIVER_RUN_ANALYSIS=1`:

1. `scripts/prepare_optiver.py`
2. `scripts/train_optiver.py`
3. `scripts/analyze_optiver.py` (optional)

### Optiver technical details

#### 2-level architecture adaptation

Optiver provides only the first two bid/ask levels, so the original 40-feature DeepLOB input is reduced to 8 features:

```text
[ask_price1, ask_size1, bid_price1, bid_size1,
 ask_price2, ask_size2, bid_price2, bid_size2]
```

This preserves the DeepLOB logic where the first `(1×2, stride=2)` convolution merges **price + size within the same side and level**.

#### Label construction

Labels are not provided directly. They are constructed from event-horizon mid-price returns:

```text
return_t(k) = (mid_{t+k} - mid_t) / mid_t
```

Then a volatility-adaptive threshold produces:

- `0 = Down`
- `1 = Stationary`
- `2 = Up`

#### Causal normalization

Preprocessing now defaults to a **causal event-wise rolling Z-score with clipping**. This keeps feature scales bounded across stocks while preserving the no-lookahead property. The older `time-id` bucket normalization is still available as an opt-in mode, but it is no longer the default because some stocks produced numerically extreme normalized values under that scheme.

Within the retained final Optiver workflow, the base-training sample mode is fixed to `volatility`.

#### Train / transfer split

The stock split is configurable in `scripts/prepare_optiver.py`. The current default holds out **10 interleaved stocks** rather than taking one contiguous `80/32` block, so the transfer set is spread across the stock-id range and the base model trains on the remaining **102 stocks**.

#### Transfer regimes implemented

`scripts/train_optiver.py` explicitly separates three settings:

1. **Universal zero-shot transfer**  
   Train on source stocks, test directly on unseen stocks.

2. **Fine-tuned transfer**  
   Freeze convolution + inception layers and fine-tune the LSTM + classifier head.

3. **Specific-stock temporal OOS**  
   Train on the earlier part of one stock and test on its later segment.

#### Training stabilizers

Because Optiver labels and confusion matrices were much less stable than
FI-2010, the current workflow intentionally does **not** reuse the FI horizon
parameters. It uses:

- per-stock relabeling during training to prevent trivial majority-class predictions:
   the retained Optiver workflow now keeps two 3-class variants:
   `Original 3` uses the prepared adaptive rolling-std labels via
   `--label-mode original`, while `rolling 3` uses a rolling historical
   quantile threshold on each stock's k-step returns via
   `--label-mode rolling-quantile-3class --quantile-stationary 0.2`;
   explicit CLI / environment label overrides are applied after the adaptive k=5
   defaults, so retained overrides now win instead of being silently reset,
- focal loss plus horizon-specific class balancing,
- checkpoint / early stopping monitored by validation κ,
- dropout before the classifier head,
- gradient clipping,
- validation-loss plateau LR scheduling,
- temporal validation during base training so early stopping sees later-in-time windows instead of a random window mix from the same stock,
- an auxiliary log-return regression head with Huber loss so the shared trunk
   is trained against both discrete direction labels and continuous future
   price moves,
- horizon-specific transfer settings:
  - k=1 keeps transfer to the LSTM/head with a smaller base LR,
   - k=5 and k=10 switch to stronger balancing and fine-tune
      `conv3 + LSTM + head` to adapt late spatial-temporal features under
      stronger cross-stock distribution shift.

### Optiver outputs

Training writes:

- `results/optiver/classification_original3/` for `Original 3`
- `results/optiver/classification_rolling3_w20000/` for the default `rolling 3`
- inside each mode-specific directory: `base_metrics_k*.pkl`, `transfer_metrics_k*.pkl`, `specific_metrics_k*.pkl`, `base_loss_k*.png`, `transfer_comparison_k*.png`, and related training figures

Practical note: full-window base training on all prepared Optiver events is not the default because the processed train split contains on the order of $10^8$ valid windows. The workflow therefore keeps a configurable per-stock cap for base and transfer sampling, while still allowing `<=0` to request the uncapped path for targeted experiments.

When `OPTIVER_RUN_ANALYSIS=1`, analysis also writes:

- `figure6_transfer_accuracy.png`
- `figure7_transfer_profit_tstats.png`
- `figure8_transfer_cum_profit.png`
- `figure9_transfer_lime.png`
- `transfer_analysis_summary.json`

#### Notebook report

Open and run:

```text
run_optiver.ipynb
```

This is the **single Optiver report notebook**. It is both:

- a compact dataset understanding notebook,
- and the final transfer-learning report.

---

## 6. Notebook philosophy

Both notebooks are intentionally lightweight:

- they **do not** own the expensive training,
- they **do** explain how training is done,
- they **do** load all final figures and tables needed for the project report.

This makes them suitable for:

- final project write-up,
- cluster-friendly reproducibility,
- rerunning with fresh artifacts,
- presentation and review.

---

## 7. Recommended end-to-end usage

### If you want the FI-2010 report

```bash
sbatch submit_deeplob.sh
```

Then open:

```text
run_deeplob_pytorch.ipynb
```

### If you want the Optiver transfer-learning report

```bash
sbatch submit_optiver.sh
sbatch --export=ALL,OPTIVER_LABEL_MODE=rolling-quantile-3class submit_optiver.sh
```

This training-first path now keeps two retained Optiver variants for the final `k=5` slice: `Original 3` and `rolling 3`.

If you also need the paper-style figures in the same output directory, rerun with `OPTIVER_RUN_ANALYSIS=1`.

Then open:

```text
run_optiver.ipynb
```

### If artifacts are missing or stale

Re-run the corresponding job with force:

```bash
sbatch submit_deeplob.sh --force
sbatch submit_optiver.sh --force
```

---

## 8. Practical reading order for learning this project

If your goal is to understand the project quickly:

1. Read this `README.md`.
2. Open `run_deeplob_pytorch.ipynb` for the main benchmark workflow.
3. Open `run_optiver.ipynb` for the application and transfer-learning extension.
4. If you want implementation details, inspect:
   - `scripts/train_deeplob.py`
   - `scripts/analyze_fi2010.py`
   - `scripts/prepare_optiver.py`
   - `scripts/train_optiver.py`
   - `scripts/analyze_optiver.py`

If your goal is to reproduce results on cluster:

1. submit the corresponding SLURM job(s),
2. wait for artifacts to finish,
3. run the notebooks as the final report surface.
