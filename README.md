# DriverFormer 
# DriverFormer

A deep learning framework for **genome-wide driver discovery**.
DriverFormer trains a Transformer + NHPP head to predict **bin-wise mutation intensity**, then runs a **variable-length LLR → GMM → DP** pipeline with multiple calibration checks to call **driver candidate intervals**.

---

## 🎯 Overview

End-to-end pipeline for:

* **NHPP training** with length-consistent rolling (box), robust Huber/IRLS-like segment weighting, and scale calibration (`log_c`)
* **Variable-length LLR scan** from bin predictions
* **Global GMM** (BIC auto/fixed) to compute p-values
* **DP selection** (weighted interval scheduling) with optional gaps
* **Post-selection FDR** (Storey or BH) and calibration plots (Deviance/df, Pearson χ²/df, Coverage 80/90)

---

## 📋 Features

* ✅ **Modular src layout** (`driverformer/*`) for clean reuse and packaging
* ✅ **Transformer encoder + NHPP head** with per-kb likelihood training
* ✅ **Roll-consistent smoothing**: apply the same box window to `y`, `μ=λΔt`, and `Δt`
* ✅ **Robust training**: Huber segment weights + residual-weighted sampler
* ✅ **Attention dumps**: last-layer snapshot per improvement, full stack at the end
* ✅ **LLR→GMM→DP**: variable intervals, BIC auto-k, gap-aware DP
* ✅ **Calibration**: Deviance/df, Pearson χ²/df, Coverage (80/90), QQ plot
* ✅ **Repro-friendly CLI** and Python API

---

## 🛠️ Installation

> You can install with or without pulling a new PyTorch build.

**A. Editable install (CUDA PyTorch will be pulled if not present)**

```bash
pip install -e .
driverformer --help
```

**B. Reuse existing PyTorch (skip heavy CUDA downloads)**

```bash
pip install -e . --no-deps
```

**C. CPU-only (lightweight)**

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e . --no-deps
```

---

## 📁 Project Structure

```
src/driverformer/
├── cli.py                # Entrypoint (argparse)
├── config.py             # CLI builder
├── data/                 # segments, labels, rolling
├── models/               # embedders, transformer, NHPP head
├── losses/               # NHPP training losses
├── train/                # training loop, attention dump
├── pipeline/             # llr_scan, gmm, dp, postsel, run
├── eval/                 # weighted objective, utilities
└── utils/                # io, stats, chrom helpers, plotting
```

---

## 🚀 Quick Start

### 1) Train (Transformer + NHPP)

```bash
driverformer \
  --cls-file /path/to/cls.pkl \
  --feat-file /path/to/feat.pkl \
  --mutations-file /path/to/events.csv \   # optional: build 1kb labels (PASS only)
  --out-dir OUT                             \
  --segment-lengths 10 50 100 \
  --batch-size 16 --epochs 30 --lr 2e-3 \
  --label-roll --label-roll-width 2 \
  --save-attention
```

**Outputs**

* `OUT/trained_model.pt` – best checkpoint
* `OUT/train_prediction.csv`, `OUT/val_prediction.csv`, `OUT/all_prediction.csv`
  (columns: `chrom,start,end,lam_pred,obs_count,bin_loglike`)
* `OUT/attn/…` – attention tensors (optional)
* `OUT/train_val_loss_per_epoch.png` – training curves

### 2) Pipeline-only (LLR→GMM→DP)

```bash
driverformer \
  --pipeline-only \
  --all-pred OUT/all_prediction.csv \
  --pipeline-out-dir OUT/postproc \
  --pipeline-gmm-auto \
  --pipeline-beta 1.5 --pipeline-gamma 0.5 \
  --pipeline-dp-gap-bp 0
```

**Pipeline outputs**

* `OUT/postproc/final_result.csv`
  (`chrom,start,end,len_bp,LLR_raw,obs_sum,exp_sum,p_post,fdr_post(=fdr),…`)
* `OUT/postproc/qq_plot.pdf`
* `OUT/postproc/llr_intervals.csv`, `prediction_per_bin.csv`

---

## 🧾 Data & I/O

* **Training inputs**

  * `cls.pkl`: list of `(chrom, w_idx, start_bp, end_bp, cls_vector, y_value)`
  * `feat.pkl`: dict `{(chrom, w_idx) -> feature_vector}`
  * Optional `--mutations-file` (`CSV/TSV`): columns `[chrom,start,end,event_type,sample,(filter)]`
    → converts to **1kb** labels (unique samples per bin; PASS only)

* **Prediction CSVs**

  * `lam_pred`: predicted per-kb rate (smoothed if `--label-roll-width>1`)
  * `obs_count`: original observed counts per bin

---

## 🔧 Key Options (selected)

* `--label-roll`, `--label-roll-width`
  Apply length-consistent box smoothing to training space (`y, μ, Δt`)
* `--save-attention`
  Save last-layer attention on validation improvements and final stack
* Pipeline:

  * `--pipeline-gmm-auto` (BIC) or `--pipeline-gmm-k` (fixed)
  * `--pipeline-beta`, `--pipeline-gamma` – DP weights: `LLR * (−log10 p)^β * (len_kb)^γ`
  * `--pipeline-dp-gap-bp` – enforce minimal gaps between intervals
  * Post-selection FDR: `--postsel-fdr-method {storey,bh}`

---

## 🐍 Python API

```python
from driverformer.train.loop import train_and_predict
from driverformer.pipeline.run import run_llr_gmm_dp_pipeline

# Train
class Args: ...
args = Args()
args.cls_file = "cls.pkl"
args.feat_file = "feat.pkl"
args.out_dir = "OUT"
args.label_roll = True
args.label_roll_width = 2
# ... set other fields as in CLI
all_pred_csv = train_and_predict(args)

# Pipeline
from pathlib import Path
run_llr_gmm_dp_pipeline(
    all_pred_path=all_pred_csv,
    out_dir=Path("OUT/postproc"),
    gmm_auto=True,
    beta=1.5, gamma=0.5,
)
```

---

## 📐 Calibration & Evaluation

* **Residual Deviance / df** (≈1 ideal; »1 over-dispersion/mismatch)
* **Pearson χ² / df** (≈1 ideal; >1 underest. variance; <1 over-est.)
* **Coverage (80/90%)** – empirical vs normal intervals
* **QQ plot** – post-selection p-values (Storey/BH)

> We recommend reporting **length-weighted averages** and showing a single **BIN-size trend** figure (or heatmap) to reveal any scale sensitivities.

---

## 🧪 Troubleshooting

* **Large downloads on install**
  → Remove `torch` from `dependencies` and run `pip install -e . --no-deps`, or install CPU-only Torch first.
* **ImportError after split**
  → Ensure `pip install -e .` from repo root; check `src/driverformer/**/__init__.py` exists.
* **CUDA issues**
  → Verify `TORCH_CUDA_ARCH_LIST`, CUDA driver version, or use CPU-only torch to validate logic.
* **No intervals selected**
  → Inspect `prediction_per_bin.csv`, try `--pipeline-sample-frac`↑ (for GMM), lower stringency, or check masking.

---

## 📚 Citation

If you use DriverFormer in your research, please cite this repository.
(Preprint/manuscript details to be added.)

```
@software{driverformer,
  author  = {Seo, Eunyoung (silverflo)},
  title   = {DriverFormer: NHPP-based transformer training and LLR→GMM→DP pipeline},
  year    = {2025},
  url     = {https://github.com/silverflower10/DriverFormer}
}
```


---

**Happy hotspot hunting!** 🔬🚀
