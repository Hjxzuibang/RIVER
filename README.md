# RIVER: Reverse Inference Via Effect Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ICDM 2026](https://img.shields.io/badge/ICDM-2026-blue.svg)]()

Anonymous code submission for ICDM 2026.

> **TL;DR:** RIVER identifies which genes were perturbed from observed single-cell expression changes by learning a causal graph, fitting a neural structural causal model (SCM), and performing greedy counterfactual search — achieving state-of-the-art reverse perturbation prediction across five Perturb-seq datasets.

## Overview

RIVER is a three-stage framework for **reverse perturbation prediction** — identifying causal perturbation targets from single-cell transcriptomic data:

1. **Stage 1 — Causal Graph Learning**: Prior-guided sparse DAG learning via DAGMA with adaptive L1 regularization and biological prior from STRING PPI.
2. **Stage 2 — Neural SCM Training**: Per-gene structural equation models trained on the learned DAG, with optional interventional consistency loss and scheduled sampling.
3. **Stage 3 — Greedy Causal Perturbation Search (GCPS)**: Counterfactual-based greedy search for the minimal intervention set explaining observed expression changes.

## Project Structure

```
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point (inference + metrics)
├── config.py                # Configuration dataclasses
├── configs/                 # Per-dataset YAML configs
│   ├── replogle_k562.yaml   # K562 essential (single-gene)
│   ├── replogle_rpe1.yaml   # RPE1 (single-gene)
│   ├── nadig_hepg2.yaml     # HepG2 (single-gene)
│   ├── nadig_jurkat.yaml    # Jurkat (single-gene)
│   └── norman.yaml          # Norman K562 (multi-gene, signature scoring)
├── model/
│   ├── causal_graph.py      # Stage 1: Prior-guided DAGMA learner
│   ├── neural_scm.py        # Stage 2: Neural Structural Causal Model
│   ├── gcps.py              # Stage 3: Greedy Causal Perturbation Search
│   └── cpd.py               # Unified three-stage pipeline
├── data/
│   ├── dataset.py           # Perturbation dataset (h5ad loader + splits)
│   ├── graph_builder.py     # Prior graph loader
│   └── preprocess.py        # Raw data preprocessing pipeline
├── training/
│   ├── trainer.py           # Three-stage trainer with checkpointing
│   └── metrics.py           # Evaluation metrics (Recall@K, nDCG, AUROC, etc.)
├── scripts/
│   ├── run_all.sh           # Reproduce all paper results
│   └── prepare_data.sh      # Data preprocessing pipeline
├── requirements.txt
└── LICENSE
```

## Installation

**Requirements:** Python >= 3.10, CUDA-capable GPU (recommended)

```bash
git clone <anonymous-repository-url>
cd RIVER
pip install -r requirements.txt
```

## Data Preparation

### Datasets

We evaluate on five Perturb-seq datasets from three studies:

| Dataset | Cell Line | Genes | Test Perts | Source |
|---------|-----------|-------|------------|--------|
| K562 essential | K562 | 3,493 | 182 | [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013) |
| HepG2 | HepG2 | 3,843 | 190 | [Nadig et al. 2024](https://doi.org/10.1038/s41588-024-02048-1) |
| Jurkat | Jurkat | 3,796 | 185 | [Nadig et al. 2024](https://doi.org/10.1038/s41588-024-02048-1) |
| RPE1 | RPE1 | 3,783 | 167 | [Replogle et al. 2022](https://doi.org/10.1016/j.cell.2022.05.013) |
| Norman | K562 | 2,006 | 20 | [Norman et al. 2019](https://doi.org/10.1038/s41588-019-0538-0) |

### Preprocessing

1. Place raw `.h5ad` files under `rawdata/{Dataset}/{CellType}.h5ad`
2. Run the preprocessing pipeline:

```bash
bash scripts/prepare_data.sh
```

Or manually:

```bash
python -m data.preprocess --input-dir rawdata --output-dir processed_data
```

This applies: gene filtering → HVG + perturbation target union (top 2,000) → library-size normalization + log1p → perturbation efficiency filtering (Z > 1.0) → rare perturbation removal (≥ 5 cells).

### Prior Graph

Biological prior graphs are constructed from STRING PPI (v12.0, combined score > 400). Place prior graphs at `processed_data/{Dataset}/prior_graph_{CellType}.pt`.

## Training

### Single dataset

```bash
python train.py --config configs/replogle_k562.yaml
```

### All datasets (reproduce paper)

```bash
bash scripts/run_all.sh
```

Key arguments for `train.py`:

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config file |
| `--seed` | Random seed (overrides config) |
| `--device` | Device, e.g. `cuda` or `cpu` |
| `--resume` | Resume from last checkpoint |

Training produces:
- `results/{exp_name}_seed{seed}/checkpoints/best_model.pt` — model checkpoint
- `results/{exp_name}_seed{seed}/training_history.csv` — per-epoch loss curves
- `results/{exp_name}_seed{seed}/metrics_summary.csv` — test set metrics

## Evaluation

### Standard inference

```bash
python evaluate.py \
    --model_dir results/river_replogle_k562_seed42/checkpoints \
    --data processed_data/Replogle/K562.h5ad \
    --output_dir results/river_replogle_k562_seed42/eval \
    --split test \
    --method RIVER
```

### Per-perturbation fine-tuning

```bash
python evaluate.py \
    --model_dir results/river_replogle_k562_seed42/checkpoints \
    --data processed_data/Replogle/K562.h5ad \
    --output_dir results/river_replogle_k562_seed42/eval_ft \
    --split test \
    --method RIVER \
    --finetune --ft_steps 200 --ft_lr 1e-3 --ft_grad_clip 1.0
```

### Evaluate external predictions

```bash
python evaluate.py \
    --predictions path/to/predictions.npz \
    --data processed_data/Replogle/K562.h5ad \
    --output_dir results/baseline_eval \
    --method BaselineName
```

`predictions.npz` should contain `pert_names` (array of strings) and `pred_scores` (`[n_perts, n_genes]`).

## Hyperparameters

The same configuration is used across all datasets (see Appendix in the paper).

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Stage 1: DAG** | Epochs | 500 |
| | Learning rate | 3 × 10⁻³ |
| | Hidden dimension | 64 |
| | λ_L1 (prior edges) | 0.002 |
| | λ_L1 (non-prior edges) | 0.02 |
| | Adjacency threshold τ | 0.3 |
| **Stage 2: SCM** | Epochs | 300 |
| | Learning rate | 1 × 10⁻³ |
| | Hidden dimension | 64 |
| | Batch size | 256 |
| **Stage 3: GCPS** | Diff. threshold | 0.5 |
| | Max interventions | 5 |
| | Score mode | `raw` (single-gene) / `signature` (multi-gene) |
| **Fine-tuning** | Steps per perturbation | 200 |
| | Learning rate | 1 × 10⁻³ |
| | Gradient clip norm | 1.0 |

## Citation

```bibtex
@inproceedings{river2026,
    title={{RIVER}: Reverse Inference Via Effect Reconstruction},
    author={Anonymous},
    booktitle={IEEE International Conference on Data Mining (ICDM)},
    year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
