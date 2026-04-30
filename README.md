# LLM Leaderboard Analysis

Tools and curated datasets for analyzing redundancy structure in multi-benchmark LLM leaderboards.

## What's here

- **Two cleaned score-matrix datasets** — one from a public leaderboard with 4,486 models, one from a contamination-controlled benchmark with 49 models.
- **A full analysis pipeline** — PCA, per-benchmark novelty diagnostics, exhaustive subset search, linear-vs-GBM comparison, bootstrap CIs, two-stage hybrid recall analysis, floor-effect diagnostic, power analysis, and figure generation.
- **A per-subset rank-fidelity table** — Kendall τ, Spearman ρ, mean R², and top-k overlap for all 62 subsets at k=1..5 on each leaderboard.
- **All paper figures** — both as PDF (vector) and PNG (raster).

## Repository layout
## Repository layout

```

├── README.md             This file
├── DATASHEET.md          Datasheet for both score-matrix datasets
├── LICENSE               MIT for code
├── Makefile              One-shot reproduction (`make all`, `make bootstrap`)
│
├── code/                 Python scripts and dependencies
│   ├── requirements.txt
│   ├── hf_*.py           HF Open LLM analyses
│   ├── livebench_*.py    LiveBench analyses
│   ├── parse_livebench.py
│   ├── extract_metadata.py
│   └── make_figures.py
│
├── data/                 Score matrices and analysis outputs
│   ├── DATA_LICENSE.md   CC-BY-4.0 license for the data
│   ├── score_matrix.csv  HF Open LLM v2 (4486 × 7)
│   ├── score_matrix_croissant.json       Croissant 1.0 metadata
│   ├── livebench_scores.csv  LiveBench (49 × 7)
│   ├── livebench_scores_croissant.json   Croissant 1.0 metadata
│   ├── model_metadata.csv  (heuristic, regenerable)
│   ├── livebench_imputation_full.csv
│   └── livebench_bootstrap_tau.csv
│
└── figures/              Paper figures (PDF + PNG)
```

## Quick start

```
pip install -r code/requirements.txt
make all
```

This runs every analysis end-to-end and regenerates the figures. Total runtime: ~5 minutes on a CPU-only laptop.

## Datasets

### `data/score_matrix.csv` — 4,486 × 7

Snapshot of the Hugging Face Open LLM Leaderboard v2 (April 2026). Columns: `model`, `IFEval`, `BBH`, `MATH`, `GPQA`, `MUSR`, `MMLU_PRO`. Scores are in [0,1].

Source: downloaded from the public `open-llm-leaderboard/contents` dataset on the Hugging Face Hub. Cleaning steps applied:
- Removed rows with any missing benchmark score
- Excluded models with `flagged = true` in upstream metadata
- Deduplicated by model identifier
- Population breakdown: 39% fine-tunes, 38% merges, 16% chat models, 7% pretrained base

### `data/livebench_scores.csv` — 49 × 7

Transcribed verbatim from Table 3 of:
> White, C., et al. *LiveBench: A Challenging, Contamination-Free LLM Benchmark.* ICLR 2025 (Spotlight). arXiv:2406.19314.

Columns: model identifier, Coding, Data Analysis, Instruction Following, Language, Math, Reasoning. Original 0–100 scale normalized to [0,1].

### `data/livebench_imputation_full.csv` — 62 × 7

Per-subset rank-fidelity table for LiveBench at k=1..5. Columns: subset, k, tau, rho, mean R², top-5 overlap, top-10 overlap.

### `data/livebench_bootstrap_tau.csv` — 1000 × 2

1,000 bootstrap resamples of Kendall τ for the best 3-of-6 LiveBench subset (Coding + Data Analysis + Instruction Following) and the worst 3-of-6 subset (Coding + Math + Reasoning).

### Generated outputs (produced by running the pipeline)

- `data/hf_imputation_full.csv` — 62-row per-subset rank-fidelity table for HF v2 (output of `hf_imputation.py`).
- `data/hf_two_stage_pareto.csv` — full Pareto frontier of the two-stage hybrid pipeline at 40 candidate-pool sizes (output of `hf_two_stage.py`).
- `data/hf_power_analysis.csv` — bootstrap CI half-width on Kendall τ at n ∈ {20, 30, 49, 75, 100, 200, 500, 1000} (output of `hf_power_analysis.py`).

## Analysis scripts

| Script | What it does |
|--------|--------------|
| `code/hf_pca.py` | PCA, effective rank, loadings, pairwise correlations, per-benchmark novelty |
| `code/hf_imputation.py` | Exhaustive subset search at k=1..5 (62 subsets) |
| `code/hf_gbm.py` | Linear regression vs gradient boosting on the best 3-of-6 subset |
| `code/hf_robustness.py` | Subsample stability (PCA at n=100..4000) |
| `code/hf_two_stage.py` | Two-stage hybrid pipeline: cost saved vs top-k recall |
| `code/hf_floor_effects.py` | Tests whether small-model effective rank is a floor-effect artifact |
| `code/hf_power_analysis.py` | Bootstrap CI half-width vs sample size |
| `code/livebench_analysis.py` | Same PCA + imputation pipeline applied to LiveBench |
| `code/livebench_bootstrap.py` | 1,000 bootstrap resamples for confidence intervals |
| `code/parse_livebench.py` | Builds `livebench_scores.csv` from White et al. Table 3 |
| `code/make_figures.py` | Generates the 4 main paper figures |

Each script prints its main outputs to stdout and includes assertion-based sanity checks against the numbers reported in the paper.

## Running individual analyses

```
make hf-pca           # PCA + novelty
make hf-imputation    # subset search
make hf-gbm           # linear vs GBM
make hf-twostage      # two-stage hybrid
make hf-floor         # floor-effects test
make hf-power         # power analysis
make livebench        # both LiveBench scripts
make figures          # main figures
```

## Reproducibility note

All scripts use `numpy.random.seed(42)` and `KFold(random_state=42)` for the cross-validation splits. Results should be exactly reproducible across machines for the 5-fold CV outputs; the 1,000-resample bootstrap will be reproducible when the same seeds are used (each bootstrap iteration uses `seed=b` for the b-th resample).

## License

MIT for code, CC-BY-4.0 for data. See LICENSE.

## Citing

A full methodology writeup and analysis are in preparation. Citation details will be added here upon publication.
