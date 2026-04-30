# Datasheet

This datasheet follows the format proposed by Gebru et al. (2021),
"Datasheets for Datasets," for the two score-matrix datasets released in
this repository.

## Motivation

**Why was the dataset created?**
To analyze the redundancy structure of multi-benchmark LLM leaderboards
and to support a per-subset rank-fidelity diagnostic for leaderboard
maintainers and benchmark designers. The accompanying paper presents the
analysis; this repository releases the underlying data and pipeline.

**Who created and funded it?**
The cleaned datasets were assembled by the authors of the accompanying paper
from publicly released upstream sources. No additional funding was used to
collect the underlying scores; we re-organize and clean what is already public.

## Composition

### Dataset 1: `data/score_matrix.csv`

**What does each instance represent?**
A single Large Language Model (LLM) submission to the Hugging Face Open LLM
Leaderboard v2, identified by its model identifier (e.g.,
`meta-llama/Meta-Llama-3-8B-Instruct`).

**How many instances?**
4,486 models after cleaning.

**What features are recorded?**
Seven columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `model` | string | n/a | Hugging Face model identifier (`org/name`) |
| `IFEval` | float | [0, 1] | Instruction-following evaluation score |
| `BBH` | float | [0, 1] | Big-Bench Hard accuracy |
| `MATH` | float | [0, 1] | MATH Level 5 accuracy |
| `GPQA` | float | [0, 1] | Graduate-level Google-Proof QA accuracy |
| `MUSR` | float | [0, 1] | Multistep Soft Reasoning accuracy |
| `MMLU_PRO` | float | [0, 1] | MMLU-Pro accuracy |

**Is there a label or target variable?**
There is no formal target. The "label" of interest in the paper is the row
mean across the six benchmark scores, which is how the upstream HF leaderboard
ranks models.

**What data is missing?**
Rows with any missing benchmark score are excluded. Models with metadata
flag `flagged = true` in the upstream snapshot are also excluded.
Per-model parameter count and submission type (chat / pretrained / fine-tune /
merge) are NOT included in `score_matrix.csv` itself; see `data/model_metadata.csv`
for a heuristic recovery from model names, with caveats.

### Dataset 2: `data/livebench_scores.csv`

**What does each instance represent?**
A single LLM evaluated on LiveBench (White et al., ICLR 2025).

**How many instances?**
49 models, taken from Table 3 of White et al.

**What features are recorded?**
Seven columns: `model`, `coding`, `data_analysis`, `instruction_following`,
`language`, `math`, `reasoning`. Scores are in [0, 1] (rescaled from the
original 0–100 reporting scale).

**What data is missing?**
None — the 49-model snapshot is taken verbatim from the published table.

## Collection

**How was the data collected?**

For `score_matrix.csv`: programmatically downloaded from the public
Hugging Face Hub dataset `open-llm-leaderboard/contents` (parquet format) on
April 2026. Cleaning pipeline:
1. Drop rows with any null benchmark score.
2. Drop rows where upstream `flagged = True`.
3. Deduplicate by model identifier (keep first).
4. Convert reported percentages to [0,1] floats.

For `livebench_scores.csv`: manually transcribed from Table 3 of White et al.
(arXiv:2406.19314). We chose transcription over scraping the live leaderboard
to make the LiveBench replication exactly reproducible from a single citation.
The 0–100 scores in the source are rescaled to [0,1] for consistency with
`score_matrix.csv`.

**Over what timeframe?**
Both snapshots reflect leaderboard state as of April 2026 (HF) and the ICLR
2025 publication window (LiveBench).

**What ethical-review processes were applied?**
None were required. The data is aggregate evaluation scores of publicly
released models on publicly released benchmarks; no human subjects, no PII,
no scraping of restricted data.

## Preprocessing / Cleaning

Documented above under "How was the data collected." All cleaning steps are
reproducible from the upstream source via the released code (data download
itself is not in the pipeline because it requires internet; the cleaned CSV
is bundled).

The original raw data (before cleaning) is preserved upstream at
`open-llm-leaderboard/contents` on Hugging Face Hub and at
arXiv:2406.19314 Table 3 for LiveBench.

## Uses

**What tasks could the dataset be used for?**
- Studying redundancy structure across benchmark suites.
- Power analyses for benchmark-design decisions.
- Per-subset rank-fidelity analysis for leaderboard maintainers.
- Construct validity research on LLM evaluation.
- Meta-evaluation of evaluation methodology.

**What tasks should the dataset NOT be used for?**
- Direct training of LLMs (the scores are evaluation outputs, not training
  data; using them as training signal could induce overfitting to
  benchmarks).
- Claims about real-world model deployment quality (these are benchmark
  scores, which are noisy proxies for capability).
- Construct-validity claims without consulting Kearns 2026 and Bean et al.
  2025 — statistical redundancy is not scientific redundancy.

## Distribution

**License:** CC-BY-4.0 for data, MIT for code. See `data/DATA_LICENSE.md`
and `LICENSE`.

**Hosting:** This repository (GitHub) and an anonymized mirror at
anonymous.4open.science.

**Attribution required:** Yes — please cite the accompanying paper and the
upstream sources (Hugging Face Open LLM Leaderboard maintainers; White et al.
2025).

## Maintenance

**Update policy:**
The `score_matrix.csv` snapshot is dated April 2026 and is not auto-refreshed.
Users wanting current data should re-run the upstream download (the script is
not bundled to avoid encouraging redistribution; see Hugging Face's
`open-llm-leaderboard/contents` documentation).

The redundancy structure can change as benchmarks saturate or as new
benchmarks are added to the leaderboard. We recommend at least quarterly
refresh for users applying the imputation framework to current data.
A practical heuristic: a benchmark whose novelty (1 − R^2_CV) drops below 0.1
on a current snapshot has become statistically near-redundant with the rest of
the suite and is a candidate for replacement.

**Versioning:**
The current release is the April 2026 snapshot. Future updates will be tagged
as releases on this repository.

## Limitations

1. **Six-column scope.** Both datasets use a six-benchmark structure. The
   redundancy analysis is exhaustive at d=6 but exhaustive search becomes
   intractable beyond d≈12 (see paper Appendix I for greedy/regularized
   variants).

2. **Population skew (HF v2).** ~39% fine-tunes and ~38% merges, both of which
   inherit performance profiles from base models. A population dominated by
   independent training runs could exhibit a different redundancy structure.

3. **Sample size (LiveBench).** n=49 yields wide bootstrap CIs on individual
   subset orderings. The global pattern (IF-inclusive subsets dominate)
   is robust; specific orderings between adjacent subsets are not.

4. **Snapshot in time.** Both datasets reflect leaderboard state at a fixed
   point. Benchmark saturation, contamination, and population shift can change
   the redundancy structure over time.

5. **Construct validity.** The datasets capture statistical redundancy among
   benchmark scores. They are silent on whether the benchmarks are
   scientifically redundant (i.e., whether they measure the same underlying
   construct). See Kearns 2026 (arXiv:2602.15532) and Bean et al. 2025
   (arXiv:2511.04703) for construct-validity treatments.

## Citation

A full methodology writeup and analysis are in preparation. Citation details
will be added here upon publication.

## Datasheet authors

The authors of the accompanying paper.
