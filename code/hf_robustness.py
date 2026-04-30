"""
HF robustness checks (Section 6.7 of paper):
  1. subsample stability: PCA on bootstrap samples at n=100,250,500,1000,2000,4000.
     Effective rank should converge ~2.55 by n>=100.
  2. type stratification: PCA per training-paradigm category (chat, pretrained,
     fine-tuned, merge). Effective ranks stay in 2.57-2.72 band.
  3. size stratification: PCA per parameter-count bin. Effective rank decreses
     monotonically with model size (3.05 -> 2.87).

The 'model' column in score_matrix.csv has form 'org/name' but doesnt have
metadata for type or size. So this script can do (1) directly and stubs out
(2)/(3) - we reproduce them in a sep script when type/size metadata is loaded.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

df = pd.read_csv("../data/score_matrix.csv")
score_cols = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
X_full = df[score_cols].values
n_full, d = X_full.shape

def effective_rank(X):
    """Spectral effective rank of standardized X. exp(entropy of eigval distribution)."""
    Xz = StandardScaler().fit_transform(X)
    _, S, _ = np.linalg.svd(Xz, full_matrices=False)
    eigs = S**2 / (X.shape[0] - 1)
    p = eigs / eigs.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H)), float(p[0])  # also return PC1 share

# === Subsample stability ===
print("=== Subsample stability ===")
print(f"{'n':>5}  {'eff_rank (mean+/-std)':>25}  {'pc1 (mean+/-std)':>25}")
for n in [100, 250, 500, 1000, 2000, 4000]:
    if n > n_full:
        continue
    ranks, pc1s = [], []
    for seed in range(30):
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_full, size=n, replace=False)
        X_s = X_full[idx]
        er, pc1 = effective_rank(X_s)
        ranks.append(er)
        pc1s.append(pc1)
    print(f"{n:>5}  {np.mean(ranks):.3f} +/- {np.std(ranks):.3f}        {np.mean(pc1s):.3f} +/- {np.std(pc1s):.3f}")

# Full data
er_full, pc1_full = effective_rank(X_full)
print(f"\nfull (n={n_full}): eff_rank = {er_full:.3f}, PC1 share = {pc1_full:.3f}")

# Quick sanity vs paper
assert abs(er_full - 2.55) < 0.05
print("\nsanity check passed - eff rank ~2.55 on full data")

# === Note re type / size stratification ===
# These need model metadata (chat/pretrained/finetune/merge labels and
# parameter counts). The labels come from the upstream HF leaderboard's
# 'type' and 'params_billions' columns of the same parquet we cached.
# In the released supplement we ship score_matrix.csv with just the scores
# to keep the data minimal; the stratification analysis lives in the paper
# but isnt re-computed here. Numbers reported in paper:
#   chat       n=693    eff_rank=2.72
#   pretrained n=324    eff_rank=2.58
#   finetune   n=1750   eff_rank=2.57
#   merge      n=1712   eff_rank=2.70
# Size stratification (paper Table 7):
#   <7.5B      n=1573   eff_rank=3.05
#   7.5-30B    n=2636   eff_rank=2.89
#   >30B       n=277    eff_rank=2.87
print("\n(stratified analyses by type/size require upstream metadata - see paper Table 7)")
