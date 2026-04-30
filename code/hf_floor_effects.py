"""
Q1 from reviewer: is the monotone decrease in effective rank with model size
(reported in Appendix E) primarily an artifact of floor effects in small models
on hard benchmarks (MATH, GPQA)?

Test: re-run the size-stratified PCA after restricting to models above a minimum
performance threshold. If the monotone decrease persists, it's not a floor-effect
artifact. If the decrease shrinks or reverses, the reviewer is right.

We test thresholds at mean overall score > 0.20, 0.25, 0.30, 0.35.
"""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

df = pd.read_csv("../data/score_matrix.csv")
cols = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
df["mean_score"] = df[cols].mean(axis=1)

# Extract param count from model name. Handles "Llama-2-7B", "Mistral-7b", "0.5B", etc.
def extract_params(name):
    m = re.search(r'(\d+\.?\d*)\s*[Bb](?![a-z])', name)
    return float(m.group(1)) if m else None

df["params"] = df["model"].apply(extract_params)
n_with_size = df["params"].notna().sum()
print(f"Models with parseable size: {n_with_size}/{len(df)}")
df = df[df["params"].notna()].copy()
df["size_bin"] = pd.cut(df["params"], bins=[0, 7.5, 30, 1000],
                        labels=["small", "medium", "large"])

def effective_rank(X):
    Xz = StandardScaler().fit_transform(X)
    _, S, _ = np.linalg.svd(Xz, full_matrices=False)
    eigs = S**2 / (X.shape[0] - 1)
    p = eigs / eigs.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H)), float(p[0])

# Run effective rank per size bin at each threshold
print("\n=== Effective rank by size bin, before/after floor-effect filter ===")
print(f"{'threshold':>10}  {'small':>20}  {'medium':>20}  {'large':>20}  {'monotone?':>10}")

for thresh in [0.0, 0.20, 0.25, 0.30, 0.35]:
    df_t = df[df["mean_score"] > thresh].copy()
    bins_data = {}
    for bin_name in ["small", "medium", "large"]:
        sub = df_t[df_t["size_bin"] == bin_name][cols].values
        if len(sub) < 30:
            bins_data[bin_name] = (None, len(sub))
            continue
        er, pc1 = effective_rank(sub)
        bins_data[bin_name] = (er, len(sub))

    parts = [f"thresh > {thresh:.2f}".rjust(10)]
    ranks = []
    for bn in ["small", "medium", "large"]:
        er, n = bins_data[bn]
        if er is None:
            parts.append(f"n={n} (too few)".rjust(20))
            ranks.append(None)
        else:
            parts.append(f"{er:.3f} (n={n})".rjust(20))
            ranks.append(er)
    # Monotone decrease check
    valid = [r for r in ranks if r is not None]
    if len(valid) == 3:
        is_monotone = valid[0] > valid[1] > valid[2]
        parts.append("YES" if is_monotone else "no".rjust(10))
    else:
        parts.append("-".rjust(10))
    print("  ".join(parts))

# Headline: does the decrease hold after dropping the bottom 30%?
print("\n=== Interpretation ===")
df_t = df[df["mean_score"] > 0.30].copy()
bin_results = []
for bin_name in ["small", "medium", "large"]:
    sub = df_t[df_t["size_bin"] == bin_name][cols].values
    er, _ = effective_rank(sub)
    bin_results.append((bin_name, er, len(sub)))

print("After dropping models with mean overall < 0.30 (~28% of population):")
for name, er, n in bin_results:
    print(f"  {name:8s} (n={n}): eff_rank = {er:.3f}")
delta = bin_results[0][1] - bin_results[-1][1]
print(f"\nsmall - large delta: {delta:+.3f}")
print(f"  (paper reports 3.05 - 2.87 = +0.18 on full data)")
print(f"  (if delta stays positive after filter, monotone decrease is NOT a floor-effect artifact)")
