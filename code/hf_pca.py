"""
HF Open LLM v2: PCA + per-benchmark novelty diagnostic.
Reproduces the numbers in Section 4 of the paper.

Inputs:  score_matrix.csv (4486 x 7, model + 6 benchmarks)
Outputs: prints PC variance, effective rank, loadings, pairwise correls,
         per-benchmark novelty (1 - R^2_CV from regressing on others).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Load. Six benchmarks in this order in the paper.
df = pd.read_csv("../data/score_matrix.csv")
score_cols = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
X = df[score_cols].values
n, d = X.shape
print(f"score matrix: {n} models x {d} benchmarks")

# Standardize before SVD - we want correlation-PCA not covariance-PCA
# (otherwise BBH dominates becuase its raw spread is larger)
scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# === PCA via SVD on the standardized matrix ===
U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
eigs = S**2 / (n - 1)
var_explained = eigs / eigs.sum()
cum_var = np.cumsum(var_explained)

print("\n=== PCA ===")
for i, (v, c) in enumerate(zip(var_explained, cum_var), 1):
    print(f"  PC{i}: {v*100:.1f}%   cumulative {c*100:.1f}%")

# Spectral effective rank = exp(entropy of normalised eigvals)
# Roy & Vetterli 2007. Higher means flatter spectrum, lower means more dominant axes.
H = -np.sum(var_explained * np.log(var_explained + 1e-12))
eff_rank = np.exp(H)
print(f"  spectral effective rank: {eff_rank:.3f} out of {d}")

print("\nPC1 loadings (should all be positive - the 'general capability' axis):")
for col, l in zip(score_cols, Vt[0]):
    print(f"  {col:10s} {l:+.3f}")

print("\nPC2 loadings (the IFEval-vs-others contrast):")
for col, l in zip(score_cols, Vt[1]):
    print(f"  {col:10s} {l:+.3f}")

# === Pairwise correlations (Section 4 of paper) ===
print("\n=== Pairwise correlations (raw, not standardized) ===")
corr = np.corrcoef(X.T)
print(pd.DataFrame(corr, index=score_cols, columns=score_cols).round(3))
off = corr[np.triu_indices(d, k=1)]
print(f"\noff-diag: min={off.min():.3f}, max={off.max():.3f}, mean={off.mean():.3f}")

# === Per-benchmark novelty: 1 - R^2_CV from regressing each benchmark on the others ===
# This is the key "per-benchmark novelty" diagnostic in Table 1 of the paper.
# Low novelty = predictable from other benchmarks = redundant.
# IFEval should come out highest (~0.50), MMLU-Pro lowest (~0.06).
print("\n=== Per-benchmark novelty (1 - R^2_CV, 5-fold) ===")
novelty = {}
for j, name in enumerate(score_cols):
    others = [k for k in range(d) if k != j]
    X_obs = Xz[:, others]
    y = Xz[:, j]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(n)
    for tr, te in kf.split(X_obs):
        m = LinearRegression().fit(X_obs[tr], y[tr])
        preds[te] = m.predict(X_obs[te])
    ss_res = np.sum((y - preds)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    nov = 1 - r2
    novelty[name] = (r2, nov)
    print(f"  {name:10s} R^2={r2:.3f}   novelty={nov:.3f}")

# Quick sanity check vs paper claims
assert abs(var_explained[0] - 0.731) < 0.005, "PC1 variance should be ~73.1%"
assert abs(eff_rank - 2.55) < 0.05, "effective rank should be ~2.55"
assert abs(novelty["IFEval"][1] - 0.503) < 0.02, "IFEval novelty should be ~0.503"
print("\nsanity checks passed - matches paper numbers")
