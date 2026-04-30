"""
Bootstrap CIs on Kendall tau for LiveBench best and worst 3-of-6 subsets.
We resample models with replacement, refit imputation, recompute tau.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau

np.random.seed(42)

df = pd.read_csv('../data/livebench_scores.csv')
score_cols = ['coding', 'data_analysis', 'instruction_following', 'language', 'math', 'reasoning']
X_full = df[score_cols].values
n_full, d = X_full.shape

def impute_tau_for_subset(X, observed_idx, seed=42):
    """Compute Kendall tau between true ranking and imputed ranking for a given subset."""
    n, d = X.shape
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    held_out = [j for j in range(d) if j not in observed_idx]

    pred_z = Xz.copy()
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kf.split(Xz):
        for j in held_out:
            X_in_tr = Xz[tr][:, list(observed_idx)]
            y_tr = Xz[tr, j]
            X_in_te = Xz[te][:, list(observed_idx)]
            m = LinearRegression().fit(X_in_tr, y_tr)
            pred_z[te, j] = m.predict(X_in_te)

    pred_full = scaler.inverse_transform(pred_z)
    true_avg = X.mean(axis=1)
    pred_avg = pred_full.mean(axis=1)
    tau, _ = kendalltau(true_avg, pred_avg)
    return tau

# Test subsets: best 3-of-6 (Coding+DA+IF) and worst 3-of-6 (Coding+Math+Reasoning)
best_subset = (0, 1, 2)   # coding, data_analysis, instruction_following
worst_subset = (0, 4, 5)  # coding, math, reasoning

# Verify on full data
print(f"Full data tau (best subset):  {impute_tau_for_subset(X_full, best_subset):.4f}")
print(f"Full data tau (worst subset): {impute_tau_for_subset(X_full, worst_subset):.4f}")

# Bootstrap: resample 49 models with replacement, recompute tau
N_BOOT = 1000
boot_best = []
boot_worst = []

for b in range(N_BOOT):
    idx = np.random.choice(n_full, size=n_full, replace=True)
    X_boot = X_full[idx]
    # Drop bootstrap samples that have zero variance in any column
    if (X_boot.std(axis=0) < 1e-6).any():
        continue
    try:
        boot_best.append(impute_tau_for_subset(X_boot, best_subset, seed=b))
        boot_worst.append(impute_tau_for_subset(X_boot, worst_subset, seed=b))
    except Exception:
        continue

boot_best = np.array(boot_best)
boot_worst = np.array(boot_worst)

print(f"\nBootstrap CI ({len(boot_best)} valid resamples):")
print(f"  Best 3-of-6 (Coding+DA+IF):       tau = {boot_best.mean():.3f}  [95% CI: {np.percentile(boot_best, 2.5):.3f}, {np.percentile(boot_best, 97.5):.3f}]")
print(f"  Worst 3-of-6 (Coding+Math+Reas):  tau = {boot_worst.mean():.3f}  [95% CI: {np.percentile(boot_worst, 2.5):.3f}, {np.percentile(boot_worst, 97.5):.3f}]")

# Also report on the all-20-subsets distribution
print(f"\nBest tau across bootstraps: {boot_best.max():.3f}")
print(f"Std on best: {boot_best.std():.3f}")
print(f"Std on worst: {boot_worst.std():.3f}")

# CI on the *difference* best - worst
diff = boot_best - boot_worst
print(f"\nBest - Worst difference:")
print(f"  Mean: {diff.mean():.3f}")
print(f"  95% CI: [{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]")
print(f"  All positive: {(diff > 0).all()}")  # if True, ordering is robust

# Save raw values for reference
pd.DataFrame({
    'best_subset_tau': boot_best,
    'worst_subset_tau': boot_worst,
}).to_csv('../data/livebench_bootstrap_tau.csv', index=False)
print(f"\nSaved bootstrap samples to livebench_bootstrap_tau.csv")
