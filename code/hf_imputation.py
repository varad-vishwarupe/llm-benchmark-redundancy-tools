"""
HF Open LLM v2: cross-benchmark score imputation.
Exhaustively evaluates all C(6, k) subsets at k=1..5 and reports
Kendall tau, Spearman rho, top-10/top-100 overlap, mean R^2.

Reproduces Table 2 (k=3) and the k-sweep in the paper.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr

np.random.seed(42)

df = pd.read_csv("../data/score_matrix.csv")
score_cols = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
X = df[score_cols].values
n, d = X.shape
print(f"matrix: {n} x {d}")

scaler = StandardScaler()
Xz = scaler.fit_transform(X)

# True ranking is just the rank order of the row mean across all 6 benchmarks.
# This matches how HF Open LLM v2 produces its leaderboard (mean of normalized scores).
true_avg = X.mean(axis=1)
true_rank = pd.Series(true_avg).rank(method="average")

def imputed_ranking(observed_idx, X_full, Xz_full, scaler):
    """5-fold CV. For each held-out bench, train linear reg on observed -> bench
    using the train fold, predict on test fold. Then re-rank by mean of
    (observed real scores + imputed scores) to mimic what a maintainer would do.
    """
    n, d = X_full.shape
    held_out = [j for j in range(d) if j not in observed_idx]
    pred_z = Xz_full.copy()  # predictions in standardized space
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2s = {j: [] for j in held_out}
    for tr, te in kf.split(Xz_full):
        for j in held_out:
            Xo_tr = Xz_full[tr][:, list(observed_idx)]
            Xo_te = Xz_full[te][:, list(observed_idx)]
            y_tr = Xz_full[tr, j]
            y_te = Xz_full[te, j]
            m = LinearRegression().fit(Xo_tr, y_tr)
            yhat = m.predict(Xo_te)
            pred_z[te, j] = yhat
            ss_res = np.sum((y_te - yhat)**2)
            ss_tot = np.sum((y_te - y_te.mean())**2)
            fold_r2s[j].append(1 - ss_res/max(ss_tot, 1e-12))
    # Bring back to raw scale and average
    pred_full = scaler.inverse_transform(pred_z)
    avg_imp = pred_full.mean(axis=1)
    mean_r2 = np.mean([np.mean(fold_r2s[j]) for j in held_out])
    return avg_imp, mean_r2

def topk_overlap(true_avg, imp_avg, k):
    true_top = set(np.argsort(-true_avg)[:k])
    imp_top = set(np.argsort(-imp_avg)[:k])
    return len(true_top & imp_top) / k

# === Exhaustive subset search ===
results = []
for k in range(1, 6):
    for subset in combinations(range(d), k):
        names = [score_cols[i] for i in subset]
        imp_avg, mean_r2 = imputed_ranking(subset, X, Xz, scaler)
        tau, _ = kendalltau(true_avg, imp_avg)
        rho, _ = spearmanr(true_avg, imp_avg)
        t10 = topk_overlap(true_avg, imp_avg, 10)
        t100 = topk_overlap(true_avg, imp_avg, 100)
        results.append({
            "k": k,
            "subset": " + ".join(names),
            "tau": tau,
            "rho": rho,
            "mean_r2": mean_r2,
            "top10": t10,
            "top100": t100,
        })

res = pd.DataFrame(results).sort_values(["k", "tau"], ascending=[True, False])
res.to_csv("../data/hf_imputation_full.csv", index=False)
print(f"\nfull table written to hf_imputation_full.csv ({len(res)} rows)")

# === Highlight: k=3 best and worst ===
k3 = res[res["k"] == 3].sort_values("tau", ascending=False)
print("\n=== k=3 top 5 ===")
print(k3.head(5)[["subset", "tau", "mean_r2", "top10", "top100"]].to_string(index=False))
print("\n=== k=3 bottom 5 ===")
print(k3.tail(5)[["subset", "tau", "mean_r2", "top10", "top100"]].to_string(index=False))

best = k3.iloc[0]
worst = k3.iloc[-1]
print(f"\nbest 3-of-6: {best['subset']}  tau={best['tau']:.3f}  mean_r2={best['mean_r2']:.3f}  top10={best['top10']:.2f}")
print(f"worst 3-of-6: {worst['subset']}  tau={worst['tau']:.3f}  mean_r2={worst['mean_r2']:.3f}  top10={worst['top10']:.2f}")
print(f"mean tau across 20 subsets: {k3['tau'].mean():.3f}")
print(f"mean R^2 across 20 subsets: {k3['mean_r2'].mean():.3f}")

# IFEval-in-subset diagnostic. The paper claims IFEval is in all 5 top subsets
# and none of the 5 bottom. Lets verfiy.
ifeval_in = k3["subset"].str.contains("IFEval")
print(f"\nIFEval in top-5 subsets: {ifeval_in.head(5).sum()}/5")
print(f"IFEval in bottom-5 subsets: {ifeval_in.tail(5).sum()}/5")

# Sanity vs paper
assert abs(best["tau"] - 0.943) < 0.01, f"best tau should be ~0.943, got {best['tau']}"
assert abs(worst["tau"] - 0.757) < 0.01, f"worst tau should be ~0.757, got {worst['tau']}"
print("\nsanity checks passed")
