"""
LiveBench replication: PCA + imputation analysis on the 49-model x 6-category matrix.
Mirrors the HF analysis exactly so we can compare apples-to-apples.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, spearmanr

np.random.seed(42)

# Load
df = pd.read_csv('../data/livebench_scores.csv')
score_cols = ['coding', 'data_analysis', 'instruction_following', 'language', 'math', 'reasoning']
X = df[score_cols].values  # 49 x 6
n, d = X.shape
print(f"Matrix: {n} models x {d} categories")

# === PCA ===
scaler = StandardScaler()
Xz = scaler.fit_transform(X)
U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
eigenvalues = S**2 / (n - 1)
total_var = eigenvalues.sum()
var_explained = eigenvalues / total_var
cum_var = np.cumsum(var_explained)

print(f"\n=== PCA ===")
print(f"Variance explained per PC: {var_explained.round(4)}")
print(f"Cumulative: {cum_var.round(4)}")

# Effective rank (entropy-based)
p = var_explained
H = -np.sum(p * np.log(p + 1e-12))
eff_rank = np.exp(H)
print(f"Spectral effective rank: {eff_rank:.3f} out of {d}")

# Loadings = V (rows = PCs, cols = benchmarks)
print(f"\nPC1 loadings: {dict(zip(score_cols, Vt[0].round(3)))}")
print(f"PC2 loadings: {dict(zip(score_cols, Vt[1].round(3)))}")

# === Pairwise correlations ===
corr = np.corrcoef(X.T)
print(f"\n=== Pairwise correlations ===")
corr_df = pd.DataFrame(corr, index=score_cols, columns=score_cols)
print(corr_df.round(3))
off_diag = corr[np.triu_indices(d, k=1)]
print(f"\nOff-diagonal: min={off_diag.min():.3f}, max={off_diag.max():.3f}, mean={off_diag.mean():.3f}")

# === Per-benchmark novelty (1 - R^2 from regressing on others) ===
print(f"\n=== Per-benchmark novelty (1 - R^2_CV) ===")
novelty = {}
for j, name in enumerate(score_cols):
    others = [k for k in range(d) if k != j]
    X_obs = Xz[:, others]
    y = Xz[:, j]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2s = []
    for tr, te in kf.split(X_obs):
        m = LinearRegression().fit(X_obs[tr], y[tr])
        pred = m.predict(X_obs[te])
        ss_res = ((y[te] - pred)**2).sum()
        ss_tot = ((y[te] - y[te].mean())**2).sum()
        r2s.append(1 - ss_res/ss_tot)
    r2 = np.mean(r2s)
    novelty[name] = (r2, 1 - r2)
    print(f"  {name:25s}  R2={r2:.3f}  novelty={1-r2:.3f}")

# === Imputation: all subsets at all k ===
print(f"\n=== Imputation: all subsets ===")
true_avg = X.mean(axis=1)  # in original scale
true_rank = np.argsort(np.argsort(-true_avg))  # rank 0 = best

results = []

def impute_and_score(observed_idx):
    """Given observed indices S, predict the rest and rerank."""
    observed_set = set(observed_idx)
    held_out = [j for j in range(d) if j not in observed_set]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pred_full = X.copy()  # initialize with truth, will overwrite held-out
    pred_z = np.zeros_like(Xz)
    pred_z[:, list(observed_idx)] = Xz[:, list(observed_idx)]

    fold_r2 = {j: [] for j in held_out}
    for tr, te in kf.split(Xz):
        for j in held_out:
            X_in_tr = Xz[tr][:, list(observed_idx)]
            y_tr = Xz[tr, j]
            X_in_te = Xz[te][:, list(observed_idx)]
            y_te = Xz[te, j]
            m = LinearRegression().fit(X_in_tr, y_tr)
            yhat = m.predict(X_in_te)
            pred_z[te, j] = yhat
            ss_res = ((y_te - yhat)**2).sum()
            ss_tot = ((y_te - y_te.mean())**2).sum() + 1e-12
            fold_r2[j].append(1 - ss_res/ss_tot)

    # Reverse z-score for held-out columns
    pred_full = scaler.inverse_transform(pred_z)
    pred_avg = pred_full.mean(axis=1)

    tau, _ = kendalltau(true_avg, pred_avg)
    rho, _ = spearmanr(true_avg, pred_avg)

    # top-k overlap
    pred_rank = np.argsort(np.argsort(-pred_avg))
    def topk_overlap(k):
        true_top = set(np.argsort(-true_avg)[:k])
        pred_top = set(np.argsort(-pred_avg)[:k])
        return len(true_top & pred_top) / k
    top10 = topk_overlap(min(10, n))
    top25 = topk_overlap(min(25, n))

    mean_r2 = np.mean([np.mean(fold_r2[j]) for j in held_out])

    return {
        'tau': tau,
        'rho': rho,
        'top10': top10,
        'top25': top25,
        'mean_r2_imputed': mean_r2,
    }

for k in range(1, d):
    print(f"\n--- k={k} ---")
    subset_results = []
    for combo in combinations(range(d), k):
        names = [score_cols[i] for i in combo]
        res = impute_and_score(combo)
        res['k'] = k
        res['observed'] = ', '.join(names)
        res['observed_idx'] = combo
        subset_results.append(res)
        results.append(res)

    sr = pd.DataFrame(subset_results).sort_values('tau', ascending=False)
    print(f"  best:  {sr.iloc[0]['observed']:50s}  tau={sr.iloc[0]['tau']:.3f}  R2={sr.iloc[0]['mean_r2_imputed']:.3f}  top10={sr.iloc[0]['top10']:.2f}")
    print(f"  worst: {sr.iloc[-1]['observed']:50s}  tau={sr.iloc[-1]['tau']:.3f}  R2={sr.iloc[-1]['mean_r2_imputed']:.3f}  top10={sr.iloc[-1]['top10']:.2f}")
    print(f"  mean tau across {len(sr)} subsets: {sr['tau'].mean():.3f}  (range {sr['tau'].min():.3f}-{sr['tau'].max():.3f})")

# Save full results
res_df = pd.DataFrame(results)
res_df.to_csv('../data/livebench_imputation_full.csv', index=False)
print(f"\nFull results saved ({len(res_df)} subsets)")

# Print top/bottom 5 at k=3
print(f"\n=== k=3, top 5 and bottom 5 ===")
k3 = res_df[res_df['k']==3].sort_values('tau', ascending=False)
for _, row in k3.head(5).iterrows():
    print(f"  TOP    {row['observed']:50s}  tau={row['tau']:.3f}  R2={row['mean_r2_imputed']:.3f}  top10={row['top10']:.2f}")
print()
for _, row in k3.tail(5).iterrows():
    print(f"  BOT    {row['observed']:50s}  tau={row['tau']:.3f}  R2={row['mean_r2_imputed']:.3f}  top10={row['top10']:.2f}")
