"""
Two-stage hybrid pipeline (Section 7 of paper):
  Stage 1: run best 3-of-6 subset on ALL submissions, impute remaining 3
  Stage 2: run all 6 benchmarks on the imputed top-K candidates
  Final ranking: from the stage-2 evaluations.

Cost saved = (1 - (k_obs * n_full + (d - k_obs) * topK) / (d * n_full))
For k_obs=3, d=6, n=4486:
  topK=100: cost = (3*4486 + 3*100) / (6*4486) = 0.5111 -> save 48.9%
  topK=50:  cost = (3*4486 + 3*50)  / (6*4486) = 0.5056 -> save 49.4%
  topK=200: cost = (3*4486 + 3*200) / (6*4486) = 0.5223 -> save 47.8%

Recall at topK = |true_topK & imputed_topK| / topK.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

df = pd.read_csv("../data/score_matrix.csv")
score_cols = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
X = df[score_cols].values
n, d = X.shape

scaler = StandardScaler()
Xz = scaler.fit_transform(X)
true_avg = X.mean(axis=1)

# best 3-of-6
observed = (0, 1, 2)  # IFEval, BBH, MATH
held_out = [3, 4, 5]

# CV-imputed predictions (same as hf_imputation.py)
pred_z = Xz.copy()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr, te in kf.split(Xz):
    for j in held_out:
        m = LinearRegression().fit(Xz[tr][:, list(observed)], Xz[tr, j])
        pred_z[te, j] = m.predict(Xz[te][:, list(observed)])

imp_avg = scaler.inverse_transform(pred_z).mean(axis=1)

# Now: for each topK candidate-pool size, what fraction of true top-10 do we recover?
def recall_at(true_avg, imp_avg, target_k, candidate_k):
    """If we evaluate the imputed top-`candidate_k` and pick the true top-`target_k`
    from within them (using their REAL averages), how many of the true top-target_k do we get?
    Since we're running all 6 benchmarks in stage 2, the candidates have their true scores
    so this reduces to: |(true_top_target_k) & (imputed_top_candidate_k)| / target_k.
    """
    true_top = set(np.argsort(-true_avg)[:target_k])
    imp_top = set(np.argsort(-imp_avg)[:candidate_k])
    return len(true_top & imp_top) / target_k

# Sweep a fine-grained range of pool sizes for the full Pareto frontier.
# Save to CSV so reviewers can plot/inspect the actual curve.
pool_sizes = list(range(10, 50, 5)) + list(range(50, 200, 10)) + list(range(200, 1001, 50))
rows = []
for cand_k in pool_sizes:
    cost_saved = 1 - (3 * n + 3 * cand_k) / (6 * n)
    rows.append({
        "candidate_pool_size": cand_k,
        "cost_saved_frac": cost_saved,
        "recall_at_10": recall_at(true_avg, imp_avg, 10, cand_k),
        "recall_at_50": recall_at(true_avg, imp_avg, 50, cand_k),
        "recall_at_100": recall_at(true_avg, imp_avg, 100, cand_k),
        "recall_at_200": recall_at(true_avg, imp_avg, 200, cand_k),
    })
pareto = pd.DataFrame(rows)
pareto.to_csv("../data/hf_two_stage_pareto.csv", index=False)
print(f"Pareto frontier saved to hf_two_stage_pareto.csv ({len(pareto)} pool sizes)\n")

print("=== Two-stage hybrid recall vs candidate-pool size (selected points) ===")
print(f"{'pool size':>10}  {'cost saved':>12}  {'recall@10':>11}  {'recall@50':>11}  {'recall@100':>12}")
for cand_k in [25, 50, 100, 150, 200, 300, 500]:
    cost_saved = 1 - (3 * n + 3 * cand_k) / (6 * n)
    r10 = recall_at(true_avg, imp_avg, 10, cand_k)
    r50 = recall_at(true_avg, imp_avg, 50, cand_k)
    r100 = recall_at(true_avg, imp_avg, 100, cand_k)
    print(f"{cand_k:>10d}  {cost_saved*100:>11.1f}%  {r10:>11.2f}  {r50:>11.2f}  {r100:>12.2f}")

# Headline number: top-100 recall at top-100 candidate pool
# (paper says 49% cost saved, 83% recall)
cand_k = 100
cost = (3 * n + 3 * cand_k) / (6 * n)
saved = 1 - cost
recall100 = recall_at(true_avg, imp_avg, 100, cand_k)
print(f"\nheadline: candidates=100, cost saved={saved*100:.1f}%, recall@100={recall100:.2f}")
assert abs(saved - 0.489) < 0.01
assert abs(recall100 - 0.83) < 0.05
print("sanity checks passed")
