"""
HF: linear regression vs gradient boosting on the best 3-of-6 subset.
The paper claims linear gives tau=0.943 with mean R^2=0.728,
GBM gives tau=0.948 with mean R^2=0.793. Difference of +0.005 in tau is
negligible for ranking - this script verifies that.

GBM hyperparams from paper: 200 trees, max_depth=3, default lr.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau

np.random.seed(42)

df = pd.read_csv("../data/score_matrix.csv")
score_cols = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
X = df[score_cols].values
n, d = X.shape

scaler = StandardScaler()
Xz = scaler.fit_transform(X)
true_avg = X.mean(axis=1)

# Best 3-of-6 from imputation script
observed = (0, 1, 2)  # IFEval, BBH, MATH
held_out = [3, 4, 5]  # GPQA, MUSR, MMLU_PRO

def cv_predict(model_fn, observed, held_out, Xz, n_splits=5):
    """Returns predictions in standardized space for held-out benches, plus per-bench R^2."""
    pred = Xz.copy()
    r2_per_bench = {j: [] for j in held_out}
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, te in kf.split(Xz):
        for j in held_out:
            Xo_tr = Xz[tr][:, list(observed)]
            Xo_te = Xz[te][:, list(observed)]
            y_tr = Xz[tr, j]
            y_te = Xz[te, j]
            m = model_fn().fit(Xo_tr, y_tr)
            yhat = m.predict(Xo_te)
            pred[te, j] = yhat
            ss_res = np.sum((y_te - yhat)**2)
            ss_tot = np.sum((y_te - y_te.mean())**2)
            r2_per_bench[j].append(1 - ss_res/max(ss_tot, 1e-12))
    return pred, {j: np.mean(rs) for j, rs in r2_per_bench.items()}

# === Linear ===
pred_lin, r2_lin = cv_predict(LinearRegression, observed, held_out, Xz)
imp_lin = scaler.inverse_transform(pred_lin).mean(axis=1)
tau_lin, _ = kendalltau(true_avg, imp_lin)
print(f"=== Linear regression ===")
print(f"  tau    = {tau_lin:.4f}")
for j in held_out:
    print(f"  R^2 [{score_cols[j]:8s}] = {r2_lin[j]:.3f}")
mean_r2_lin = np.mean(list(r2_lin.values()))
print(f"  mean R^2 = {mean_r2_lin:.3f}")

# === Gradient boosting ===
def gbm():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)

pred_gbm, r2_gbm = cv_predict(gbm, observed, held_out, Xz)
imp_gbm = scaler.inverse_transform(pred_gbm).mean(axis=1)
tau_gbm, _ = kendalltau(true_avg, imp_gbm)
print(f"\n=== Gradient boosting (200 trees, depth 3) ===")
print(f"  tau    = {tau_gbm:.4f}")
for j in held_out:
    print(f"  R^2 [{score_cols[j]:8s}] = {r2_gbm[j]:.3f}")
mean_r2_gbm = np.mean(list(r2_gbm.values()))
print(f"  mean R^2 = {mean_r2_gbm:.3f}")

print(f"\n=== Comparison ===")
print(f"  delta tau = {tau_gbm - tau_lin:+.4f}  (paper claim: +0.005)")
print(f"  delta R^2 = {mean_r2_gbm - mean_r2_lin:+.4f}")
print(f"  MUSR R^2: linear={r2_lin[4]:.3f} -> GBM={r2_gbm[4]:.3f}  (paper: 0.516 -> 0.631)")

# Sanity
assert abs(tau_lin - 0.943) < 0.01
assert abs(tau_gbm - 0.948) < 0.01
print("\nsanity checks passed")
