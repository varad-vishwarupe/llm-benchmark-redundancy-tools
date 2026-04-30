"""
Generate the 4 main figures from Section 4 and 6 of the paper.
Outputs PDF + PNG into ./figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

os.makedirs("../figures", exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10

df_hf = pd.read_csv("../data/score_matrix.csv")
hf_cols = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU_PRO"]
X_hf = df_hf[hf_cols].values
n_hf = len(X_hf)

df_lb = pd.read_csv("../data/livebench_scores.csv")
lb_cols = ["coding", "data_analysis", "instruction_following", "language", "math", "reasoning"]
X_lb = df_lb[lb_cols].values
n_lb = len(X_lb)

# ---- Figure 1: HF scree plot ----
Xz = StandardScaler().fit_transform(X_hf)
_, S, _ = np.linalg.svd(Xz, full_matrices=False)
eigs = S**2 / (n_hf - 1)
var_hf = eigs / eigs.sum()

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.bar(range(1, 7), var_hf * 100, color="#3b6ea8", alpha=0.85, edgecolor="black", linewidth=0.6)
for i, v in enumerate(var_hf):
    ax.text(i + 1, v*100 + 1, f"{v*100:.1f}%", ha="center", fontsize=9)
ax.set_xlabel("Principal component")
ax.set_ylabel("Variance explained (%)")
ax.set_xticks(range(1, 7))
ax.set_ylim(0, 80)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("../figures/scree_plot.pdf", bbox_inches="tight")
plt.savefig("../figures/scree_plot.png", dpi=150, bbox_inches="tight")
plt.close()
print("../figures/scree_plot.{pdf,png}")

# ---- Figure 2: HF correlation heatmap ----
corr = np.corrcoef(X_hf.T)
fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(hf_cols, rotation=45, ha="right")
ax.set_yticklabels(hf_cols)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(corr[i,j]) > 0.7 else "black", fontsize=9)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("../figures/correlation_heatmap.pdf", bbox_inches="tight")
plt.savefig("../figures/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("../figures/correlation_heatmap.{pdf,png}")

# ---- Figure 3: All k=3 subsets ranked by tau ----
# requires hf_imputation_full.csv to exist; if not, skip
if os.path.exists("../data/hf_imputation_full.csv"):
    res = pd.read_csv("../data/hf_imputation_full.csv")
    k3 = res[res["k"] == 3].sort_values("tau", ascending=True).reset_index(drop=True)
    has_ifeval = k3["subset"].str.contains("IFEval")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#c0392b" if not has else "#2c7a4d" for has in has_ifeval]
    ax.barh(range(len(k3)), k3["tau"], color=colors, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(k3)))
    ax.set_yticklabels(k3["subset"], fontsize=8)
    ax.set_xlabel(r"Kendall $\tau$ vs full-evaluation ranking")
    ax.set_xlim(0.7, 1.0)
    ax.axvline(0.943, color="black", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.text(0.943, -0.5, "best", ha="center", fontsize=8)
    # Legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#2c7a4d", edgecolor="black", label="includes IFEval"),
           Patch(facecolor="#c0392b", edgecolor="black", label="excludes IFEval")]
    ax.legend(handles=leg, loc="lower right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("../figures/imputation_subsets.pdf", bbox_inches="tight")
    plt.savefig("../figures/imputation_subsets.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("../figures/imputation_subsets.{pdf,png}")
else:
    print("(skip imputation_subsets.png - run hf_imputation.py first)")

# ---- Figure 4: cross-leaderboard comparison (HF vs LiveBench) ----
Xz_lb = StandardScaler().fit_transform(X_lb)
_, S_lb, _ = np.linalg.svd(Xz_lb, full_matrices=False)
eigs_lb = S_lb**2 / (n_lb - 1)
var_lb = eigs_lb / eigs_lb.sum()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))

# Left: side-by-side scree
ax = axes[0]
w = 0.4
x = np.arange(1, 7)
ax.bar(x - w/2, var_hf * 100, w, label=f"HF Open LLM (n={n_hf})", color="#3b6ea8")
ax.bar(x + w/2, var_lb * 100, w, label=f"LiveBench (n={n_lb})", color="#d35400")
ax.set_xticks(x)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance explained (%)")
ax.set_title("Variance per PC: both leaderboards show single dominant axis", fontsize=10)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: tau vs k for both
ax = axes[1]
if os.path.exists("../data/hf_imputation_full.csv") and os.path.exists("../data/livebench_imputation_full.csv"):
    hf_res = pd.read_csv("../data/hf_imputation_full.csv")
    lb_res = pd.read_csv("../data/livebench_imputation_full.csv")
    for res_df, lab, c in [(hf_res, "HF Open LLM (mean)", "#3b6ea8"),
                            (lb_res, "LiveBench (mean)", "#d35400")]:
        means = res_df.groupby("k")["tau"].mean().reset_index()
        maxes = res_df.groupby("k")["tau"].max().reset_index()
        ax.plot(means["k"], means["tau"], "-o", label=lab, color=c)
        ax.plot(maxes["k"], maxes["tau"], "--^", label=lab.replace("mean", "max"),
                color=c, alpha=0.7)
    ax.set_xlabel("Number of observed benchmarks $k$")
    ax.set_ylabel(r"Kendall $\tau$ vs full ranking")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_title("Imputation fidelity vs k: same shape, different leaderboards", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0.55, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("../figures/livebench_vs_hf.pdf", bbox_inches="tight")
plt.savefig("../figures/livebench_vs_hf.png", dpi=150, bbox_inches="tight")
plt.close()
print("../figures/livebench_vs_hf.{pdf,png}")

print("\nall main figures written to figures/")
