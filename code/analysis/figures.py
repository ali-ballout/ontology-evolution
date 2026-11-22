"""Figures 1-3 of the paper from the Study D reports (run analyze_d.py first). Writes reports/figures/fig*.pdf and .png.
usage: figures.py"""
import os, sys
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8986", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
COL = {"names only": "#2a78d6", "hybrid": "#eb6834", "R0": "#1baf7a", "GBR_del_c": "#eda100", "L0": "#e87ba4", "GNN_R": "#008300", "R0+L0": "#6250d6"}
LAB = {"names only": "Name search", "hybrid": "Hybrid heuristic", "R0": "Reasoner ranker", "GBR_del_c": "Tuned ranker", "L0": "History-trained ranker", "GNN_R": "GNN", "R0+L0": "Side-by-side list"}
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
FD = os.path.join(paths.REPORTS, "figures"); os.makedirs(FD, exist_ok=True)
L = pd.read_csv(os.path.join(paths.REPORTS, "study_d_levels.csv")); Cm = pd.read_csv(os.path.join(paths.REPORTS, "study_d_comparisons.csv"))
q1 = Cm[(Cm.part == "Q1") & (Cm.b == "L0")]
cuts = ["2019-07-01", "2021-07-01", "2023-07-01"]
# ---- Figure 2: recall@10 by view ----
views = [("new class", "New-class placement"), ("entailed", "Re-parenting, entailed"), ("not entailed", "Re-parenting, not entailed")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
for ax, (v, title) in zip(axs, views):
    models = ["names only", "hybrid", "GBR_del_c", "GNN_R"] if v == "new class" else ["names only", "hybrid", "R0", "GBR_del_c", "L0", "GNN_R", "R0+L0"]
    w = 0.8 / len(models)
    for j, m in enumerate(models):
        ys = []
        for c in cuts:
            if m == "R0+L0":
                r = q1[(q1.cutoff == c) & (q1.view == v)]; ys.append(r.a_val.iloc[0] if len(r) else np.nan)
            else:
                r = L[(L.cutoff == c) & (L.view == v) & (L.model == m)]; ys.append(r.hit10.iloc[0] if len(r) else np.nan)
        x = np.arange(3) + (j - (len(models) - 1) / 2) * w
        ax.bar(x, ys, w * 0.86, color=COL[m], label=LAB[m], edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["2019", "2021", "2023"]); ax.set_title(title, fontsize=9, loc="left")
    ax.set_ylim(0, 1.0); ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.6); ax.set_axisbelow(True)
axs[0].set_ylabel("Recall@10 (any correct parent)")
h, l = [], []
for ax in axs:
    for hh, ll in zip(*ax.get_legend_handles_labels()):
        if ll not in l: h.append(hh); l.append(ll)
fig.legend(h, l, ncol=4, loc="lower center", frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(os.path.join(FD, "fig2_recall10.pdf")); fig.savefig(os.path.join(FD, "fig2_recall10.png"), dpi=300); plt.close(fig)
# ---- Figure 3: unsafe suggestions by kind, re-parenting ----
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
models = ["names only", "hybrid", "R0", "GBR_del_c", "L0", "GNN_R"]; w = 0.8 / len(models)
for ax, (col, title) in zip(axs, [("cycle", "Subclass cycle"), ("unsat", "Unsatisfiable class")]):
    for j, m in enumerate(models):
        ys = []
        for c in cuts:
            r = L[(L.cutoff == c) & (L.view == "re-parenting all") & (L.model == m)]; ys.append(r[col].iloc[0] if len(r) else np.nan)
        x = np.arange(3) + (j - (len(models) - 1) / 2) * w
        ax.bar(x, np.array(ys) * 100, w * 0.86, color=COL[m], label=LAB[m], edgecolor="white", linewidth=0.6)
    ax.set_xticks(range(3)); ax.set_xticklabels(["2019", "2021", "2023"]); ax.set_title(title, fontsize=9, loc="left")
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.6); ax.set_axisbelow(True); ax.set_ylim(0, 9); ax.set_yticks(range(0, 10, 2))
axs[0].set_ylabel("Top-10 suggestions (%)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, ncol=6, loc="lower center", frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=1.2)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FD, "fig3_unsafe.pdf")); fig.savefig(os.path.join(FD, "fig3_unsafe.png"), dpi=300); plt.close(fig)
# ---- Figure 1: framework schematic ----
fig, ax = plt.subplots(figsize=(7.2, 3.2)); ax.set_xlim(2016.6, 2027.2); ax.set_ylim(-0.6, 3.6); ax.axis("off")
ax.annotate("", xy=(2027.0, 0), xytext=(2016.8, 0), arrowprops=dict(arrowstyle="->", color="#52514e", lw=1))
for y in range(2017, 2027): ax.plot([y, y], [-0.06, 0.06], color="#52514e", lw=0.8); ax.text(y, -0.25, str(y), ha="center", fontsize=7.5, color="#52514e")
ax.plot([2026.88, 2026.88], [-0.1, 3.1], color="#0b0b0b", lw=1, ls=":"); ax.text(2026.88, 3.25, "Nov 2026 release\n(targets judged here)", ha="center", fontsize=7.5)
for k, (c, col) in enumerate(zip([2019.5, 2021.5, 2023.5], ["#2a78d6", "#eb6834", "#008300"])):
    y = 0.75 + k * 0.85
    ax.add_patch(FancyBboxPatch((c - 2.0, y - 0.13), 1.95, 0.26, boxstyle="round,pad=0.01", fc="#f1f0ec", ec="#8a8986", lw=0.6))
    ax.text(c - 1.02, y, "edit history (2 years)" if k else "(no history window)", ha="center", va="center", fontsize=7)
    ax.plot([c, c], [0, y + 0.2], color=col, lw=1.4, zorder=1)
    ax.text(c, y + 0.27, f"cutoff 1 Jul {int(c)}", ha="center", fontsize=7.5, color="#0b0b0b")
    ax.annotate("", xy=(2026.8, y), xytext=(c + 0.05, y), arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.text(c + 0.12, y - 0.2, "test window (horizons 1 y, 2 y, 3+ y)", ha="left", fontsize=6.8, color="#52514e", zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))
ax.text(2016.7, 3.45, "Every release a model sees is dated on or before its cutoff", fontsize=8, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(FD, "fig1_framework.pdf")); fig.savefig(os.path.join(FD, "fig1_framework.png"), dpi=300); plt.close(fig)
print("ok")
