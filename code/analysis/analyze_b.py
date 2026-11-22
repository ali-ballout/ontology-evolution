"""Study B analysis against the pre-declared comparisons (protocols/study_b.md).
Inputs: results/study_b/ranks/*.csv.gz (GNN regimes per seed; tuned ranker), results/study_a/ranks (training-free, same queries).
Per view: per-ontology MRR (GNN: mean over seeds), ontologies with >= 10 queries; paired differences: wins, mean, median,
ontology bootstrap 95% CI (4,000 resamples), per-seed means. RGPU (device check) is excluded from all comparisons.
Writes reports/study_b_comparisons.csv; the printed report goes to stdout.  usage: analyze_b.py"""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
RES = paths.RESULTS; os.makedirs(paths.REPORTS, exist_ok=True)
G = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_b/ranks/*.csv.gz") if "__ranker" not in f], ignore_index=True)
K = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_b/ranks/*__ranker.csv.gz")], ignore_index=True)
A = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_a/ranks/*.csv") if os.path.getsize(f) > 20], ignore_index=True)
A = A[A.model.isin(["names only", "hybrid", "reasoner: nearest inferred ancestor"])].rename(columns={"model": "regime"}).assign(seed=0)
meta = G.drop_duplicates(["cutoff", "ontology", "task", "i"])[["cutoff", "ontology", "task", "i", "kind", "h", "entailed"]]
A = A.drop(columns=["kind", "h", "entailed"]).merge(meta, on=["cutoff", "ontology", "task", "i"])
for d in (G, K, A): d["rr"] = 1 / d["rank"]
def views(d):
    return {"A not entailed": d[(d.task == "A") & (d.entailed == False)], "A all": d[d.task == "A"], "A entailed": d[(d.task == "A") & (d.entailed == True)], "B new class": d[d.task == "B"]}
def boot(x, B=4000):
    r = np.random.default_rng(0); return np.percentile([r.choice(x, len(x)).mean() for _ in range(B)], [2.5, 97.5])
OUT = []
def cmp(label, C, view, fs, a_df, b_df, a_name, b_name, seeds=True):
    """a_df, b_df: rows of one view; returns dict and prints"""
    nq = a_df.groupby("ontology").i.nunique(); keep = nq[nq >= 10].index
    pa = a_df[a_df.ontology.isin(keep)].groupby("ontology").rr.mean(); pb = b_df[b_df.ontology.isin(keep)].groupby("ontology").rr.mean()
    d = (pa - pb).dropna()
    if len(d) < 3: return
    lo, hi = boot(d.values); s = ""
    if seeds and a_df.seed.nunique() > 1:
        ps = a_df[a_df.ontology.isin(d.index)].groupby(["seed", "ontology"]).rr.mean().unstack(0); s = " | per seed " + " ".join(f"{(ps[c] - pb[d.index]).mean():+.3f}" for c in ps)
    print(f"  {C[:4]} {fs:8s} {view:15s} {a_name} vs {b_name}: {pa[d.index].mean():.3f} vs {pb[d.index].mean():.3f}, wins {int((d > 0).sum())}/{len(d)}, mean {d.mean():+.3f}, median {d.median():+.3f}, CI {lo:+.3f}..{hi:+.3f}{s}")
    OUT.append(dict(comparison=label, cutoff=C, foldset=fs, view=view, a=a_name, b=b_name, a_mrr=pa[d.index].mean(), b_mrr=pb[d.index].mean(), n=len(d), wins=int((d > 0).sum()), mean=d.mean(), median=d.median(), ci_lo=lo, ci_hi=hi))
CUTS = sorted(G.cutoff.unique())
print("== Device check: 2019 std fold 0, R seed 0 (CPU) vs RGPU seed 0 (GPU), same seed")
g0 = G[(G.cutoff == CUTS[0]) & (G.foldset == "std") & (G.fold == 0) & (G.seed == 0)]
for v, d in views(g0).items():
    a = d[d.regime == "R"].groupby("ontology").rr.mean(); b = d[d.regime == "RGPU"].groupby("ontology").rr.mean(); x = (b - a).dropna()
    sp = G[(G.cutoff == CUTS[0]) & (G.foldset == "std") & (G.fold == 0) & (G.regime == "R")]; sv = views(sp)[v].groupby(["seed", "ontology"]).rr.mean().unstack(0)
    print(f"  {v:15s} GPU − CPU per ontology: mean {x.mean():+.4f}, max |diff| {x.abs().max():.4f}; spread between the 3 seeds (mean |seed − seed mean|): {(sv.sub(sv.mean(axis=1), axis=0)).abs().mean().mean():.4f}")
G = G[G.regime != "RGPU"]
print("\n== H1: GNN R vs hybrid (and names only), per cutoff and fold set")
for C in CUTS:
    for fs in ["std", "famA", "famB", "famBsize"]:
        g = G[(G.cutoff == C) & (G.foldset == fs) & (G.regime == "R")]; a = A[(A.cutoff == C) & A.ontology.isin(g.ontology.unique())]
        for v in ["A not entailed", "B new class", "A all"]:
            cmp("H1", C, v, fs, views(g)[v], views(a[a.regime == "hybrid"])[v], "GNN R", "hybrid")
            if v == "B new class": cmp("H1-names", C, v, fs, views(g)[v], views(a[a.regime == "names only"])[v], "GNN R", "names only")
print("\n== H1: effect of related training ontologies vs training size (famB vs famBsize, std vs famA on the same ontologies)")
for C in CUTS:
    for v in ["A not entailed", "B new class"]:
        b = G[(G.cutoff == C) & (G.foldset == "famB") & (G.regime == "R")]; bs = G[(G.cutoff == C) & (G.foldset == "famBsize") & (G.regime == "R")]
        cmp("H1-size", C, v, "famB", views(b)[v], views(bs)[v], "famB (all others)", "famBsize (size-matched)")
        st = G[(G.cutoff == C) & (G.foldset == "std") & (G.regime == "R")]; fa = G[(G.cutoff == C) & (G.foldset == "famA") & (G.regime == "R")]
        cmp("H1-related", C, v, "famA", views(st[st.ontology.isin(fa.ontology.unique())])[v], views(fa)[v], "std (related in training)", "famA (no related)")
print("\n== H4: history regimes vs R (std folds)")
for C in CUTS[1:]:
    g = G[(G.cutoff == C) & (G.foldset == "std")]
    for v in ["A not entailed", "A entailed", "B new class"]:
        for r in ["M", "E", "N", "X"]: cmp("H4", C, v, "std", views(g[g.regime == r])[v], views(g[g.regime == "R"])[v], f"GNN {r}", "GNN R")
        cmp("H4-XvsE", C, v, "std", views(g[g.regime == "X"])[v], views(g[g.regime == "E"])[v], "GNN X", "GNN E")
print("\n== H5: GNN vs tuned non-neural ranker (std folds)")
for C in CUTS:
    g = G[(G.cutoff == C) & (G.foldset == "std")]; k = K[K.cutoff == C]
    for v in ["A not entailed", "A all", "B new class"]:
        cmp("H5", C, v, "std", views(g[g.regime == "R"])[v], views(k[k.regime == "GBR_del"])[v], "GNN R", "GBR_del")
        if (k.regime == "GBR_delhist").any(): cmp("H5", C, v, "std", views(g[g.regime == "M"])[v], views(k[k.regime == "GBR_delhist"])[v], "GNN M", "GBR_delhist")
        a = A[(A.cutoff == C) & (A.regime == "hybrid") & A.ontology.isin(k.ontology.unique())]
        cmp("H5-ranker-vs-hybrid", C, v, "std", views(k[k.regime == "GBR_del"])[v], views(a)[v], "GBR_del", "hybrid", seeds=False)
print("\n== Ablations (2023, std): NAMES / STRUCT / NOREL vs R")
g = G[(G.cutoff == CUTS[2]) & (G.foldset == "std")]
for v in ["A not entailed", "A entailed", "B new class"]:
    for r in ["NAMES", "STRUCT", "NOREL"]: cmp("ablation", CUTS[2], v, "std", views(g[g.regime == r])[v], views(g[g.regime == "R"])[v], f"GNN {r}", "GNN R")
print("\n== Reasoner vs GNN R on entailed / not-entailed (std)")
for C in CUTS:
    g = G[(G.cutoff == C) & (G.foldset == "std") & (G.regime == "R")]; a = A[(A.cutoff == C) & (A.regime == "reasoner: nearest inferred ancestor")]
    for v in ["A entailed", "A not entailed", "A all"]: cmp("reasoner", C, v, "std", views(g)[v], views(a[a.ontology.isin(g.ontology.unique())])[v], "GNN R", "reasoner")
pd.DataFrame(OUT).to_csv(os.path.join(paths.REPORTS, "study_b_comparisons.csv"), index=False)
