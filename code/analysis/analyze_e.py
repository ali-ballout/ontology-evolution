"""Study E analysis (GNN size sensitivity) against its protocol (protocols/study_e.md).
Inputs: results/study_e/val/*.csv (validation ranks of the size-selection models), results/study_e/ranks/*.csv.gz (test
ranks, every size and seed), results/study_b/ranks (the Study B GNN and the tuned ranker GBR_del on the same queries).
Selection per fold: highest validation MRR (mean over held-out ontologies), ties to the smaller network.
Per view: per-ontology MRR (GNN: mean over seeds), ontologies with >= 10 queries; paired differences with ontology
bootstrap 95% intervals (4,000 resamples) and, where >= 8 clusters, cluster bootstrap over 20%-overlap clusters.
Writes reports/study_e_comparisons.csv; report on stdout.  usage: analyze_e.py [RESULTS_DIR]"""
import glob, os, sys, itertools, lzma, pickle, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import paths
RES = sys.argv[1] if len(sys.argv) > 1 else paths.RESULTS; os.makedirs(paths.REPORTS, exist_ok=True)
C = "2023-07-01"; SIZES = ["L2d24", "L4d24", "L6d24", "L4d48", "L4d96"]; NPAR = {"L2d24": 8017, "L4d24": 13921, "L6d24": 19825, "L4d48": 54337, "L4d96": 214657}
V = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/study_e/val/{C}_std_*.csv"))], ignore_index=True)
E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/study_e/ranks/{C}_std_*.csv.gz"))], ignore_index=True)
B = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_b/ranks/{C}_std_*.csv.gz")], ignore_index=True)
K = B[B.regime == "GBR_del"]; R = B[B.regime == "R"]
for d in (V, E, K, R): d["rr"] = 1 / d["rank"]
D = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read())); G = D["graphs"][C]
names = sorted(G); ids = {n: set(G[n]["ids"]) for n in names}; EL = {n: {(G[n]["ids"][a], G[n]["ids"][b]) for a, b in G[n]["isa"]} for n in names}; par = {n: n for n in names}
def root(x):
    while par[x] != x: x = par[x]
    return x
for a, b in itertools.combinations(names, 2):
    if max(len(ids[a] & ids[b]) / max(1, min(len(ids[a]), len(ids[b]))), len(EL[a] & EL[b]) / max(1, min(len(EL[a]), len(EL[b])))) >= 0.20: par[root(a)] = root(b)
CL = {n: root(n) for n in names}
print("== Selection on held-out training ontologies (validation MRR, seed 0)")
vm = V.groupby(["fold", "config", "ontology"]).rr.mean().groupby(["fold", "config"]).mean().unstack()[SIZES]
print(vm.round(3).to_string())
SEL = {}
for f, row in vm.iterrows():
    best = row.max(); SEL[f] = min([s for s in SIZES if row[s] == best], key=lambda s: NPAR[s])
print("selected size per fold:", SEL)
def views(d):
    return {"not entailed": d[(d.task == "A") & (d.entailed == False)], "new class": d[d.task == "B"], "entailed": d[(d.task == "A") & (d.entailed == True)]}
def pom(d):
    nq = d.groupby("ontology").i.nunique(); keep = nq[nq >= 10].index; d = d[d.ontology.isin(keep)]
    return d.groupby(["ontology", "seed"]).rr.mean().groupby("ontology").mean()
def boot(x, grp, B=4000):
    r = np.random.default_rng(0); u = np.unique(grp); idx = {g: np.flatnonzero(grp == g) for g in u}
    return np.percentile([np.mean(np.concatenate([x[idx[g]] for g in r.choice(u, len(u))])) for _ in range(B)], [2.5, 97.5]), len(u)
OUT = []
def cmp(label, view, a, b, an, bn):
    pa, pb = pom(a), pom(b); d = (pa - pb).dropna()
    if len(d) < 3: return None
    (lo, hi), _ = boot(d.values, np.array(d.index)); (clo, chi), ncl = boot(d.values, np.array([CL[o] for o in d.index]))
    cl = f"; cluster ({ncl}) {clo:+.3f}..{chi:+.3f}" if ncl >= 8 else ""
    print(f"  {view:13s} {an} vs {bn}: {pa[d.index].mean():.3f} vs {pb[d.index].mean():.3f}, wins {int((d > 0).sum())}/{len(d)}, diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{cl}")
    r = dict(comparison=label, view=view, a=an, b=bn, a_mrr=pa[d.index].mean(), b_mrr=pb[d.index].mean(), n=len(d), wins=int((d > 0).sum()), mean=d.mean(), ci_lo=lo, ci_hi=hi,
             n_clusters=ncl, cl_lo=clo if ncl >= 8 else np.nan, cl_hi=chi if ncl >= 8 else np.nan); OUT.append(r); return r
S = pd.concat([E[(E.fold == f) & (E.config == s)] for f, s in SEL.items()]).assign(config="selected")
print("\n== Check: L4d24 retrained with the Study E code vs the Study B GNN (same network, same seeds)")
for v in views(E): cmp("check", v, views(E[E.config == "L4d24"])[v], views(R)[v], "L4d24 (Study E)", "GNN (Study B)")
print("\n== Primary: selected-size GNN vs tuned ranker (GBR_del)")
PRIM = {v: cmp("primary", v, views(S)[v], views(K)[v], "selected GNN", "tuned ranker") for v in ["not entailed", "new class", "entailed"]}
print("\n== Every size vs tuned ranker, and the best size on test (chosen after the fact, not deployable)")
lev = {}
for s in SIZES:
    for v in ["not entailed", "new class", "entailed"]:
        r = cmp("size vs ranker", v, views(E[E.config == s])[v], views(K)[v], s, "tuned ranker"); lev[(s, v)] = r["a_mrr"] if r else np.nan
for v in ["not entailed", "new class"]:
    best = max(SIZES, key=lambda s: lev[(s, v)]); print(f"  best size on test for {v}: {best}"); cmp("best on test", v, views(E[E.config == best])[v], views(K)[v], f"{best} (best on test)", "tuned ranker")
print("\n== Every size vs L4d24")
for s in SIZES:
    if s != "L4d24":
        for v in ["not entailed", "new class", "entailed"]: cmp("size vs L4d24", v, views(E[E.config == s])[v], views(E[E.config == "L4d24"])[v], s, "L4d24")
print("\n== Criterion: H5 stays rejected if, for the selected-size GNN, the interval of (GNN − ranker) includes zero or lies below zero on both views")
above = [v for v in ["not entailed", "new class"] if PRIM[v] and PRIM[v]["ci_lo"] > 0]
print("  views with the interval above zero:", above or "none", "->", "H5 STAYS REJECTED" if not above else "H5 REJECTION REVISED for " + ", ".join(above))
pd.DataFrame(OUT).to_csv(os.path.join(paths.REPORTS, "study_e_comparisons.csv"), index=False)
