"""Random deletion on grouped folds (Table 10): GNN (mean over three seeds) against the hybrid (w = 0.25), names available;
independent ontologies, connected block, GO/Uberon/CL, all; ontology and overlap-cluster bootstrap intervals (4,000).
Writes reports/deletion_grouped.json.  usage: summary_grouped.py"""
import os, sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
D = os.path.join(paths.DATA, "deletion"); R = os.path.join(paths.RESULTS, "deletion", "grouped")
GJ = json.load(open(f"{D}/grouped_folds.json")); BLOCK = GJ[0]["test"]; LARGE = GJ[0]["extra"]; IND = GJ[1]["test"] + GJ[2]["test"]; CL = json.load(open(f"{D}/clusters.json"))
G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/grouped_gnn_ranks_f*_s*.csv"))]); G = G[G.withheld == 0]
HY = pd.read_csv(f"{R}/hybrid_ranks.csv"); HY = HY[(HY.model == "name vote + structural vote (w=0.25)") & (HY.withheld == 0)]
g = G.assign(rr=1 / G["rank"]).groupby(["ontology", "seed"]).rr.mean().groupby("ontology").mean()
h = HY.assign(rr=1 / HY["rank"]).groupby("ontology").rr.mean()
d = (g - h).dropna(); print("ontologies", len(d), "seeds per fold", G.groupby("fold").seed.nunique().to_dict())
def boot(x, grp, B=4000):
    r = np.random.default_rng(0); u = np.unique(grp); idx = {k: np.flatnonzero(grp == k) for k in u}
    return np.percentile([np.mean(np.concatenate([x[idx[k]] for k in r.choice(u, len(u))])) for _ in range(B)], [2.5, 97.5])
out = {}
for lab, sel in (("independent", [o for o in d.index if o in IND]), ("block", [o for o in d.index if o in BLOCK]), ("large", [o for o in d.index if o in LARGE]), ("all", list(d.index))):
    x = d[sel]; out[lab] = dict(n=len(x), gnn=g[sel].mean(), hybrid=h[sel].mean(), wins=int((x > 0).sum()), mean=x.mean(), median=x.median())
    if lab == "all":
        lo, hi = boot(x.values, np.arange(len(x))); clo, chi = boot(x.values, np.array([CL[o] for o in sel]))
        out[lab].update(ci=[lo, hi], cl_ci=[clo, chi], n_clusters=len(set(CL[o] for o in sel)))
for k, v in out.items(): print(k, {a: (round(float(b), 3) if isinstance(b, float) else b) for a, b in v.items()})
os.makedirs(paths.REPORTS, exist_ok=True); json.dump(out, open(os.path.join(paths.REPORTS, "deletion_grouped.json"), "w"), default=float, indent=1)
print(d.round(3).sort_values().to_string())
