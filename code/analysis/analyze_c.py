"""Study C analysis against the protocol (protocols/study_c.md).
Part 1 (H1 for the ranker): GBR_del_c vs hybrid (and names only) on famA / famB / famBsize, not-entailed re-parenting and
new classes; rejected if the famA interval includes zero at a majority of cutoffs for both tasks.
Part 2 (H6): re-parenting at 2021 and 2023, std folds. S3 vs R0 (Study A reasoner ranker) and vs L0 on all edits (both
intervals must exclude 0, S3 better) and S3 vs L0 on non-entailed edits (lower bound > -0.01). S1, S2, S4 against the same
criterion (descriptive); oracle = R0's rank if the target is entailed, else L0's. Hits@1/10 and coverage.
Statistics: per-ontology MRR, ontology bootstrap (4,000); cluster bootstrap over 20% overlap clusters recomputed at the
cutoff when >= 8 clusters; block / independent split. Writes reports/study_c_comparisons.csv; report on stdout.
usage: analyze_c.py"""
import glob, os, sys, itertools, lzma, pickle, numpy as np, pandas as pd
import gzip
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
RES = paths.RESULTS; os.makedirs(paths.REPORTS, exist_ok=True)
C_ = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_c/ranks/*.csv.gz") if len(gzip.open(f).read(10)) > 5], ignore_index=True)   # empty files = test ontologies with no re-parenting queries (h6)
C_["rr"] = 1 / C_["rank"]
A = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_a/ranks/*.csv") if os.path.getsize(f) > 20], ignore_index=True)
A = A[A.model.isin(["names only", "hybrid", "reasoner: nearest inferred ancestor"])].rename(columns={"model": "regime"})
meta = C_.drop_duplicates(["cutoff", "ontology", "task", "i"])[["cutoff", "ontology", "task", "i", "kind", "h", "entailed"]]
A = A.drop(columns=["kind", "h", "entailed"]).merge(meta, on=["cutoff", "ontology", "task", "i"]); A["rr"] = 1 / A["rank"]
A["regime"] = A.regime.replace({"reasoner: nearest inferred ancestor": "R0"})
D = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read()))
CL = {}
for C in D["cuts"]:
    G = D["graphs"][C]; names = sorted(G); ids = {n: set(G[n]["ids"]) for n in names}; E = {n: {(G[n]["ids"][a], G[n]["ids"][b]) for a, b in G[n]["isa"]} for n in names}; par = {n: n for n in names}
    def f(x):
        while par[x] != x: x = par[x]
        return x
    for a, b in itertools.combinations(names, 2):
        if max(len(ids[a] & ids[b]) / max(1, min(len(ids[a]), len(ids[b]))), len(E[a] & E[b]) / max(1, min(len(E[a]), len(E[b])))) >= 0.20: par[f(a)] = f(b)
    CL[C] = {n: f(n) for n in names}
BLOCK = {C: set(D["families"][C]["block"]) for C in D["cuts"]}
def boot(d, groups, B=4000):
    r = np.random.default_rng(0); u = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in u}
    return np.percentile([np.mean(np.concatenate([d[idx[g]] for g in r.choice(u, len(u))])) for _ in range(B)], [2.5, 97.5]), len(u)
OUT = []
def cmp(part, C, fs, view, a, b, an, bn, metric="rr"):
    nq = a.groupby("ontology").i.nunique(); keep = nq[nq >= 10].index
    pa = a[a.ontology.isin(keep)].groupby("ontology")[metric].mean(); pb = b[b.ontology.isin(keep)].groupby("ontology")[metric].mean(); d = (pa - pb).dropna()
    if len(d) < 3: return None
    (lo, hi), _ = boot(d.values, np.array(d.index)); g = np.array([CL[C][o] for o in d.index]); (clo, chi), ncl = boot(d.values, g)
    cl = f"; cluster CI ({ncl}) {clo:+.3f}..{chi:+.3f}" if ncl >= 8 else f"; {ncl} clusters (too few)"
    blk = [o for o in d.index if o in BLOCK[C]]; ind = [o for o in d.index if o not in BLOCK[C]]
    sp = f"; block {d[blk].mean():+.3f} ({len(blk)}), independent {d[ind].mean():+.3f} ({len(ind)})" if blk and ind else ""
    print(f"  {C[:4]} {fs:8s} {view:15s} {metric:3s} {an} vs {bn}: {pa[d.index].mean():.3f} vs {pb[d.index].mean():.3f}, wins {int((d > 0).sum())}/{len(d)}, mean {d.mean():+.3f}, CI {lo:+.3f}..{hi:+.3f}{cl}{sp}")
    r = dict(part=part, cutoff=C, foldset=fs, view=view, metric=metric, a=an, b=bn, a_val=pa[d.index].mean(), b_val=pb[d.index].mean(), n=len(d), wins=int((d > 0).sum()), mean=d.mean(), ci_lo=lo, ci_hi=hi, n_clusters=ncl, cl_lo=clo if ncl >= 8 else None, cl_hi=chi if ncl >= 8 else None)
    OUT.append(r); return r
def V(d):
    return {"A not entailed": d[(d.task == "A") & (d.entailed == False)], "A all": d[d.task == "A"], "A entailed": d[(d.task == "A") & (d.entailed == True)], "B new class": d[d.task == "B"]}
print("== Part 1: H1 for the tuned ranker (GBR_del_c) on family-held-out folds")
H1 = {}
for C in sorted(C_.cutoff.unique()):
    for fs in ["famA", "famB", "famBsize"]:
        g = C_[(C_.cutoff == C) & (C_.foldset == fs) & (C_.regime == "GBR_del_c")]
        if g.empty: continue
        a = A[(A.cutoff == C) & A.ontology.isin(g.ontology.unique())]
        for v in ["A not entailed", "B new class"]:
            r = cmp("H1", C, fs, v, V(g)[v], V(a[a.regime == "hybrid"])[v], "ranker", "hybrid")
            if fs == "famA" and r: H1[(C, v)] = r["ci_lo"] > 0
            if v == "B new class": cmp("H1", C, fs, v, V(g)[v], V(a[a.regime == "names only"])[v], "ranker", "names only")
print("  famA intervals excluding zero:", H1)
print("\n== Part 2: H6, re-parenting, std folds")
for C in sorted(C_[C_.task.eq("A") & C_.foldset.eq("std")].cutoff.unique()):
    g = C_[(C_.cutoff == C) & (C_.foldset == "std")]; a = A[(A.cutoff == C) & A.ontology.isin(g.ontology.unique())]
    r0 = a[a.regime == "R0"]; L0 = g[g.regime == "L0"]
    m = L0.merge(r0[["ontology", "task", "i", "rank"]], on=["ontology", "task", "i"], suffixes=("", "_r0"))
    orc = m.assign(rank=np.where(m.entailed == True, m.rank_r0, m["rank"]), regime="Oracle"); orc["rr"] = 1 / orc["rank"]
    models = {k: g[g.regime == k] for k in ["S3", "S1", "S2", "S4", "L0", "L0d"]}; models["Oracle"] = orc
    for metric in ["rr"]:
        for s in ["S3", "S1", "S2", "S4", "Oracle", "L0d"]:
            for v in ["A all", "A entailed", "A not entailed"]:
                cmp("H6", C, "std", v, V(models[s])[v], V(r0)[v], s, "R0", metric); cmp("H6", C, "std", v, V(models[s])[v], V(L0)[v], s, "L0", metric)
    for k in ["h1", "h10"]:
        for d in list(models.values()) + [r0, L0]: d[k] = d["rank"] <= (1 if k == "h1" else 10)
    for s in ["S3", "L0"]:
        for v in ["A all", "A not entailed"]:
            for k in ["h1", "h10"]: cmp("H6-hits", C, "std", v, V(models[s])[v], V(r0)[v], s, "R0", k)
    print(f"  {C[:4]} coverage (target among L0/S3 candidates): {g[g.regime == 'L0'].covered.mean():.3f}; median candidates {g[g.regime == 'L0'].n_candidates.median():.0f}")
pd.DataFrame(OUT).to_csv(os.path.join(paths.REPORTS, "study_c_comparisons.csv"), index=False)
print("\n== H6 criterion for S3 and the descriptive strategies")
O = pd.DataFrame(OUT)
for C in sorted(O[O.part == "H6"].cutoff.unique()):
    for s in ["S3", "S1", "S2", "S4"]:
        x = O[(O.part == "H6") & (O.cutoff == C) & (O.a == s) & (O.metric == "rr")]
        allR0 = x[(x.view == "A all") & (x.b == "R0")].iloc[0]; allL0 = x[(x.view == "A all") & (x.b == "L0")].iloc[0]; ne = x[(x.view == "A not entailed") & (x.b == "L0")].iloc[0]
        ok1 = allR0.ci_lo > 0 and allL0.ci_lo > 0; ok2 = ne.ci_lo > -0.01
        print(f"  {C[:4]} {s}: all vs R0 {allR0['mean']:+.3f} [{allR0.ci_lo:+.3f}], all vs L0 {allL0['mean']:+.3f} [{allL0.ci_lo:+.3f}], non-entailed vs L0 {ne['mean']:+.3f} [lower {ne.ci_lo:+.3f}] -> (1) {ok1}, (2) {ok2}, success {ok1 and ok2}")
print("\n== Part 3: runtime (2023 test ontologies)")
t = pd.read_csv(f"{RES}/study_c/timing.csv"); print(t.describe().loc[["50%", "max"]].round(2).to_string())
print(t.sort_values("classes").tail(3)[["ontology", "classes", "features_ms", "L0_ms", "S3_ms", "GNN_cpu_ms", "GNN_cuda_ms"]].round(1).to_string(index=False))
