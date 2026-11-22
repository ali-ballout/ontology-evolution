"""Study D analysis against the protocol (protocols/study_d.md).
M1 any-correct recall@5/10/20, M2 review effort, M3 unsafe share of the top 10 (also split into subclass cycles and unsatisfiable classes), M4 anatomy of wrong top-10 suggestions;
H7 (tuned ranker vs names only, new classes); Q1 (side-by-side reasoner + L0 list, re-parenting 2021/2023); Q2 (M4 shares);
breakdowns by edit type, size and family. Per-ontology values (ontologies with >= 10 queries in the view, as in Studies
A-C); ontology bootstrap (4,000); cluster bootstrap over 20% overlap clusters when >= 8; block / independent split.
Queries whose Nov 2026 parents include no class that existed at the cutoff cannot be answered and are left out.
Inputs: results/study_d/measures (from study_d/measures.py). Writes reports/study_d_*.csv; report on stdout.
usage: analyze_d.py"""
import glob, os, sys, itertools, lzma, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
RES = paths.RESULTS; os.makedirs(paths.REPORTS, exist_ok=True)
q = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_d/measures/q_*.csv.gz")], ignore_index=True)
s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{RES}/study_d/measures/s_*.csv.gz"))], ignore_index=True)
D = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read()))
_k = ["cutoff", "ontology", "task", "i", "model"]
q = q.merge(s.groupby(_k)[["cycle", "disjoint"]].mean().rename(columns={"disjoint": "unsat"}).reset_index(), on=_k, how="left")
NOC = q[q.n_correct == 0].drop_duplicates(["cutoff", "ontology", "task", "i"])
print(f"queries without an answerable parent (left out): {len(NOC)} of {len(q.drop_duplicates(['cutoff','ontology','task','i']))}, all new classes: {NOC.task.eq('B').all()}")
q = q[q.n_correct > 0].copy(); q["hit5"], q["hit10"], q["hit20"] = q.hit5.astype(float), q.hit10.astype(float), q.hit20.astype(float)
SIZE = {(C, n): len(D["graphs"][C][n]["ids"]) for C in D["cuts"] for n in D["graphs"][C]}
q["size"] = [("< 3,000" if SIZE[k] < 3000 else "3,000-10,000" if SIZE[k] <= 10000 else "> 10,000") for k in zip(q.cutoff, q.ontology)]
BLOCK = {C: set(D["families"][C]["block"]) for C in D["cuts"]}; q["family"] = ["block" if o in BLOCK[C] else "independent" for C, o in zip(q.cutoff, q.ontology)]
CL = {}
for C in D["cuts"]:
    G = D["graphs"][C]; names = sorted(G); ids = {n: set(G[n]["ids"]) for n in names}; E = {n: {(G[n]["ids"][a], G[n]["ids"][b]) for a, b in G[n]["isa"]} for n in names}; par = {n: n for n in names}
    def f(x):
        while par[x] != x: x = par[x]
        return x
    for a, b in itertools.combinations(names, 2):
        if max(len(ids[a] & ids[b]) / max(1, min(len(ids[a]), len(ids[b]))), len(E[a] & E[b]) / max(1, min(len(E[a]), len(E[b])))) >= 0.20: par[f(a)] = f(b)
    CL[C] = {n: f(n) for n in names}
def boot(d, groups, B=4000):
    r = np.random.default_rng(0); u = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in u}
    return np.percentile([np.mean(np.concatenate([d[idx[g]] for g in r.choice(u, len(u))])) for _ in range(B)], [2.5, 97.5]), len(u)
def V(d):
    return {"new class": d[d.task == "B"], "re-parenting all": d[d.task == "A"], "entailed": d[(d.task == "A") & (d.entailed == True)],
            "not entailed": d[(d.task == "A") & (d.entailed == False)], "moved": d[(d.task == "A") & (d.kind == "moved")], "added": d[(d.task == "A") & (d.kind == "added")]}
def pom(d, m):
    nq = d.groupby("ontology").i.nunique(); keep = nq[nq >= 10].index
    return d[d.ontology.isin(keep)].groupby("ontology")[m].mean()
OUT, LEV = [], []
def cmp(part, C, view, a, b, an, bn, m="hit10", quiet=False):
    pa, pb = pom(a, m), pom(b, m); d = (pa - pb).dropna()
    if len(d) < 3: return None
    (lo, hi), _ = boot(d.values, np.array(d.index)); (clo, chi), ncl = boot(d.values, np.array([CL[C][o] for o in d.index]))
    blk = [o for o in d.index if o in BLOCK[C]]; ind = [o for o in d.index if o not in BLOCK[C]]
    r = dict(part=part, cutoff=C, view=view, metric=m, a=an, b=bn, a_val=pa[d.index].mean(), b_val=pb[d.index].mean(), n=len(d), wins=int((d > 0).sum()), mean=d.mean(), ci_lo=lo, ci_hi=hi,
             n_clusters=ncl, cl_lo=clo if ncl >= 8 else np.nan, cl_hi=chi if ncl >= 8 else np.nan, block=d[blk].mean() if blk else np.nan, n_block=len(blk), indep=d[ind].mean() if ind else np.nan, n_indep=len(ind))
    if not quiet:
        cl = f"; cluster ({ncl}) {clo:+.3f}..{chi:+.3f}" if ncl >= 8 else ""
        print(f"  {C[:4]} {view:17s} {m:6s} {an} vs {bn}: {r['a_val']:.3f} vs {r['b_val']:.3f}, wins {r['wins']}/{len(d)}, diff {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]{cl}; block {r['block']:+.3f} ({len(blk)}), indep {r['indep']:+.3f} ({len(ind)})")
    OUT.append(r); return r
print("\n== M1-M3 levels (ontology means)")
for C in sorted(q.cutoff.unique()):
    for view, d in V(q[q.cutoff == C]).items():
        for mo, g in d.groupby("model"):
            row = dict(cutoff=C, view=view, model=mo, n_ont=len(pom(g, "hit10")), n_q=len(g)); row.update({m: pom(g, m).mean() for m in ["hit5", "hit10", "hit20", "effort", "unsafe", "cycle", "unsat"]}); LEV.append(row)
LEV = pd.DataFrame(LEV); LEV.to_csv(os.path.join(paths.REPORTS, "study_d_levels.csv"), index=False)
with pd.option_context("display.width", 200, "display.max_rows", 500): print(LEV.round(3).to_string(index=False))
print("\n== H7: tuned ranker vs names only, new classes, recall@10; unsafe share of the ranker's top 10")
H7 = {}
for C in sorted(q.cutoff.unique()):
    d = q[q.cutoff == C]; r = cmp("H7", C, "new class", V(d[d.model == "GBR_del_c"])["new class"], V(d[d.model == "names only"])["new class"], "ranker", "names only")
    us = LEV[(LEV.cutoff == C) & (LEV.view == "new class") & (LEV.model == "GBR_del_c")].unsafe.iloc[0]
    H7[C] = (r["mean"] >= 0.10 and r["ci_lo"] > 0, us <= 0.05); print(f"     criterion 1 (>= +0.10, CI > 0): {H7[C][0]}; criterion 2 (unsafe {us:.3f} <= 0.05): {H7[C][1]}")
sup = sum(all(v) for v in H7.values()); print(f"  H7: both criteria at {sup}/3 cutoffs -> {'SUPPORTED' if sup >= 2 else 'NOT SUPPORTED'}")
print("\n== Comparisons (recall@10, effort, unsafe; unsafe = cycle or unsat)")
for C in sorted(q.cutoff.unique()):
    d = q[q.cutoff == C]; M = {m: d[d.model == m] for m in d.model.unique()}
    for view in ["new class", "re-parenting all", "entailed", "not entailed"]:
        for a, b in [("GBR_del_c", "hybrid"), ("GBR_del_c", "GNN_R"), ("GBR_del_c", "names only"), ("GBR_del_c", "R0"), ("L0", "GBR_del_c"), ("L0", "R0")]:
            if a in M and b in M and len(V(M[a])[view]) and len(V(M[b])[view]):
                for m in ["hit10", "effort", "unsafe"]:
                    if view == "new class" and m == "unsafe": continue
                    cmp("cmp", C, view, V(M[a])[view], V(M[b])[view], a, b, m)
print("\n== Unsafe suggestions by kind, re-parenting: hierarchy cycle (suggestion is an inferred descendant) and unsatisfiable class (declared-disjoint pair)")
for C in sorted(q.cutoff.unique()):
    d = q[q.cutoff == C]; M = {m: d[d.model == m] for m in d.model.unique()}
    for a, b in [("GBR_del_c", "GNN_R"), ("GBR_del_c", "names only"), ("GBR_del_c", "hybrid"), ("GBR_del_c", "R0"), ("GNN_R", "R0"), ("L0", "GBR_del_c")]:
        if a in M and b in M:
            for m in ["cycle", "unsat"]: cmp("kind", C, "re-parenting all", V(M[a])["re-parenting all"], V(M[b])["re-parenting all"], a, b, m)
print("\n== Q1: side-by-side list (reasoner top 5 + L0 top 5, interleaved, filled to 10) vs each list's top 10")
A = s[(s.task == "A") & s.model.isin(["R0", "L0"]) & s.cutoff.isin(["2021-07-01", "2023-07-01"])]
mrows = []
for k, g in A.groupby(["cutoff", "ontology", "i"]):
    r0 = g[g.model == "R0"].sort_values("pos"); l0 = g[g.model == "L0"].sort_values("pos")
    if r0.empty or l0.empty: continue
    seen, lst = set(), []
    for a, b in itertools.zip_longest(r0[["s", "correct"]].values.tolist(), l0[["s", "correct"]].values.tolist()):
        for e in (a, b):
            if e and e[0] not in seen and len(lst) < 10: seen.add(e[0]); lst.append(e[1])
    first = next((j + 1 for j, c in enumerate(lst) if c), 11)
    mrows.append(dict(cutoff=k[0], ontology=k[1], i=k[2], hit10_m=float(first <= 10), hit5_m=float(first <= 5), first_m=first))
Mg = pd.DataFrame(mrows)
qa = q[(q.task == "A") & q.cutoff.isin(["2021-07-01", "2023-07-01"])]
merged = qa[qa.model == "L0"].merge(Mg, on=["cutoff", "ontology", "i"]).assign(model="R0+L0", hit10=lambda x: x.hit10_m, hit5=lambda x: x.hit5_m)
for C in ["2021-07-01", "2023-07-01"]:
    d = qa[qa.cutoff == C]; mm = merged[merged.cutoff == C]
    for view in ["re-parenting all", "entailed", "not entailed"]:
        cmp("Q1", C, view, V(mm)[view], V(d[d.model == "L0"])[view], "R0+L0", "L0"); cmp("Q1", C, view, V(mm)[view], V(d[d.model == "R0"])[view], "R0+L0", "R0")
        cmp("Q1", C, view, V(mm)[view], V(d[d.model == "GBR_del_c"])[view], "R0+L0", "GBR_del_c")
print("\n== Q2 / M4: anatomy of wrong top-10 suggestions (share of wrong suggestions, ontology means)")
an = []
for (C, task, mo), g in q[q.model.isin(["GBR_del_c", "names only", "R0", "L0", "GNN_R", "hybrid"])].groupby(["cutoff", "task", "model"]):
    po = g.groupby("ontology")[["a_red", "b_near", "c_trans", "d_uns", "n_wrong"]].sum(); po = po[g.groupby("ontology").i.nunique() >= 10]
    sh = po[["a_red", "b_near", "c_trans", "d_uns"]].div(po.n_wrong, axis=0).mean()
    an.append(dict(cutoff=C, task=task, model=mo, n_ont=len(po), redundant=sh.a_red, near_miss=sh.b_near, transient=sh.c_trans, unsupported=sh.d_uns, wrong_per_list=g.n_wrong.mean()))
AN = pd.DataFrame(an); AN.to_csv(os.path.join(paths.REPORTS, "study_d_anatomy.csv"), index=False); print(AN.round(3).to_string(index=False))
print("\n== Breakdowns (tuned ranker and names only / R0; recall@10, ontology means)")
BR = []
for C in sorted(q.cutoff.unique()):
    for by in ["size", "family"]:
        for (task, lev), g in q[q.cutoff == C].groupby(["task", by]):
            for mo in ["GBR_del_c", "names only", "R0", "L0", "GNN_R"]:
                x = g[g.model == mo]
                if len(x): p = pom(x, "hit10"); BR.append(dict(cutoff=C, by=by, level=lev, task=task, model=mo, n_ont=len(p), hit10=p.mean(), unsafe=pom(x, "unsafe").mean()))
BR = pd.DataFrame(BR); BR.to_csv(os.path.join(paths.REPORTS, "study_d_breakdowns.csv"), index=False)
with pd.option_context("display.max_rows", 500): print(BR.pivot_table(index=["by", "level", "task", "cutoff"], columns="model", values="hit10").round(3).to_string())
pd.DataFrame(OUT).to_csv(os.path.join(paths.REPORTS, "study_d_comparisons.csv"), index=False)
