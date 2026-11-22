"""Dependence audit for the learned-ranker comparisons (no new model runs).
1. Standard folds: share of test queries whose target link is also an is-a link in the fold's training material, and the
   main comparisons with those queries removed.
2. Family-held-out fold sets: number of families and test ontologies behind each analysis.
3. Cutoffs: share of the sampled queries of a cutoff that are also sampled queries of an earlier cutoff.
4. Candidate list of the tuned ranker: share of queries with a target among the hybrid's top 300, per task.
5. New-class targets: queries by whether their target parents are parents in the current release, and the new-class
   results restricted to queries whose targets are all current parents (results/study_a/newclass_target_status.csv).
Per-ontology MRR, ontologies with >= 10 queries, ontology bootstrap 95% intervals (4,000 resamples).
usage: audit_dependence.py   writes reports/dependence_audit.csv; the printed report goes to stdout."""
import glob, os, sys, lzma, pickle, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
RES = paths.RESULTS; os.makedirs(paths.REPORTS, exist_ok=True); OUT = []
DATA = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read()))
def boot(x, B=4000):
    r = np.random.default_rng(0); return np.percentile([r.choice(x, len(x)).mean() for _ in range(B)], [2.5, 97.5])
def per_ont(d):
    nq = d.groupby("ontology").i.nunique(); return d[d.ontology.isin(nq[nq >= 10].index)].groupby("ontology").rr.mean()
G = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_b/ranks/*.csv.gz") if "__ranker" not in f], ignore_index=True); G = G[G.regime == "R"]
K = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_b/ranks/*_std_*__ranker.csv.gz")], ignore_index=True); K = K[K.regime == "GBR_del"].drop(columns=["in_training"])
A = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_a/ranks/*.csv") if os.path.getsize(f) > 20], ignore_index=True)
A = A[A.model == "hybrid"][["cutoff", "ontology", "task", "i", "rank"]]
S = G[G.foldset == "std"]; meta = S.drop_duplicates(["cutoff", "ontology", "task", "i"])[["cutoff", "ontology", "task", "i", "entailed", "in_training"]]
K = K.merge(meta[["cutoff", "ontology", "task", "i", "in_training"]], on=["cutoff", "ontology", "task", "i"]); A = A.merge(meta, on=["cutoff", "ontology", "task", "i"])
for d in (S, K, A): d["rr"] = 1 / d["rank"]
S = S.groupby(["cutoff", "ontology", "task", "i", "entailed", "in_training"], dropna=False).rr.mean().reset_index()
VIEWS = {"non-entailed re-parenting": lambda d: d[(d.task == "A") & (d.entailed == False)], "new classes": lambda d: d[d.task == "B"]}
CUTS = sorted(meta.cutoff.unique())
print("== 1. Standard folds: test targets that are also is-a links in the fold's training material")
for C in CUTS:
    for v, f in VIEWS.items():
        m = f(meta[meta.cutoff == C]); nq = m.groupby("ontology").i.nunique(); onts = nq[nq >= 10].index; m = m[m.ontology.isin(onts)]
        byo = m.groupby("ontology").in_training.mean(); top = byo[byo > 0].sort_values(ascending=False)
        print(f"  {C[:4]} {v}: {int(m.in_training.sum())} of {len(m)} queries ({m.in_training.mean():.1%}) in {int((byo > 0).sum())} of {len(onts)} ontologies" + (f"; highest {', '.join(f'{o} {x:.0%}' for o, x in top.head(3).items())}" if len(top) else ""))
        OUT.append(dict(part="overlap", cutoff=C, view=v, queries=len(m), overlapping=int(m.in_training.sum()), ontologies=len(onts), ontologies_affected=int((byo > 0).sum())))
        for label, a, b in (("GNN − hybrid", S, A), ("tuned ranker − hybrid", K, A), ("tuned ranker − GNN", K, S)):
            for excl in (False, True):
                fa, fb = f(a[a.cutoff == C]), f(b[b.cutoff == C])
                if excl: fa, fb = fa[~fa.in_training.astype(bool)], fb[~fb.in_training.astype(bool)]
                d = (per_ont(fa) - per_ont(fb)).dropna(); lo, hi = boot(d.values)
                print(f"      {label:22s} {'overlapping queries removed' if excl else 'all queries':27s}: {d.mean():+.3f}, CI {lo:+.3f}..{hi:+.3f}, wins {int((d > 0).sum())}/{len(d)}")
                OUT.append(dict(part="comparison", cutoff=C, view=v, comparison=label, overlapping_removed=excl, n=len(d), mean=d.mean(), ci_lo=lo, ci_hi=hi, wins=int((d > 0).sum())))
print("\n== 2. Families behind the family-held-out analyses (test ontologies with >= 10 queries)")
F = G[G.foldset.isin(["famA", "famB"])].drop_duplicates(["cutoff", "foldset", "ontology", "task", "i"])
for C in CUTS:
    fam = DATA["families"][C]
    for fs in ("famA", "famB"):
        for v, f in VIEWS.items():
            m = f(F[(F.cutoff == C) & (F.foldset == fs)]); nq = m.groupby("ontology").i.nunique(); n = int((nq >= 10).sum())
            print(f"  {C[:4]} {fs} {v}: {n} test ontologies in {1 if fs == 'famA' else n} famil{'y (the connected block of ' + str(len(fam['block'])) + ' ontologies)' if fs == 'famA' else 'ies (each independent ontology is its own family)'}")
            OUT.append(dict(part="families", cutoff=C, foldset=fs, view=v, ontologies=n, families=1 if fs == "famA" else n))
print("\n== 3. Sampled queries shared between cutoffs (same child and target for re-parenting; same class for new classes)")
def keys(C, task): return {(n, t["x"], t["y"]) if task == "A" else (n, t["x"]) for n, Q in DATA["queries"][C].items() for t in Q[task]}
for task, name in (("A", "re-parenting"), ("B", "new classes")):
    for i, C in enumerate(CUTS[1:], 1):
        for E in CUTS[:i]:
            a, b = keys(C, task), keys(E, task); onts = {k[0] for k in b}; a2 = {k for k in a if k[0] in onts}
            print(f"  {name}: {len(a & b)} of {len(a)} queries at {C[:4]} are also queries at {E[:4]} ({len(a & b) / len(a):.0%}; {len(a & b) / max(1, len(a2)):.0%} within ontologies sampled at both)")
            OUT.append(dict(part="cutoffs", view=name, cutoff=C, earlier=E, queries=len(a), shared=len(a & b)))
print("\n== 4. Tuned ranker's candidate list (hybrid top 300): share of queries with a target in it, std folds")
R = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_d/ranker_ranks/*.csv.gz")], ignore_index=True)
for C in CUTS:
    r = R[R.cutoff == C]
    for task, name in (("A", "re-parenting"), ("B", "new classes")):
        x = r[r.task == task]; print(f"  {C[:4]} {name}: {x.covered.mean():.3f} of {len(x)} queries; mean over ontologies {x.groupby('ontology').covered.mean().mean():.3f}")
        OUT.append(dict(part="candidates", cutoff=C, view=name, queries=len(x), covered=x.covered.mean()))
print("\n== 5. New-class queries by target status, and results on queries whose targets are all parents in the current release")
T = pd.read_csv(f"{RES}/study_a/newclass_target_status.csv"); T["status"] = np.where(T.current == 0, "none", np.where(T.current == T.targets, "all", "some"))
for C in CUTS:
    t = T[T.cutoff == C]; print(f"  {C[:4]}: {len(t)} queries; all targets current {int((t.status == 'all').sum())} ({(t.status == 'all').mean():.1%}), some {int((t.status == 'some').sum())}, none {int((t.status == 'none').sum())}")
    OUT.append(dict(part="newclass targets", cutoff=C, queries=len(t), all_current=int((t.status == "all").sum()), some_current=int((t.status == "some").sum()), none_current=int((t.status == "none").sum())))
print(f"  total: {len(T)} queries; all {int((T.status == 'all').sum())}, some {int((T.status == 'some').sum())}, none {int((T.status == 'none').sum())}")
TS = T[[n in DATA["test"][C] for C, n in zip(T.cutoff, T.ontology)]]; nA = sum(len(Q["A"]) for C in CUTS for n, Q in DATA["queries"][C].items()); nAt = sum(len(DATA["queries"][C][n]["A"]) for C in CUTS for n in DATA["test"][C])
print(f"  in the test sets (ontologies with >= 10 queries in one task; {sum(len(DATA['test'][C]) for C in CUTS)} sets): {len(TS)} new-class queries; all {int((TS.status == 'all').sum())} ({(TS.status == 'all').mean():.1%}), some {int((TS.status == 'some').sum())} ({(TS.status == 'some').mean():.1%}), none {int((TS.status == 'none').sum())} ({(TS.status == 'none').mean():.1%}); re-parenting queries {nAt} of {nA} sampled")
AN = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_a/ranks/*.csv") if os.path.getsize(f) > 20], ignore_index=True); AN = AN[(AN.model == "names only") & (AN.task == "B")].copy(); AN["rr"] = 1 / AN["rank"]
KC = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_c/ranks/h1_*.csv.gz")], ignore_index=True); KC = KC[KC.task == "B"].copy(); KC["rr"] = 1 / KC["rank"]
GF = G[(G.task == "B") & G.foldset.isin(["famA", "famB"])].copy(); GF["rr"] = 1 / GF["rank"]; GF = GF.groupby(["cutoff", "foldset", "ontology", "i"]).rr.mean().reset_index()
keep = T[T.status == "all"][["cutoff", "ontology", "i"]]
def restrict(d): return d.merge(keep, on=["cutoff", "ontology", "i"])
B_ = lambda d, C: d[(d.cutoff == C) & (d.task == "B")] if "task" in d else d[d.cutoff == C]
for C in CUTS:
    lv = {k: (per_ont(B_(d, C)).mean(), per_ont(restrict(B_(d, C))).mean()) for k, d in (("name search", AN), ("hybrid", A), ("GNN", S), ("tuned ranker", K))}
    print(f"  {C[:4]} std, MRR all queries -> restricted: " + "; ".join(f"{k} {a:.3f} -> {b:.3f}" for k, (a, b) in lv.items()))
    for label, a, b in (("GNN − hybrid", S, A), ("tuned ranker − hybrid", K, A), ("tuned ranker − GNN", K, S)):
        d0 = (per_ont(B_(a, C)) - per_ont(B_(b, C))).dropna(); d = (per_ont(restrict(B_(a, C))) - per_ont(restrict(B_(b, C)))).dropna(); lo, hi = boot(d.values)
        print(f"      {label:22s}: all {d0.mean():+.3f} -> restricted {d.mean():+.3f}, CI {lo:+.3f}..{hi:+.3f}, wins {int((d > 0).sum())}/{len(d)}")
        OUT.append(dict(part="newclass restricted", cutoff=C, foldset="std", comparison=label, all=d0.mean(), mean=d.mean(), ci_lo=lo, ci_hi=hi, n=len(d), wins=int((d > 0).sum())))
    for fs in ("famA", "famB"):
        for label, a in (("GNN − hybrid", GF[GF.foldset == fs]), ("tuned ranker − hybrid", KC[KC.foldset == fs])):
            a = a[a.cutoff == C]; h = A[(A.cutoff == C) & (A.task == "B")]
            d0 = (per_ont(a) - per_ont(h)).dropna(); d = (per_ont(restrict(a)) - per_ont(restrict(h))).dropna(); lo, hi = boot(d.values)
            print(f"      {fs} {label:22s}: all {d0.mean():+.3f} -> restricted {d.mean():+.3f}, CI {lo:+.3f}..{hi:+.3f}, wins {int((d > 0).sum())}/{len(d)}")
            OUT.append(dict(part="newclass restricted", cutoff=C, foldset=fs, comparison=label, all=d0.mean(), mean=d.mean(), ci_lo=lo, ci_hi=hi, n=len(d), wins=int((d > 0).sum())))
pd.DataFrame(OUT).to_csv(os.path.join(paths.REPORTS, "dependence_audit.csv"), index=False)
