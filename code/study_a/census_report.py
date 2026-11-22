"""Study A census report (Table 2 and the census statements in the text) from results/study_a/census.csv and
census_ontology.csv, with the families and test ontologies of data/studyB.pkl.xz. Writes reports/study_a_census.txt.
usage: census_report.py"""
import os, sys, io, lzma, pickle, contextlib, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
R = pd.read_csv(os.path.join(paths.RESULTS, "study_a", "census.csv")); O = pd.read_csv(os.path.join(paths.RESULTS, "study_a", "census_ontology.csv"))
D = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read()))
own = R[R.own == True]; CUTS = paths.CUTS
def tot(d, **k):
    for c, v in k.items(): d = d[d[c] == v]
    return int(d["count"].sum())
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    print("ontologies per cutoff:", {C: int((O.cutoff == C).sum()) for C in CUTS}, "| reasoner ok:", {C: int(((O.cutoff == C) & O.reasoner_ok).sum()) for C in CUTS})
    print("\n== Own-namespace edits after each cutoff (all first appearances; still present in current in brackets)")
    for C in CUTS:
        g = own[own.cutoff == C]; nw = g[g.kind.isin(["new_placeable", "new_outside"])]
        print(f"{C}: moved {tot(g, kind='moved')} ({tot(g, kind='moved', persists=True)}), added {tot(g, kind='added')} ({tot(g, kind='added', persists=True)}), "
              f"new classes {int(nw['count'].sum())} ({int(nw[nw.persists == True]['count'].sum())}), of which outside the task {tot(g, kind='new_outside') / nw['count'].sum():.1%}")
    print("\n== Entailed share of re-parenting edits (own), pooled and per ontology (ontologies with >= 10 edits of that kind)")
    for C in CUTS:
        g = own[(own.cutoff == C) & own.kind.isin(["moved", "added"]) & own.entailed.notna()]; per = {}
        for k in ["added", "moved"]:
            x = g[g.kind == k]; e = x[x.entailed == True]["count"].sum() / x["count"].sum()
            po = x.groupby("ontology").apply(lambda d: pd.Series(dict(n=d["count"].sum(), e=d[d.entailed == True]["count"].sum() / d["count"].sum())), include_groups=False)
            po = po[po.n >= 10]; per[k] = po.e
            print(f"{C} {k:6s}: pooled {e:.1%}; {len(po)} ontologies, median {po.e.median():.1%}, range {po.e.min():.0%}–{po.e.max():.0%}")
        both = pd.concat([per["moved"], per["added"]], axis=1, keys=["m", "a"]).dropna(); dd = both.m - both.a
        print(f"   paired moved − added entailed share: {len(both)} ontologies, moved higher in {int((dd > 0).sum())}, median difference {dd.median():+.1%}")
    print("\n== Horizon: share of own edits that appear within 1 year of the cutoff")
    for C in CUTS:
        g = own[own.cutoff == C]
        print(f"{C}: within 1 year — " + ", ".join(f"{k} {g[(g.kind == k) & (g.horizon <= 1.0)]['count'].sum() / g[g.kind == k]['count'].sum():.0%}" for k in ["moved", "added", "new_placeable"]))
    print("\n== Reverted later (own, first appeared but absent from the current release)")
    for C in CUTS:
        g = own[own.cutoff == C]
        print(f"{C}: " + ", ".join(f"{k} {g[(g.kind == k) & (g.persists == False)]['count'].sum() / g[g.kind == k]['count'].sum():.0%}" for k in ["added", "moved", "new_outside", "new_placeable"]))
    print("\n== Ontologies with test material (own: >= 10 re-parenting edits / >= 10 placeable new classes, still present); independent in brackets")
    for C in CUTS:
        g = own[(own.cutoff == C) & (own.persists == True)]; ind = set(D["families"][C]["independent"])
        a = g[g.kind.isin(["moved", "added"])].groupby("ontology")["count"].sum(); a = set(a[a >= 10].index)
        b = g[g.kind == "new_placeable"].groupby("ontology")["count"].sum(); b = set(b[b >= 10].index)
        print(f"{C}: re-parenting {len(a)} ({len(a & ind)}), new classes {len(b)} ({len(b & ind)})")
    print("\n== GO's weight in re-parenting edits, and entailed share without GO")
    for C in CUTS:
        g = own[(own.cutoff == C) & own.kind.isin(["moved", "added"])]; go = g[g.ontology == "go"]; ng = g[g.ontology != "go"]
        es = lambda d, k: d[(d.kind == k) & (d.entailed == True)]["count"].sum() / d[(d.kind == k) & d.entailed.notna()]["count"].sum()
        print(f"{C}: GO {go['count'].sum() / g['count'].sum():.0%} of own re-parenting edits; without GO entailed share moved {es(ng, 'moved'):.0%}, added {es(ng, 'added'):.0%}; GO alone moved {es(go, 'moved'):.0%}")
    print("\n== Families at each cutoff (0.5% class / is-a overlap, connected components)")
    for C in CUTS: print(f"{C}: block {len(D['families'][C]['block'])}, independent {len(D['families'][C]['independent'])}")
os.makedirs(paths.REPORTS, exist_ok=True); open(os.path.join(paths.REPORTS, "study_a_census.txt"), "w").write(buf.getvalue()); print(buf.getvalue())
