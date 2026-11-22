"""Aggregate the Study A training-free ranks (Table 3). Per view (cutoff x task/kind x entailment x horizon): per-ontology MRR / Hits@10 (ontologies
with >= 10 queries in the view), then the mean over ontologies (and the number of ontologies). Also: per-ontology paired
change from the 1-year to the 3+-year horizon. Writes reports/study_a_training_free.csv; prints the main tables.
usage: aggregate_training_free.py"""
import glob, os, sys, pandas as pd, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
R = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(paths.RESULTS, "study_a", "ranks", "*.csv")) if os.path.getsize(f) > 20], ignore_index=True)
R["rr"] = 1 / R["rank"]; R["h10"] = R["rank"] <= 10
MODELS = ["names only", "name vote", "structural vote", "hybrid", "reasoner: nearest inferred ancestor"]
def view(d, name):
    nq = d[d.model == "names only"].groupby("ontology").size(); keep = nq[nq >= 10].index; d = d[d.ontology.isin(keep)]
    if len(keep) < 3: return None
    po = d.groupby(["ontology", "model"])[["rr", "h10"]].mean().reset_index(); t = po.groupby("model")[["rr", "h10"]].mean()
    return t.assign(view=name, ontologies=len(keep), queries=int(nq[keep].sum())).reset_index()
out = []
for C, g in R.groupby("cutoff"):
    A, B = g[g.task == "A"], g[g.task == "B"]
    for name, d in [("A all", A), ("A entailed", A[A.entailed == True]), ("A not entailed", A[A.entailed == False]), ("A moved", A[A.kind == "moved"]), ("A added", A[A.kind == "added"]), ("B new class", B)]:
        for h in ["all", "1y", "2y", "3y+"]:
            t = view(d if h == "all" else d[d.h == h], f"{name} | {h}")
            if t is not None: out.append(t.assign(cutoff=C))
T = pd.concat(out); os.makedirs(paths.REPORTS, exist_ok=True); T.to_csv(os.path.join(paths.REPORTS, "study_a_training_free.csv"), index=False)
for C in sorted(T.cutoff.unique()):
    print(f"\n======== cutoff {C}")
    t = T[(T.cutoff == C) & T.model.isin(MODELS)].copy(); t["cell"] = t.rr.map("{:.3f}".format) + "/" + t.h10.map("{:.2f}".format)
    p = t.pivot_table(index="view", columns="model", values="cell", aggfunc="first")[[m for m in MODELS if m in t.model.unique()]]
    p["ont/q"] = t.drop_duplicates("view").set_index("view").apply(lambda r: f"{r.ontologies}/{r.queries}", axis=1)
    print(p.to_string())
print("\n== Paired change in hybrid and reasoner MRR from 1y to 3y+ (per ontology with >= 10 queries in both bins)")
for C, g in R.groupby("cutoff"):
    for task, d in [("A not entailed", g[(g.task == "A") & (g.entailed == False)]), ("A entailed", g[(g.task == "A") & (g.entailed == True)]), ("B", g[g.task == "B"])]:
        for m in ["hybrid", "names only"] + (["reasoner: nearest inferred ancestor"] if task.startswith("A") else []):
            x = d[d.model == m]; c = x.groupby(["ontology", "h"]).rr.agg(["mean", "size"]).reset_index(); c = c[c["size"] >= 10]
            w = c.pivot(index="ontology", columns="h", values="mean").dropna(subset=[k for k in ["1y", "3y+"] if k in c.h.unique()]) if {"1y", "3y+"} <= set(c.h) else None
            if w is None or len(w) < 3: continue
            dd = w["3y+"] - w["1y"]; print(f"{C} {task:15s} {m:38s}: {len(w)} ontologies, 3y+ minus 1y median {dd.median():+.3f}, lower in {int((dd < 0).sum())}")
