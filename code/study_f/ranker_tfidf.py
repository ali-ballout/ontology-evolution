"""Study F3: the Study B tuned ranker (deletion signal) and the hybrid heuristic with the sentence-encoder similarity set to
zero everywhere (TF-IDF-only pipeline), on the std folds of one cutoff.
usage: ranker_tfidf.py CUT FOLD   writes results/study_f/tfidf_ranks/<CUT>_std_<FOLD>_<ontology>__ranker.csv.gz (regime GBR_del)
       and results/study_f/tfidf_ranks/<CUT>_std_<FOLD>_<ontology>__hybrid.csv.gz"""
import sys, os, runpy, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import lib
_sims = lib.View.sims
def sims(s, q, named=None):
    t, e, nm = _sims(s, q, named); return t, np.zeros_like(e), nm
lib.View.sims = sims
OD = os.path.join(lib.ROOT, "results", "study_f", "tfidf_ranks"); os.makedirs(OD, exist_ok=True)
lib.out_dir = lambda *p: OD
lib.DATA = dict(lib.DATA, history={})   # deletion signal only
CUT, FOLD = sys.argv[1], int(sys.argv[2])
runpy.run_path(os.path.join(HERE, "..", "ranker.py"), run_name="__main__")
import ranker_c_funcs as RF
for n in lib.DATA["folds"][CUT]["std"][FOLD]["test"]:
    out = os.path.join(OD, f"{CUT}_std_{FOLD}_{n}__hybrid.csv.gz")
    if os.path.exists(out): continue
    Q = lib.DATA["queries"][CUT][n]; g = lib.DATA["graphs"][CUT][n]; N0 = len(g["ids"]); v = lib.view_of(g, [(t["x"], t["name"]) for t in Q["B"]]); rows = []
    for task in ("A", "B"):
        qs = Q[task]
        for s in range(0, len(qs), 25):
            part = qs[s:s + 25]; q = np.array([v.ix[t["x"]] for t in part]); _, hyb, _ = RF.scores(v, v.isa, v.rel, q, N0)
            for i, t in enumerate(part):
                if task == "A": tg = [v.ix[t["y"]]]; ex = [v.ix[p] for p in t["excl"] if p in v.ix]
                else: tg = [v.ix[p] for p in t["targets"] if p in v.ix]; ex = []
                rows.append(dict(cutoff=CUT, fold=FOLD, ontology=n, task=task, i=s + i, regime="hybrid", rank=lib.filtered_rank(hyb[i], q[i], tg, ex, N0), entailed=t.get("entailed")))
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False, compression="gzip"); os.replace(out + ".tmp", out)
