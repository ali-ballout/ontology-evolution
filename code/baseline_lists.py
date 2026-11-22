"""Study D: top-20 lists from the training-free rankers (names only, hybrid, reasoner ranker R0) for every test query at
the three cutoffs, with the same features, exclusions and output format as export_d.py.
Writes results/study_d/lists/<CUT>_training_free_<ontology>.csv.gz.  usage: baseline_lists.py"""
import os
import numpy as np, pandas as pd
from lib import DATA, view_of, ROOT
import ranker_c_funcs as RC
OD = os.path.join(ROOT, "results", "study_d", "lists"); os.makedirs(OD, exist_ok=True)
for C in DATA["cuts"]:
    for n in DATA["test"][C]:
        out = f"{OD}/{C}_training_free_{n}.csv.gz"
        if os.path.exists(out): continue
        Q = DATA["queries"][C][n]; g = DATA["graphs"][C][n]; N0 = len(g["ids"]); v = view_of(g, [(t["x"], t["name"]) for t in Q["B"]]); RS = RC.Reasoner(v, RC.HIER[C][n])
        par = {}
        for a, b in g["isa"]: par.setdefault(int(a), set()).add(int(b))
        rows = []
        for task in ("A", "B"):
            qs = Q[task]
            for s0 in range(0, len(qs), 25):
                part = qs[s0:s0 + 25]; q = np.array([v.ix[t["x"]] for t in part]); Xq, hyb, top = RC.scores(v, v.isa, v.rel, q, N0)
                for i in range(len(part)):
                    bad = [q[i]] + list(par.get(int(q[i]), ()))
                    L = {"names only": Xq[i, :, 2].astype(np.float64), "hybrid": hyb[i].astype(np.float64)}
                    if task == "A":
                        _, anc = RS.feats(q[i], len(v.ids)); near = np.zeros(len(v.ids))
                        for y, d in anc.items():
                            if y < N0: near[y] = 1.0 / d
                        hs = hyb[i] / (np.abs(hyb[i][:N0]).max() + 1) * 0.5; L["R0"] = 10 * near + hs
                    for name, sc in L.items():
                        sc = sc.copy(); sc[N0:] = -np.inf; sc[bad] = -np.inf
                        rows.append(dict(cutoff=C, ontology=n, task=task, i=s0 + i, model=name, top20="|".join(v.ids[j] for j in np.argsort(-sc, kind="stable")[:20])))
        pd.DataFrame(rows).to_csv(out, index=False, compression="gzip"); print(C, n, len(rows), flush=True)
