"""Study F2: name search with three name signals on the shared future-edit queries at every cutoff: character n-gram TF-IDF
only, sentence encoder (MiniLM) only, and their sum (the name search of the paper; checked against the Study A ranks).
usage: encoder_check.py   writes results/study_f/encoder_ranks.csv.gz"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lib import DATA, ROOT, view_of, filtered_rank
rows = []
for C in DATA["cuts"]:
    for n, Q in sorted(DATA["queries"][C].items()):
        g = DATA["graphs"][C][n]; N0 = len(g["ids"]); v = view_of(g, [(t["x"], t["name"]) for t in Q["B"]])
        for task in ("A", "B"):
            qs = Q[task]
            for s0 in range(0, len(qs), 50):
                part = qs[s0:s0 + 50]; q = np.array([v.ix[t["x"]] for t in part]); St, Se, _ = v.sims(q)
                for k, S in (("tfidf", St), ("encoder", Se), ("sum", St + Se)):
                    for j, t in enumerate(part):
                        if task == "A": tg = [v.ix[t["y"]]]; ex = [v.ix[p] for p in t["excl"] if p in v.ix and p != t["y"]]; ent = t["entailed"]
                        else: tg = [v.ix[p] for p in t["targets"] if p in v.ix]; ex = []; ent = None
                        rows.append(dict(cutoff=C, ontology=n, task=task, i=s0 + j, signal=k, rank=filtered_rank(S[j], q[j], tg, ex, N0), entailed=ent))
        print(C, n, flush=True)
od = os.path.join(ROOT, "results", "study_f"); os.makedirs(od, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(od, "encoder_ranks.csv.gz"), index=False, compression="gzip")
