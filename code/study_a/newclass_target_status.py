"""For every sampled new-class query: how many of its target parents (parents at first appearance or in the current
release, existing at the cutoff) are parents of the class in the current release. Needs the raw snapshots.
usage: newclass_target_status.py   writes results/study_a/newclass_target_status.csv
(columns cutoff, ontology, i, targets, current: number of targets, number that are parents in the current release)"""
import os, sys, lzma, pickle, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths, queries as AQ
D = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read())); rows = []
for C in D["cuts"]:
    for n, Q in sorted(D["queries"][C].items()):
        if not Q["B"]: continue
        _, pcur, _ = AQ.load("cur", n)
        for i, t in enumerate(Q["B"]): rows.append(dict(cutoff=C, ontology=n, i=i, targets=len(t["targets"]), current=len(set(t["targets"]) & pcur.get(t["x"], set()))))
pd.DataFrame(rows).to_csv(os.path.join(paths.RESULTS, "study_a", "newclass_target_status.csv"), index=False)
