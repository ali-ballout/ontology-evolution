"""Test evaluation: every SPLIT=full model of one fold on its test ontologies, with the shared Study A queries and the
filtered ranks of evaluate.py (re-parenting: other parents removed; new class: best of all targets).
usage: eval_test_e.py CUT FOLD   writes results/study_e/ranks/<CUT>_std_<FOLD>_<ontology>.csv.gz"""
import sys, os, glob, re, numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from elib import *
CUT, FOLD = sys.argv[1], int(sys.argv[2]); F = DATA["folds"][CUT]["std"][FOLD]; G = DATA["graphs"][CUT]; OD = out_dir("ranks")
MODELS = sorted(glob.glob(os.path.join(ROOT, "results", "study_e", "models", f"{CUT}_std_{FOLD}_L*d*_full_s*.pt")))
print(f"device {DEV}; {len(MODELS)} models", flush=True)
nets = {}
for f in MODELS:
    L, D, s = map(int, re.search(r"_L(\d+)d(\d+)_full_s(\d+)\.pt$", f).groups()); m = NBFX(d=D, L=L); m.load_state_dict(torch.load(f, map_location="cpu")); nets[(L, D, s)] = m.to(DEV).eval()
for n in F["test"]:
    out = os.path.join(OD, f"{CUT}_std_{FOLD}_{n}.csv.gz")
    if os.path.exists(out): continue
    Q = DATA["queries"][CUT][n]; g = G[n]; N0 = len(g["ids"]); v = view_of(g, [(t["x"], t["name"]) for t in Q["B"]]); gr = on_dev(graph(v)); rows = []
    for task in ("A", "B"):
        qs = Q[task]
        if not qs: continue
        q = np.array([v.ix[t["x"]] for t in qs])
        for (L, D, s), m in nets.items():
            with torch.no_grad(): sc = torch.cat([m(gr, q[i:i + 25], v.chans(q[i:i + 25])) for i in range(0, len(q), 25)]).cpu().numpy()
            for i, t in enumerate(qs):
                if task == "A": tg = [v.ix[t["y"]]]; ex = [v.ix[p] for p in t["excl"] if p in v.ix]
                else: tg = [v.ix[p] for p in t["targets"] if p in v.ix]; ex = []
                rows.append(dict(cutoff=CUT, fold=FOLD, ontology=n, task=task, i=i, config=tag(L, D), L=L, d=D, seed=s, rank=filtered_rank(sc[i], q[i], tg, ex, N0),
                                 kind=t.get("kind", "new"), h=t["h"], entailed=t.get("entailed")))
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False, compression="gzip"); os.replace(out + ".tmp", out); print(n, len(rows), flush=True)
