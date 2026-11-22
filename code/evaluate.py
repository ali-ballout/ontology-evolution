"""Evaluate every trained GNN of one fold on its test ontologies (the shared query samples).
usage: evaluate.py CUT FOLDSET FOLD   writes results/study_b/ranks/<CUT>_<FOLDSET>_<FOLD>_<ontology>.csv.gz
Columns: cutoff, foldset, fold, ontology, task (A re-parenting / B new class), i (query index in the shared sample),
regime, seed, rank, kind, h (horizon bin), entailed, in_training (a target link is an is-a link anywhere in the fold's
training material: cutoff snapshots and history examples of the training ontologies)."""
import sys, os, glob, re, numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from lib import *
CUT, FS, FOLD = sys.argv[1], sys.argv[2], int(sys.argv[3]); F = DATA["folds"][CUT][FS][FOLD]; G = DATA["graphs"][CUT]
MODELS = sorted(glob.glob(os.path.join(ROOT, "results", "study_b", "models", f"{CUT}_{FS}_{FOLD}_*_s*.pt")))
TL = set()
for n in F["train"]:
    g = G[n]; TL |= {(g["ids"][a], g["ids"][b]) for a, b in g["isa"]}
    h = DATA["history"].get(CUT, {}).get(n)
    if h: TL |= {(a["x"], a["y"]) for a in h["A"]} | {(x, p) for x, b in h["B"].items() for p in b["parents"]}
OD = out_dir("ranks")
for n in F["test"]:
    out = os.path.join(OD, f"{CUT}_{FS}_{FOLD}_{n}.csv.gz")
    if os.path.exists(out): continue
    Q = DATA["queries"][CUT][n]; g = G[n]; N0 = len(g["ids"]); rows = []
    nets = {}
    for f in MODELS:
        reg, seed = re.search(r"_([A-Z]+)_s(\d+)\.pt$", f).groups(); m = NBF(); m.load_state_dict(torch.load(f)); nets[(reg, int(seed))] = m.eval()
    views = {}
    for mode in sorted({mode_of(r) for r, _ in nets}):
        v = view_of(g, [(t["x"], t["name"]) for t in Q["B"]], mode); views[mode] = (v, graph(v))
    for task in ("A", "B"):
        qs = Q[task]
        if not qs: continue
        for (reg, seed), m in nets.items():
            v, gr = views[mode_of(reg)]; q = np.array([v.ix[t["x"]] for t in qs])
            with torch.no_grad(): sc = torch.cat([m(gr, torch.tensor(q[s:s + 25]), v.chans(q[s:s + 25])) for s in range(0, len(q), 25)]).numpy()
            for i, t in enumerate(qs):
                if task == "A": tg = [v.ix[t["y"]]]; ex = [v.ix[p] for p in t["excl"] if p in v.ix]; links = [(t["x"], t["y"])]
                else: tg = [v.ix[p] for p in t["targets"] if p in v.ix]; ex = []; links = [(t["x"], p) for p in t["targets"]]
                rows.append(dict(cutoff=CUT, foldset=FS, fold=FOLD, ontology=n, task=task, i=i, regime=reg, seed=seed, rank=filtered_rank(sc[i], q[i], tg, ex, N0),
                                 kind=t.get("kind", "new"), h=t["h"], entailed=t.get("entailed"), in_training=any(l in TL for l in links)))
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False, compression="gzip"); os.replace(out + ".tmp", out); print(CUT, FS, FOLD, n, len(rows), flush=True)
