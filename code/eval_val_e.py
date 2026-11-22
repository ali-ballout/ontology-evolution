"""Validation for size selection: every SPLIT=val model of one fold on the held-out 20% of its training ontologies.
Queries: in each held-out ontology, 10% of is-a links hidden (rng 7 + position in the sorted held-out list), up to 100
hidden links; the child is the query, every class a candidate, other parents removed (filtered rank, ties half).
usage: eval_val_e.py CUT FOLD   writes results/study_e/val/<CUT>_std_<FOLD>.csv"""
import sys, os, glob, re, numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from elib import *
CUT, FOLD = sys.argv[1], int(sys.argv[2]); out = os.path.join(out_dir("val"), f"{CUT}_std_{FOLD}.csv")
VAL = sorted(split(CUT, FOLD)[0]); G = DATA["graphs"][CUT]; rows = []
MODELS = sorted(glob.glob(os.path.join(ROOT, "results", "study_e", "models", f"{CUT}_std_{FOLD}_L*d*_val_s*.pt")))
print(f"device {DEV}; {len(MODELS)} models; held-out ontologies {VAL}", flush=True)
nets = {}
for f in MODELS:
    L, D, s = map(int, re.search(r"_L(\d+)d(\d+)_val_s(\d+)\.pt$", f).groups()); m = NBFX(d=D, L=L); m.load_state_dict(torch.load(f, map_location="cpu")); nets[(L, D, s)] = m.to(DEV).eval()
for k, n in enumerate(VAL):
    v = view_of(G[n]); isa = v.isa; rng = np.random.default_rng(7 + k); msk = rng.random(len(isa)) < 0.10; obs, hid = isa[~msk], isa[msk]
    hid = hid[rng.choice(len(hid), min(100, len(hid)), replace=False)] if len(hid) else hid
    par = {}
    for a, b in isa: par.setdefault(a, set()).add(b)
    gr = on_dev(graph(v, obs)); q = hid[:, 0]; N0 = len(v.ids)
    for (L, D, s), m in nets.items():
        with torch.no_grad(): sc = torch.cat([m(gr, q[i:i + 25], v.chans(q[i:i + 25])) for i in range(0, len(q), 25)]).cpu().numpy()
        for i, (a, b) in enumerate(hid):
            rows.append(dict(cutoff=CUT, fold=FOLD, ontology=n, i=i, config=tag(L, D), L=L, d=D, seed=s, rank=filtered_rank(sc[i], a, [b], list(par[a] - {b}), N0)))
    print(n, len(hid), flush=True)
pd.DataFrame(rows).to_csv(out + ".tmp", index=False); os.replace(out + ".tmp", out); print("done", flush=True)
