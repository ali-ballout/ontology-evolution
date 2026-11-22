"""Study F1: score the Study B deletion-trained GNNs (regime R, std folds, 3 seeds), name search and the hybrid heuristic
on the fixed deletion queries (results/study_f/deletion_queries.pkl.xz). The view is the cutoff snapshot with the hidden
is-a links removed; candidates are all classes; the child's other told parents are removed (filtered rank, ties half).
usage: score_deletion.py CUT FOLD [MODELDIR]   (default results/study_b/models)   writes results/study_f/deletion_ranks/<CUT>_<FOLD>_<ontology>.csv.gz
in_training: the hidden link is an is-a link in the fold's training material (as in Study B's evaluation)."""
import sys, os, glob, re, lzma, pickle, numpy as np, pandas as pd, torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lib import DATA, ROOT, NBF, view_of, graph, filtered_rank
import ranker_c_funcs as RF
CUT, FOLD = sys.argv[1], int(sys.argv[2]); MD = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "study_b", "models"); F = DATA["folds"][CUT]["std"][FOLD]; G = DATA["graphs"][CUT]
Q = pickle.loads(lzma.decompress(open(os.path.join(ROOT, "results", "study_f", "deletion_queries.pkl.xz"), "rb").read()))[CUT]
nets = {}
for f in sorted(glob.glob(os.path.join(MD, f"{CUT}_std_{FOLD}_R_s*.pt"))):
    m = NBF(); m.load_state_dict(torch.load(f, map_location="cpu")); nets[int(re.search(r"_s(\d+)\.pt$", f).group(1))] = m.eval()
assert len(nets) == 3, nets.keys()
TL = set()
for n in F["train"]:
    g = G[n]; TL |= {(g["ids"][a], g["ids"][b]) for a, b in g["isa"]}
    h = DATA["history"].get(CUT, {}).get(n)
    if h: TL |= {(a["x"], a["y"]) for a in h["A"]} | {(x, p) for x, b in h["B"].items() for p in b["parents"]}
OD = os.path.join(ROOT, "results", "study_f", "deletion_ranks"); os.makedirs(OD, exist_ok=True)
for n in F["test"]:
    out = os.path.join(OD, f"{CUT}_{FOLD}_{n}.csv.gz")
    if os.path.exists(out): continue
    g = G[n]; isa = np.asarray(g["isa"], dtype=int).reshape(-1, 2); hid = Q[n]["hidden"]; qi = Q[n]["queries"]; N = len(g["ids"])
    v = view_of(dict(ids=g["ids"], names=g["names"], isa=isa[~hid].tolist(), rel=g["rel"])); gr = graph(v)
    par = {}
    for a, b in isa: par.setdefault(int(a), set()).add(int(b))
    x = isa[qi, 0]; y = isa[qi, 1]; rows = []; obs = isa[~hid]; rel = np.asarray(g["rel"], dtype=int).reshape(-1, 2)
    SC = {}
    for s0 in range(0, len(x), 25):
        q = x[s0:s0 + 25]; St, Se, _ = v.sims(q); S2 = St + Se
        _, hyb, _ = RF.scores(v, obs, rel, q, N)
        SC.setdefault("names only", []).append(S2); SC.setdefault("hybrid", []).append(hyb)
        with torch.no_grad():
            for s, m in nets.items(): SC.setdefault(f"GNN s{s}", []).append(m(gr, torch.tensor(q), v.chans(q)).numpy())
    for k in SC: SC[k] = np.concatenate(SC[k])
    for i in range(len(x)):
        ex = [p for p in par[int(x[i])] if p != y[i]]; it = (g["ids"][x[i]], g["ids"][y[i]]) in TL
        for k, sc in SC.items(): rows.append(dict(cutoff=CUT, fold=FOLD, ontology=n, i=i, model=k, rank=filtered_rank(sc[i], int(x[i]), [int(y[i])], ex, N), in_training=it))
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False, compression="gzip"); os.replace(out + ".tmp", out); print(CUT, FOLD, n, len(x), flush=True)
