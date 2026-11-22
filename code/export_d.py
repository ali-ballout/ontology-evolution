"""Study D: export top-20 suggestion lists for one cutoff and std fold. usage: export_d.py CUT FOLD
Models: the tuned ranker GBR_del_c (results/study_c/models/h1_<CUT>_std_<FOLD>.pkl, both tasks; candidates = hybrid top 300,
as trained), the history-trained ranker L0 (results/study_c/models/h6_<CUT>_std_<FOLD>.pkl if present; re-parenting;
candidates = hybrid top 300 plus inferred ancestors), and the Study B GNN (results/study_b/models/<CUT>_std_<FOLD>_R_s0.pt),
all run on the CPU.
Lists never contain the query, its parents at the cutoff, or classes absent at the cutoff.
Writes results/study_d/lists/<CUT>_std_<FOLD>_<ontology>.csv.gz (cutoff, ontology, task, i, model, top20 as '|'-joined class ids)."""
import os, sys
import pickle, numpy as np, pandas as pd, torch
import lib
from lib import DATA, view_of, graph, NBF, ROOT
from ranker_c_funcs import scores, Reasoner, HIER
torch.set_num_threads(2)
C, K_ = sys.argv[1], int(sys.argv[2]); F = DATA["folds"][C]["std"][K_]; G = DATA["graphs"][C]
OD = os.path.join(ROOT, "results", "study_d", "lists"); os.makedirs(OD, exist_ok=True)
M1 = pickle.load(open(os.path.join(ROOT, "results", "study_c", "models", f"h1_{C}_std_{K_}.pkl"), "rb"))["models"]["GBR_del_c"]
p6 = os.path.join(ROOT, "results", "study_c", "models", f"h6_{C}_std_{K_}.pkl"); M6 = pickle.load(open(p6, "rb"))["models"]["L0"] if os.path.exists(p6) else None
gp = os.path.join(ROOT, "results", "study_b", "models", f"{C}_std_{K_}_R_s0.pt"); net = None
if os.path.exists(gp):
    net = NBF(); net.load_state_dict(torch.load(gp, map_location="cpu")); net = net.eval()
for n in F["test"]:
    out = os.path.join(OD, f"{C}_std_{K_}_{n}.csv.gz")
    if os.path.exists(out): continue
    Q = DATA["queries"][C][n]; g = G[n]; N0 = len(g["ids"]); v = view_of(g, [(t["x"], t["name"]) for t in Q["B"]]); RS = Reasoner(v, HIER[C][n])
    par = {}
    for a, b in g["isa"]: par.setdefault(int(a), set()).add(int(b))
    gr = graph(v) if net is not None else None; rows = []
    for task in ("A", "B"):
        qs = Q[task]
        for s0 in range(0, len(qs), 25):
            part = qs[s0:s0 + 25]; q = np.array([v.ix[t["x"]] for t in part]); Xq, hyb, top = scores(v, v.isa, v.rel, q, N0)
            gs = None
            if net is not None:
                with torch.no_grad(): gs = net(gr, torch.tensor(q), v.chans(q)).detach().cpu().numpy()
            for i, t in enumerate(part):
                bad = [q[i]] + list(par.get(int(q[i]), ()))
                lists = {}
                sc = hyb[i].astype(np.float64).copy(); sc[top[i]] = 1e6 + M1.predict(Xq[i, top[i]]); lists["GBR_del_c"] = sc
                if task == "A" and M6 is not None:
                    R, anc = RS.feats(q[i], len(v.ids)); cset = np.array(list(dict.fromkeys(list(top[i]) + [y for y in anc if y < N0])))
                    sc = hyb[i].astype(np.float64).copy(); sc[cset] = 1e6 + M6.predict(Xq[i, cset]); lists["L0"] = sc
                if gs is not None: lists["GNN_R"] = gs[i].astype(np.float64)
                for name, sc in lists.items():
                    sc = sc.copy(); sc[N0:] = -np.inf; sc[bad] = -np.inf
                    rows.append(dict(cutoff=C, ontology=n, task=task, i=s0 + i, model=name, top20="|".join(v.ids[j] for j in np.argsort(-sc, kind="stable")[:20])))
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False, compression="gzip"); os.replace(out + ".tmp", out); print(C, K_, n, len(rows), flush=True)
