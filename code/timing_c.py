"""Study C part 3: runtime per test ontology at the 2023 cutoff (std folds). For each ontology: time to build the view
(names, TF-IDF, embeddings), per-query time for the ranker's feature extraction, L0 and S3 inference over the candidate set,
and the Study B GNN (regime R, seed 0) forward pass on the CPU and, if available, the GPU. 50 queries per ontology.
Needs results/study_c/models/h6_2023-07-01_std_<fold>.pkl (from ranker_c.py) and results/study_b/models/2023-07-01_std_<fold>_R_s0.pt
(Study B). Writes results/study_c/timing.csv. HermiT classification time is measured by code/prepare/hermit_timing.py."""
import os, sys, time, pickle, lzma, numpy as np, pandas as pd, torch
import lib
from lib import DATA, view_of, graph, NBF, ROOT
import ranker_c_funcs as rc
C = "2023-07-01"; rows = []; torch.set_num_threads(1)
for k, F in enumerate(DATA["folds"][C]["std"]):
    mp = os.path.join(ROOT, "results", "study_c", "models", f"h6_{C}_std_{k}.pkl"); gp = os.path.join(ROOT, "results", "study_b", "models", f"{C}_std_{k}_R_s0.pt")
    if not os.path.exists(mp): print("missing", mp); continue
    M = pickle.load(open(mp, "rb"))["models"]
    for n in F["test"]:
        Q = DATA["queries"][C][n]["A"][:50]
        if not Q: continue
        g = DATA["graphs"][C][n]; N0 = len(g["ids"]); t = time.time(); v = view_of(g); tv = time.time() - t
        RS = rc.Reasoner(v, rc.HIER[C][n]); q = np.array([v.ix[x["x"]] for x in Q]); t = time.time(); Xq, hyb, top = rc.scores(v, v.isa, v.rel, q, N0); tf = (time.time() - t) / len(q)
        tl = ts = 0.0
        for i in range(len(q)):
            R, anc = RS.feats(q[i], len(v.ids)); cset = np.array(list(dict.fromkeys(list(top[i]) + [y for y in anc if y < N0])))
            t = time.time(); M["L0"].predict(Xq[i, cset]); tl += time.time() - t
            t = time.time(); M["S3"].predict(np.concatenate([Xq[i, cset], R[cset]], 1)); ts += time.time() - t
        row = dict(ontology=n, classes=N0, view_s=tv, features_ms=tf * 1000, L0_ms=tl / len(q) * 1000, S3_ms=ts / len(q) * 1000)
        for dev in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            try:
                m = NBF(); m.load_state_dict(torch.load(gp, map_location="cpu")); m = m.to(dev).eval(); gr = lib.to_device(graph(v), dev); qq = torch.tensor(q[:25], device=dev)
                with torch.no_grad():
                    m(gr, qq, v.chans(q[:25]))
                    if dev == "cuda": torch.cuda.synchronize()
                    t = time.time(); m(gr, qq, v.chans(q[:25]))
                    if dev == "cuda": torch.cuda.synchronize()
                row[f"GNN_{dev}_ms"] = (time.time() - t) / 25 * 1000
            except Exception as e: row[f"GNN_{dev}_ms"] = None; print(n, dev, "GNN timing failed:", str(e)[:120])
        rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "study_c", "timing.csv"), index=False)
