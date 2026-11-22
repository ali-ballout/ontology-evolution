"""Tuned non-neural ranker (Study B, H5): gradient-boosted learning-to-rank that reranks the hybrid heuristic's top K=300
candidates, using the same inputs as the GNN: name similarity (TF-IDF, MiniLM, names-only score, look-alike name vote),
the 18 structural features (feats.py), the hybrid score and the candidate's hybrid rank, plus each relative to the query's
best candidate. Candidates outside the top K keep the hybrid order below the reranked ones.
Training: the fold's training ontologies only. Signals: 'GBR_del' (up to 100 random-deletion queries per ontology) and,
where the cutoff has histories, 'GBR_delhist' (plus up to 100 edit-history queries per ontology).
Tuning: 20% of training ontologies held out (fixed rng); grid of 8 settings; best validation MRR; refit on all.
LightGBM LambdaRank if installed, otherwise scikit-learn HistGradientBoostingClassifier (pointwise), reported in the log.
usage: ranker.py CUT FOLD   (std folds)   writes results/study_b/ranks/<CUT>_std_<FOLD>_<ontology>__ranker.csv.gz"""
import sys, os, itertools, numpy as np, pandas as pd, scipy.sparse as sp
from lib import DATA, view_of, z, filtered_rank, out_dir
import feats as FT
try:
    import lightgbm as lgb; LGB = True
except ImportError: LGB = False
from sklearn.ensemble import HistGradientBoostingClassifier
CUT, FOLD = sys.argv[1], int(sys.argv[2]); F = DATA["folds"][CUT]["std"][FOLD]; G = DATA["graphs"][CUT]; K = 300; OD = out_dir("ranks")
if all(os.path.exists(os.path.join(OD, f"{CUT}_std_{FOLD}_{n}__ranker.csv.gz")) for n in F["test"]): sys.exit(0)
def scores(v, isa, rel, q, N0):
    St, Se, _ = v.sims(q); S2 = St + Se; S2[np.arange(len(q)), q] = 0; S2[:, N0:] = 0
    P = sp.csr_matrix((np.ones(len(isa)), (isa[:, 0], isa[:, 1])), shape=(len(v.ids),) * 2)
    Sk = S2.copy(); th = -np.sort(-Sk, 1)[:, 9:10]; Sk[(Sk < th) | (Sk <= 0)] = 0; nv = np.asarray((sp.csr_matrix(Sk) @ P).todense())
    M = FT.mats(dict(ids=v.ids, rel=[(a, 0, b) for a, b in rel]), isa); S = FT.feats(M, q); hyb = z(nv) + z(S2) + 0.25 * z(S[:, :, 4])
    hyb[:, N0:] = -1e9; hyb[np.arange(len(q)), q] = -1e9
    order = np.argsort(-hyb, 1); hr = np.empty_like(order); hr[np.arange(len(q))[:, None], order] = np.arange(hyb.shape[1])[None, :]
    nf = np.stack([St, Se, S2, nv], 2); nf = np.concatenate([nf, nf / (nf.max(1, keepdims=True) + 1e-9)], 2)
    X = np.concatenate([nf, S, hyb[:, :, None], np.log1p(hr)[:, :, None]], 2).astype(np.float32)
    return X, hyb, order[:, :K]
def examples(n, kind, rng):
    """yields (view, isa, rel, N0, [(query, positive, other parents)])"""
    if kind == "del":
        g = G[n]; v = view_of(g); isa = v.isa; m = rng.random(len(isa)) < 0.10; obs, hid = isa[~m], isa[m]
        hid = hid[rng.choice(len(hid), min(100, len(hid)), replace=False)] if len(hid) else hid; par = {}
        for a, b in isa: par.setdefault(a, set()).add(b)
        return v, obs, v.rel, len(g["ids"]), [(a, b, par[a] - {b}) for a, b in hid]
    h = DATA["history"][CUT][n]; g1 = DATA["graphs"][h["r1"]][n]; v = view_of(g1, [(x, b["name"]) for x, b in h["B"].items()])
    ex = [(a["x"], a["y"]) for a in h["A"]] + [(x, p) for x, b in h["B"].items() for p in b["parents"]]; ex = [ex[i] for i in rng.choice(len(ex), min(100, len(ex)), replace=False)]
    return v, v.isa, v.rel, len(g1["ids"]), [(v.ix[c], v.ix[p], {v.ix[x] for x in h["par"].get(c, []) if x in v.ix} - {v.ix[p]}) for c, p in ex]
def build(onts, kinds, seed):
    X, y, grp, rr = [], [], [], []; rng = np.random.default_rng(seed)
    for n in onts:
        for kind in kinds:
            if kind == "hist" and n not in DATA["history"].get(CUT, {}): continue
            v, isa, rel, N0, ex = examples(n, kind, rng)
            for s in range(0, len(ex), 25):
                part = ex[s:s + 25]; q = np.array([a for a, _, _ in part]); Xq, hyb, top = scores(v, isa, rel, q, N0)
                for i, (a, pos, oth) in enumerate(part):
                    cand = [c for c in top[i] if c not in oth and c != pos][:K - 1] + [pos]
                    X.append(Xq[i, cand]); y.append((np.array(cand) == pos).astype(int)); grp.append(len(cand)); rr.append(int(np.sum(hyb[i] > hyb[i, pos])) + 1)
    return X, y, grp, rr
GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne) for nl, lr, ne in itertools.product([15, 63], [0.05, 0.1], [200, 500])]
def fit(X, y, grp, p):
    Xa, ya = np.concatenate(X), np.concatenate(y)
    if LGB: return lgb.LGBMRanker(objective="lambdarank", num_leaves=p["num_leaves"], learning_rate=p["learning_rate"], n_estimators=p["n_estimators"], n_jobs=2, verbose=-1, random_state=0).fit(Xa, ya, group=grp)
    return HistGradientBoostingClassifier(max_leaf_nodes=p["num_leaves"], learning_rate=p["learning_rate"], max_iter=p["n_estimators"], random_state=0).fit(Xa, ya)
def predict(m, Xc): return m.predict(Xc) if LGB else m.predict_proba(Xc)[:, 1]
def val_mrr(m, X, y): return float(np.mean([1.0 / (1 + np.sum(predict(m, Xi) > predict(m, Xi)[yi.argmax()])) for Xi, yi in zip(X, y)]))
SIGNALS = [("GBR_del", ["del"])] + ([("GBR_delhist", ["del", "hist"])] if DATA["history"].get(CUT) else [])
rng = np.random.default_rng(100 + FOLD); tr = list(F["train"]); rng.shuffle(tr); nval = max(2, len(tr) // 5); VAL, FIT = tr[:nval], tr[nval:]
if os.environ.get("SMOKE"): VAL, FIT, GRID = VAL[:1], FIT[:2], GRID[:1]; F = dict(F, test=F["test"][:1])   # quick functional test only
MODELS = {}
for name, kinds in SIGNALS:
    Xf, yf, gf, _ = build(FIT, kinds, 1); Xv, yv, _, _ = build(VAL, kinds, 2)
    res = [(val_mrr(fit(Xf, yf, gf, p), Xv, yv), i) for i, p in enumerate(GRID)]; best = GRID[max(res)[1]]
    print(CUT, FOLD, name, "LightGBM" if LGB else "sklearn", "validation MRR by setting:", [round(r, 3) for r, _ in res], "best", best, flush=True)
    Xa, ya, ga, _ = build(FIT + VAL, kinds, 3); MODELS[name] = fit(Xa, ya, ga, best)
for n in F["test"]:
    out = os.path.join(OD, f"{CUT}_std_{FOLD}_{n}__ranker.csv.gz")
    if os.path.exists(out): continue
    Q = DATA["queries"][CUT][n]; g = G[n]; N0 = len(g["ids"]); v = view_of(g, [(t["x"], t["name"]) for t in Q["B"]]); rows = []
    for task in ("A", "B"):
        qs = Q[task]
        for s in range(0, len(qs), 25):
            part = qs[s:s + 25]; q = np.array([v.ix[t["x"]] for t in part]); Xq, hyb, top = scores(v, v.isa, v.rel, q, N0)
            for i, t in enumerate(part):
                if task == "A": tg = [v.ix[t["y"]]]; ex = [v.ix[p] for p in t["excl"] if p in v.ix]
                else: tg = [v.ix[p] for p in t["targets"] if p in v.ix]; ex = []
                for name, m in MODELS.items():
                    sc = hyb[i].astype(np.float64).copy(); sc[top[i]] = 1e6 + predict(m, Xq[i, top[i]])
                    rows.append(dict(cutoff=CUT, foldset="std", fold=FOLD, ontology=n, task=task, i=s + i, regime=name, seed=0, rank=filtered_rank(sc, q[i], tg, ex, N0), kind=t.get("kind", "new"), h=t["h"], entailed=t.get("entailed"), in_training=None))
    pd.DataFrame(rows).to_csv(out + ".tmp", index=False, compression="gzip"); os.replace(out + ".tmp", out); print(CUT, FOLD, n, len(rows), flush=True)
