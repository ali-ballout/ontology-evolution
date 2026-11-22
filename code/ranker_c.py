"""Study C ranker jobs (see protocols/study_c.md). usage: ranker_c.py MODE CUT FOLDSET FOLD
  MODE h1  tuned ranker, deletion signal ('GBR_del_c'), on any fold set (family-held-out folds for H1); both tasks.
  MODE h6  std folds at cutoffs with histories; re-parenting queries (task A). Models, all on the same examples and the same
           candidate set (hybrid top 300 plus every inferred ancestor of the query):
             L0d  deletion signal            L0  deletion + history          S3  deletion + history + reasoner features
           and strategies derived from L0's scores and the reasoner signal:
             S1 reasoner-first (inferred ancestors nearest first, ties by L0, then L0's order)
             S2 L0 with inferred descendants removed (would create a cycle)
             S4 calibrated fusion z(L0) + w x (1 / inferred distance) on inferred ancestors; w tuned on held-out training
                ontologies' history edits
Reasoner features per candidate: inferred ancestor of the query (0/1), 1 / inferred distance, inferred descendant (0/1).
  test and history examples: HermiT classification of the visible snapshot (cutoff snapshot / older snapshot r1);
  deletion examples: transitive closure of the observed is-a graph after hiding (the hidden link cannot leak).
Tuning: 20% of the fold's training ontologies held out (fixed rng), grid of 27 settings (Study B grid extended to simpler
settings), best validation MRR, refit on fit + validation examples. Pre-cutoff data only.
Writes results/study_c/ranks/<MODE>_<CUT>_<FOLDSET>_<FOLD>_<ontology>.csv.gz (or into RANKS_DIR if set), models in
results/study_c/models/, log on stdout."""
import sys, os, time, itertools, pickle, lzma, collections, numpy as np, pandas as pd, scipy.sparse as sp
from lib import DATA, view_of, z, filtered_rank, ROOT
import feats as FT
try:
    import lightgbm as lgb; LGB = True
except ImportError: LGB = False
from sklearn.ensemble import HistGradientBoostingClassifier
from ranker_c_funcs import HIER, scores, Reasoner, told_up
MODE, CUT, FS, FOLD = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); K = 300
F = DATA["folds"][CUT][FS][FOLD]; G = DATA["graphs"][CUT]
OD = os.environ.get("RANKS_DIR", os.path.join(ROOT, "results", "study_c", "ranks")); MD = os.path.join(ROOT, "results", "study_c", "models")
for d in (OD, MD): os.makedirs(d, exist_ok=True)
outp = lambda n: os.path.join(OD, f"{MODE}_{CUT}_{FS}_{FOLD}_{n}.csv.gz")
if all(os.path.exists(outp(n)) for n in F["test"]): sys.exit(0)
GRID = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne) for nl, lr, ne in itertools.product([7, 15, 63], [0.02, 0.05, 0.1], [100, 200, 500])]
WGRID = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 1e3]
def examples(n, kind, rng):
    """(view, isa, rel, N0, reasoner, [(query, positive, other true parents)])"""
    if kind == "del":
        g = G[n]; v = view_of(g); isa = v.isa; m = rng.random(len(isa)) < 0.10; obs, hid = isa[~m], isa[m]
        hid = hid[rng.choice(len(hid), min(100, len(hid)), replace=False)] if len(hid) else hid; par = {}
        for a, b in isa: par.setdefault(a, set()).add(b)
        return v, obs, v.rel, len(g["ids"]), Reasoner(v, told_up(v, obs)), [(a, b, par[a] - {b}) for a, b in hid]
    h = DATA["history"][CUT][n]; g1 = DATA["graphs"][h["r1"]][n]; v = view_of(g1, [(x, b["name"]) for x, b in h["B"].items()])
    ex = [(a["x"], a["y"]) for a in h["A"]] + [(x, p) for x, b in h["B"].items() for p in b["parents"]]; ex = [ex[i] for i in rng.choice(len(ex), min(100, len(ex)), replace=False)]
    return v, v.isa, v.rel, len(g1["ids"]), Reasoner(v, HIER[h["r1"]][n]), [(v.ix[c], v.ix[p], {v.ix[x] for x in h["par"].get(c, []) if x in v.ix} - {v.ix[p]}) for c, p in ex]
def build(onts, kinds, seed, with_anc):
    """per training query: base features, reasoner features, labels over its candidates"""
    out = []; rng = np.random.default_rng(seed)
    for n in onts:
        for kind in kinds:
            if kind == "hist" and n not in DATA["history"].get(CUT, {}): continue
            v, isa, rel, N0, RS, ex = examples(n, kind, rng)
            for s0 in range(0, len(ex), 25):
                part = ex[s0:s0 + 25]; q = np.array([a for a, _, _ in part]); Xq, hyb, top = scores(v, isa, rel, q, N0)
                for i, (a, pos, oth) in enumerate(part):
                    R, anc = RS.feats(a, len(v.ids)); cset = list(dict.fromkeys(list(top[i]) + ([y for y in anc if y < N0] if with_anc else [])))
                    cand = [c for c in cset if c not in oth and c != pos] + [pos]
                    out.append(dict(X=Xq[i, cand], R=R[cand], y=(np.array(cand) == pos).astype(int), kind=kind))
    return out
def fit(rows, use_R, p):
    X = np.concatenate([np.concatenate([r["X"], r["R"]], 1) if use_R else r["X"] for r in rows]); y = np.concatenate([r["y"] for r in rows]); grp = [len(r["y"]) for r in rows]
    if LGB: return lgb.LGBMRanker(objective="lambdarank", num_leaves=p["num_leaves"], learning_rate=p["learning_rate"], n_estimators=p["n_estimators"], n_jobs=2, verbose=-1, random_state=0).fit(X, y, group=grp)
    return HistGradientBoostingClassifier(max_leaf_nodes=p["num_leaves"], learning_rate=p["learning_rate"], max_iter=p["n_estimators"], random_state=0).fit(X, y)
def pred(m, X): return m.predict(X) if LGB else m.predict_proba(X)[:, 1]
def mrr(m, rows, use_R):
    rr = []
    for r in rows:
        s = pred(m, np.concatenate([r["X"], r["R"]], 1) if use_R else r["X"]); rr.append(1.0 / (1 + np.sum(s > s[r["y"].argmax()])))
    return float(np.mean(rr))
def tune(name, FITR, VALR, use_R):
    t = time.time(); res = [(mrr(fit(FITR, use_R, p), VALR, use_R), i) for i, p in enumerate(GRID)]; best = GRID[max(res)[1]]
    print(f"{MODE} {CUT} {FS} {FOLD} {name} {'LightGBM' if LGB else 'sklearn'} best {best} validation MRR {max(res)[0]:.3f} (grid {min(res)[0]:.3f}..{max(res)[0]:.3f}) {time.time() - t:.0f}s", flush=True)
    return best
rng = np.random.default_rng(100 + FOLD); tr = list(F["train"]); rng.shuffle(tr); nval = max(2, len(tr) // 5); VAL, FIT = tr[:nval], tr[nval:]
if os.environ.get("SMOKE"): VAL, FIT, GRID, F = VAL[:1], FIT[:2], GRID[:2], dict(F, test=F["test"][:1])
MODELS, WBEST = {}, None
if MODE == "h1":
    FR, VR = build(FIT, ["del"], 1, False), build(VAL, ["del"], 2, False); best = tune("GBR_del_c", FR, VR, False); MODELS["GBR_del_c"] = (fit(FR + VR, False, best), False)
else:
    FRd, VRd = build(FIT, ["del"], 1, True), build(VAL, ["del"], 2, True); FRh, VRh = build(FIT, ["hist"], 4, True), build(VAL, ["hist"], 5, True)
    b = tune("L0d", FRd, VRd, False); MODELS["L0d"] = (fit(FRd + VRd, False, b), False)
    b = tune("L0", FRd + FRh, VRd + VRh, False); L0fit = fit(FRd + FRh, False, b); MODELS["L0"] = (fit(FRd + FRh + VRd + VRh, False, b), False)
    b = tune("S3", FRd + FRh, VRd + VRh, True); MODELS["S3"] = (fit(FRd + FRh + VRd + VRh, True, b), True)
    def fused(rows, w):
        rr = []
        for r in rows:
            s = pred(L0fit, r["X"]); s = (s - s.mean()) / (s.std() + 1e-9) + w * r["R"][:, 0] * r["R"][:, 1]; rr.append(1.0 / (1 + np.sum(s > s[r["y"].argmax()])))
        return float(np.mean(rr))
    wres = [(fused(VRh, w), w) for w in WGRID]; WBEST = max(wres)[1]
    print(f"{MODE} {CUT} {FS} {FOLD} S4 fusion weight by validation MRR on history edits: {[(w, round(m, 3)) for m, w in wres]} -> w = {WBEST}", flush=True)
pickle.dump(dict(models={k: m for k, (m, _) in MODELS.items()}, w=WBEST, lgb=LGB), open(os.path.join(MD, f"{MODE}_{CUT}_{FS}_{FOLD}.pkl"), "wb"))
for n in F["test"]:
    if os.path.exists(outp(n)): continue
    Q = DATA["queries"][CUT][n]; g = G[n]; N0 = len(g["ids"]); v = view_of(g, [(t["x"], t["name"]) for t in Q["B"]]); RS = Reasoner(v, HIER[CUT][n]); rows = []; t0 = time.time(); nq = 0
    for task in (("A", "B") if MODE == "h1" else ("A",)):
        qs = Q[task]
        for s0 in range(0, len(qs), 25):
            part = qs[s0:s0 + 25]; q = np.array([v.ix[t["x"]] for t in part]); Xq, hyb, top = scores(v, v.isa, v.rel, q, N0)
            for i, t in enumerate(part):
                nq += 1
                if task == "A": tg = [v.ix[t["y"]]]; ex = [v.ix[p] for p in t["excl"] if p in v.ix]
                else: tg = [v.ix[p] for p in t["targets"] if p in v.ix]; ex = []
                R, anc = RS.feats(q[i], len(v.ids)) if task == "A" else (np.zeros((len(v.ids), 3), np.float32), {})
                cset = np.array(list(dict.fromkeys(list(top[i]) + ([y for y in anc if y < N0] if MODE == "h6" else []))))
                covered = any(x in set(cset.tolist()) for x in tg); comp = {}
                for name, (m, useR) in MODELS.items():
                    comp[name] = pred(m, np.concatenate([Xq[i, cset], R[cset]], 1) if useR else Xq[i, cset])
                if MODE == "h6":
                    sL = comp["L0"]; rk = np.argsort(np.argsort(sL)) / len(sL); isanc, inv = R[cset, 0], R[cset, 1]
                    dist = np.where(isanc > 0, 1.0 / np.maximum(inv, 1e-9), 0)
                    comp["S1"] = rk + isanc * (1000 - dist); comp["S2"] = sL.copy(); comp["S4"] = (sL - sL.mean()) / (sL.std() + 1e-9) + WBEST * isanc * inv
                for name, vsc in comp.items():
                    sc = hyb[i].astype(np.float64).copy(); sc[cset] = 1e6 + vsc
                    if name == "S2": sc[R[:, 2] > 0] = -1e12
                    rows.append(dict(cutoff=CUT, foldset=FS, fold=FOLD, ontology=n, task=task, i=s0 + i, regime=name, seed=0, rank=filtered_rank(sc, q[i], tg, ex, N0),
                                     kind=t.get("kind", "new"), h=t["h"], entailed=t.get("entailed"), covered=covered, n_candidates=len(cset)))
    pd.DataFrame(rows).to_csv(outp(n) + ".tmp", index=False, compression="gzip"); os.replace(outp(n) + ".tmp", outp(n))
    print(f"{MODE} {CUT} {FS} {FOLD} {n}: {nq} queries, {(time.time() - t0) / max(1, nq) * 1000:.0f} ms per query (features + all models)", flush=True)
