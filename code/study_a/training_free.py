"""Study A: predictability of future subclass edits by horizon and edit type, training-free rankers only.
Queries: the shared samples (see queries.py; the sampling code is repeated here unchanged).
Rankers: names only (TF-IDF + MiniLM), name vote, structural vote, hybrid (standardised name vote + standardised names-only
score + 0.25 x standardised structural vote) and, for task A, the reasoner rankers (inferred ancestors of x in rC first:
by distance / by hybrid score).
Ranks: all rC classes are candidates; the query's parents at rC, at first appearance and in the current release are
filtered (task A); in task B every rC parent at first appearance or in the current release is a target and the best
counts. Ties count half. usage: training_free.py C n[,n...]   writes results/study_a/ranks/<C>_<n>.csv"""
import sys, os, collections, datetime, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, paths, feats as SB
C = sys.argv[1]; YEARS = paths.YEARS; CUR = paths.CURRENT; OUTD = os.path.join(paths.RESULTS, "study_a", "ranks"); os.makedirs(OUTD, exist_ok=True)
_e = [np.load(os.path.join(paths.DATA, f"emb_{k}.npz"), allow_pickle=True) for k in (1, 2, 3)]   # name embeddings, stored in three parts
EIX = {x: i for i, x in enumerate(np.concatenate([p["names"] for p in _e]))}; EMB = np.concatenate([p["E"] for p in _e]); del _e
yr = lambda a, b: (datetime.date.fromisoformat(b) - datetime.date.fromisoformat(a)).days / 365.25
z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
IDX = paths.snapshot_index(); HIER = paths.hierarchies()
def load(D, n):
    o = obo2.load(paths.snapshot_file("current" if D == "cur" else D, n)); I = o["ids"]; par = collections.defaultdict(set)
    for a, b in o["isa"]: par[I[a]].add(I[b])
    return o, par, dict(zip(I, o["names"]))
def rank(s, ci, tg, excl, N0):
    s = s.astype(np.float64).copy(); s[N0:] = -1e18; s[ci] = -1e18; best = 1e9
    for t in tg:
        x = s.copy(); oth = [o for o in excl + tg if o != t]; x[oth] = -1e18; v = x[t]; best = min(best, 1 + (x > v).sum() + 0.5 * ((x == v).sum() - 1))
    return float(best)
for n in sys.argv[2].split(","):
    out = os.path.join(OUTD, f"{C}_{n}.csv")
    if os.path.exists(out) or not IDX[C].get(n, {}).get("ok"): continue
    rC, pC, _ = load(C, n); cC = set(rC["ids"]); later = [D for D in YEARS if D > C and IDX[D].get(n, {}).get("ok")] + ["cur"]
    _, pcur, ncur = load("cur", n); own = lambda c: c.split(":")[0].lower() == n
    H = HIER[C][n]
    def anc(c):
        d, fr = {c: 0}, [c]
        while fr:
            nx = []
            for x in fr:
                for y in H.get(x, []):
                    if y not in d: d[y] = d[x] + 1; nx.append(y)
            fr = nx
        d.pop(c); return d
    A, B, seenL, seenC = [], {}, set(), set()
    for D in later:
        _, pD, nD = load(D, n); h = yr(C, CUR if D == "cur" else D); hb = "1y" if h < 1.5 else ("2y" if h < 2.5 else "3y+")
        for x, ps in pD.items():
            if not own(x): continue
            if x not in cC:
                if x in seenC: continue
                seenC.add(x)
                if ps & cC and x in ncur: B[x] = dict(h=hb, name=nD.get(x, "") or ncur.get(x, ""), tg=(ps | pcur.get(x, set())) & cC, firstpar=ps)
                continue
            for y in sorted(ps - pC.get(x, set())):   # sorted: sampling must not depend on string hashing
                if y in cC and (x, y) not in seenL:
                    seenL.add((x, y))
                    if y in pcur.get(x, set()): A.append(dict(x=x, y=y, h=hb, kind="moved" if pC.get(x, set()) - ps else "added", excl=pC.get(x, set()) | ps | pcur.get(x, set())))
    rng = np.random.default_rng(31); qa, qb = [], []
    for hb in ["1y", "2y", "3y+"]:
        a = [t for t in A if t["h"] == hb]; qa += [a[i] for i in sorted(rng.choice(len(a), min(150, len(a)), replace=False))] if a else []
        b = sorted(x for x, t in B.items() if t["h"] == hb); qb += sorted(rng.choice(b, min(150, len(b)), replace=False).tolist()) if b else []
    if not qa and not qb: pd.DataFrame(columns=["cutoff"]).to_csv(out, index=False); print(C, n, "no targets", flush=True); continue
    ids = list(rC["ids"]) + qb; names = list(rC["names"]) + [B[x]["name"] for x in qb]; ix = {c: i for i, c in enumerate(ids)}; N0 = len(rC["ids"]); N = len(ids)
    named = np.array([bool(x.strip()) for x in names]); V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform([x or "_" for x in names])
    E = np.zeros((N, EMB.shape[1]), np.float32); hit = [(i, EIX[x]) for i, x in enumerate(names) if x.strip() and x in EIX]
    if hit: a, b = zip(*hit); E[list(a)] = EMB[list(b)]
    isa = np.array(rC["isa"], dtype=int).reshape(-1, 2); M = SB.mats(dict(ids=ids, rel=rC["rel"]), isa); P = sp.csr_matrix((np.ones(len(isa)), (isa[:, 0], isa[:, 1])), shape=(N, N))
    rows = []
    for task, qs in (("A", qa), ("B", qb)):
        for s0 in range(0, len(qs), 50):
            part = qs[s0:s0 + 50]; q = np.array([ix[t["x"]] if task == "A" else ix[t] for t in part])
            m = named[None, :] & named[q][:, None]; St = np.where(m, (V[q] @ V.T).toarray(), 0); Se = np.where(m, E[q] @ E.T, 0); S2 = St + Se
            S2[np.arange(len(q)), q] = 0; S2[:, N0:] = 0; Sk = S2.copy(); th = -np.sort(-Sk, 1)[:, 9:10]; Sk[(Sk < th) | (Sk <= 0)] = 0
            nv = np.asarray((sp.csr_matrix(Sk) @ P).todense()); sv = SB.feats(M, q)[:, :, 4]; hyb = z(nv) + z(S2) + 0.25 * z(sv)
            SC = {"names only": S2, "name vote": z(nv) + z(S2), "structural vote": sv + 1e-6 * np.log1p(M["ideg"])[None, :], "hybrid": hyb}
            for i, t in enumerate(part):
                if task == "A":
                    tg = [ix[t["y"]]]; excl = [ix[p] for p in t["excl"] if p in ix and p != t["y"]]; ad = anc(t["x"]); ent = t["y"] in ad
                    near = np.zeros(N); isanc = np.zeros(N)
                    for y, d in ad.items():
                        if y in ix and ix[y] < N0: near[ix[y]] = 1.0 / d; isanc[ix[y]] = 1
                    hs = hyb[i] / (np.abs(hyb[i]).max() + 1) * 0.5
                    sc = dict(SC); sc = {k: v[i] for k, v in sc.items()}; sc["reasoner: nearest inferred ancestor"] = 10 * near + hs; sc["reasoner: ancestors by hybrid"] = 10 * isanc + hs
                    meta = dict(kind=t["kind"], h=t["h"], entailed=ent)
                else:
                    tg = [ix[p] for p in B[t]["tg"]]; excl = []; sc = {k: v[i] for k, v in SC.items()}; meta = dict(kind="new", h=B[t]["h"], entailed=None)
                for k, v in sc.items(): rows.append(dict(cutoff=C, ontology=n, task=task, i=s0 + i, model=k, rank=rank(v, q[i], tg, excl, N0), **meta))
    pd.DataFrame(rows).to_csv(out, index=False); print(C, n, len(qa), len(qb), flush=True)
