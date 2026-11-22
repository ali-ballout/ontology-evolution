"""Snapshot loading and the shared query samples (Study A; reused by the data preparation and Study D).
For a cutoff C and ontology n: rC = snapshot at C; later yearly snapshots (1 July 2020 ... 2025) and the current release.
Targets (own namespace, still present in the current release), sampled with fixed rng 31, up to 150 per horizon bin
(1 year, 2 years, 3+ years) and task:
  A re-parenting: link (x, y), x and y in rC, absent at rC; kind moved/added; entailed by rC (HermiT)
  B new class:    class x absent from rC, first appearing with at least one parent in rC; added to the rC graph as an
                  isolated node with its name (from the snapshot where it first appears)"""
import sys, os, collections, datetime, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, paths
YEARS = paths.YEARS; CUR = paths.CURRENT
yr = lambda a, b: (datetime.date.fromisoformat(b) - datetime.date.fromisoformat(a)).days / 365.25
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
def queries(C, n):
    """query samples (identical to those of training_free.py): returns (rC, qa, qb, B, anc)"""
    if not IDX[C].get(n, {}).get("ok"): return None
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
    return rC, qa, qb, B, anc
