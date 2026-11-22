"""Shared Study C helpers: hybrid features, reasoner features (imported by ranker_c.py and timing_c.py)."""
import os, pickle, lzma, collections, numpy as np, scipy.sparse as sp
from lib import z, ROOT
import feats as FT
K = 300
HIER = pickle.loads(lzma.decompress(open(os.path.join(ROOT, "data", "hier.pkl.xz"), "rb").read()))
def scores(v, isa, rel, q, N0):
    St, Se, _ = v.sims(q); S2 = St + Se; S2[np.arange(len(q)), q] = 0; S2[:, N0:] = 0
    P = sp.csr_matrix((np.ones(len(isa)), (isa[:, 0], isa[:, 1])), shape=(len(v.ids),) * 2)
    Sk = S2.copy(); th = -np.sort(-Sk, 1)[:, 9:10]; Sk[(Sk < th) | (Sk <= 0)] = 0; nv = np.asarray((sp.csr_matrix(Sk) @ P).todense())
    M = FT.mats(dict(ids=v.ids, rel=[(a, 0, b) for a, b in rel]), isa); S = FT.feats(M, q); hyb = z(nv) + z(S2) + 0.25 * z(S[:, :, 4])
    hyb[:, N0:] = -1e9; hyb[np.arange(len(q)), q] = -1e9
    order = np.argsort(-hyb, 1); hr = np.empty_like(order); hr[np.arange(len(q))[:, None], order] = np.arange(hyb.shape[1])[None, :]
    nf = np.stack([St, Se, S2, nv], 2); nf = np.concatenate([nf, nf / (nf.max(1, keepdims=True) + 1e-9)], 2)
    return np.concatenate([nf, S, hyb[:, :, None], np.log1p(hr)[:, :, None]], 2).astype(np.float32), hyb, order[:, :K]
class Reasoner:
    """up: {id: [superclass ids]} over a view's ids; ancestors with distance and descendants of a query (by view index)"""
    def __init__(s, v, up):
        s.v = v; s.up = collections.defaultdict(list); s.down = collections.defaultdict(list)
        for a, bs in up.items():
            if a in v.ix:
                for b in bs:
                    if b in v.ix: s.up[v.ix[a]].append(v.ix[b]); s.down[v.ix[b]].append(v.ix[a])
    def bfs(s, q, nb):
        d, fr = {q: 0}, [q]
        while fr:
            nx = []
            for x in fr:
                for y in nb.get(x, []):
                    if y not in d: d[y] = d[x] + 1; nx.append(y)
            fr = nx
        d.pop(q); return d
    def feats(s, q, N):
        R = np.zeros((N, 3), np.float32); anc = s.bfs(q, s.up)
        for y, dist in anc.items(): R[y, 0] = 1; R[y, 1] = 1.0 / dist
        for y in s.bfs(q, s.down): R[y, 2] = 1
        return R, anc
def told_up(v, obs):
    up = collections.defaultdict(list)
    for a, b in obs: up[v.ids[a]].append(v.ids[b])
    return up
