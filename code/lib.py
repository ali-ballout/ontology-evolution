"""Shared code for the learned rankers: data loading, the visible-ontology view with its name channels, the query-conditioned
message-passing network (NBFNet style, with layer checkpointing) and filtered ranks. Input modes for the ablations: full (default), names (no edges, no degree features: names only),
struct (no name channels: structure only), norel (is-a edges only, relation edges removed)."""
import lzma, pickle, os, numpy as np, torch, torch.nn as nn, torch.utils.checkpoint
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = pickle.loads(lzma.decompress(open(os.path.join(ROOT, "data", "studyB.pkl.xz"), "rb").read()))
_e = [np.load(os.path.join(ROOT, "data", f"emb_{k}.npz"), allow_pickle=True) for k in (1, 2, 3)]   # name embeddings, stored in three parts
EIX = {x: i for i, x in enumerate(np.concatenate([p["names"] for p in _e]))}; EMB = np.concatenate([p["E"] for p in _e]); del _e
z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
class View:
    """visible ontology: ids, names, is-a (index pairs), relations (index pairs); TF-IDF and MiniLM name similarity"""
    def __init__(s, ids, names, isa, rel, mode="full"):
        s.ids, s.names, s.mode = list(ids), list(names), mode; s.ix = {c: i for i, c in enumerate(s.ids)}
        s.isa = np.asarray(isa, dtype=int).reshape(-1, 2); s.rel = np.asarray(rel, dtype=int).reshape(-1, 2)
        s.named = np.array([bool(x.strip()) for x in s.names]); s.V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform([x or "_" for x in s.names])
        s.E = np.zeros((len(s.names), EMB.shape[1]), np.float32); hit = [(i, EIX[x]) for i, x in enumerate(s.names) if x.strip() and x in EIX]
        if hit: a, b = zip(*hit); s.E[list(a)] = EMB[list(b)]
    def sims(s, q, named=None):
        named = s.named if named is None else (named & s.named); m = named[None, :] & named[q][:, None]
        return np.where(m, (s.V[q] @ s.V.T).toarray(), 0).astype(np.float32), np.where(m, s.E[q] @ s.E.T, 0).astype(np.float32), named
    def chans(s, q, named=None):
        t, e, nm = s.sims(q, named); c = np.stack([t, e, np.broadcast_to(nm[None, :], t.shape).astype(np.float32)], 2)
        return np.zeros_like(c) if s.mode == "struct" else c
def view_of(g, extra=(), mode="full"):
    """graph dict from the data file -> View; extra = [(id, name)] appended as isolated nodes"""
    return View(g["ids"] + [c for c, _ in extra], g["names"] + [nm for _, nm in extra], g["isa"], g["rel"], mode)
def graph(v, isa=None):
    n = len(v.ids); obs = v.isa if isa is None else np.asarray(isa, dtype=int).reshape(-1, 2); rel = v.rel
    if v.mode == "names": obs, rel = obs[:0], rel[:0]
    if v.mode == "norel": rel = rel[:0]
    Es = [(obs[:, 0], obs[:, 1]), (obs[:, 1], obs[:, 0]), (rel[:, 0], rel[:, 1]), (rel[:, 1], rel[:, 0])]
    deg = np.stack([np.log1p(np.bincount(d, minlength=n)) for _, d in Es], 1).astype(np.float32)
    return dict(n=n, E=[(torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long)) for a, b in Es], deg=torch.tensor(deg))
class NBF(nn.Module):
    def __init__(s, d=24, L=4, C=3):
        super().__init__(); s.d, s.L = d, L; s.q = nn.Parameter(torch.randn(d) * 0.5); s.f = nn.Linear(4, d); s.li = nn.Linear(C, d); s.lo = nn.Linear(C, d)
        s.W = nn.ModuleList(nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(4)) for _ in range(L)); s.S = nn.ModuleList(nn.Linear(d, d) for _ in range(L)); s.N = nn.ModuleList(nn.LayerNorm(d) for _ in range(L))
        s.out = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
    def layer(s, l, h, h0, E):
        m = torch.zeros_like(h)
        for t, (src, dst) in enumerate(E):
            if len(src): m = m.index_add(0, dst, s.W[l][t](h)[src])
        return s.N[l](h + torch.relu(s.S[l](h) + m + h0))
    def forward(s, g, q, lx):
        B = len(q); dev = s.q.device; f = s.f(g["deg"]).unsqueeze(1).expand(-1, B, -1); lz = torch.from_numpy(np.ascontiguousarray(lx)).transpose(0, 1).to(dev)
        h = torch.zeros(g["n"], B, s.d, device=dev); h[q, torch.arange(B, device=dev)] = s.q; h = h + s.li(lz); h0 = h
        for l in range(s.L):
            h = torch.utils.checkpoint.checkpoint(s.layer, l, h, h0, g["E"], use_reentrant=False) if torch.is_grad_enabled() else s.layer(l, h, h0, g["E"])
        return s.out(torch.cat([h, f, s.lo(lz)], 2)).squeeze(2).t()
def filtered_rank(score, ci, targets, excl, N0):
    """candidates = the first N0 nodes; other true parents removed; best target counts; ties count half"""
    s = np.asarray(score, dtype=np.float64).copy(); s[N0:] = -1e18; s[ci] = -1e18; best = 1e9
    for t in targets:
        x = s.copy(); oth = [o for o in list(excl) + list(targets) if o != t]; x[oth] = -1e18; v = x[t]
        best = min(best, 1 + (x > v).sum() + 0.5 * ((x == v).sum() - 1))
    return float(best)
def to_device(g, dev):
    """graph dict from graph() moved to a torch device"""
    return dict(n=g["n"], E=[(a.to(dev), b.to(dev)) for a, b in g["E"]], deg=g["deg"].to(dev))
def mode_of(regime): return {"NAMES": "names", "STRUCT": "struct", "NOREL": "norel"}.get(regime, "full")
def out_dir(*p):
    d = os.path.join(ROOT, "results", "study_b", *p); os.makedirs(d, exist_ok=True); return d
