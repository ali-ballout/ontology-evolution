"""Shared code for the GNN size study: the Study B network with configurable depth (L) and width (d), placed on one device.
The architecture is that of lib.NBF (same layers, same parameter names); only L and d vary. Data, views, graphs and filtered
ranks come from lib unchanged. Device: E_DEVICE (default cuda when available, else cpu)."""
import os, numpy as np, torch, torch.nn as nn, torch.utils.checkpoint
from lib import DATA, ROOT, view_of, graph, filtered_rank
DEV = torch.device(os.environ.get("E_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu"))
CONFIGS = [(2, 24), (4, 24), (6, 24), (4, 48), (4, 96)]   # (L, d); (4, 24) is the Study B network
def tag(L, d): return f"L{L}d{d}"
class NBFX(nn.Module):
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
        dev = s.q.device; B = len(q); q = torch.as_tensor(q, device=dev)
        lz = (lx if torch.is_tensor(lx) else torch.from_numpy(np.ascontiguousarray(lx))).to(dev).transpose(0, 1)
        f = s.f(g["deg"]).unsqueeze(1).expand(-1, B, -1)
        h = torch.zeros(g["n"], B, s.d, device=dev); h[q, torch.arange(B, device=dev)] = s.q; h = h + s.li(lz); h0 = h
        for l in range(s.L):
            h = torch.utils.checkpoint.checkpoint(s.layer, l, h, h0, g["E"], use_reentrant=False) if torch.is_grad_enabled() else s.layer(l, h, h0, g["E"])
        return s.out(torch.cat([h, f, s.lo(lz)], 2)).squeeze(2).t()
def on_dev(g):
    return dict(n=g["n"], E=[(torch.as_tensor(a).to(DEV), torch.as_tensor(b).to(DEV)) for a, b in g["E"]], deg=torch.as_tensor(g["deg"]).to(DEV))
def split(cut, fold):
    """the tuned ranker's tuning split: 20% of the fold's training ontologies held out (rng 100 + fold)"""
    F = DATA["folds"][cut]["std"][fold]; rng = np.random.default_rng(100 + fold); tr = list(F["train"]); rng.shuffle(tr); nval = max(2, len(tr) // 5)
    return tr[:nval], tr[nval:]
def out_dir(*p):
    d = os.path.join(ROOT, "results", "study_e", *p); os.makedirs(d, exist_ok=True); return d
