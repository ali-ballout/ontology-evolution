"""Random-deletion benchmark: completing ontologies never seen in training (current releases).
Pool: the OBO ontologies with 800 to 8,700 classes and at least 0.8 is-a links per class. In each test ontology 10% of the
is-a links are hidden (fixed rng) and up to 300 of their children are queries, ranked against every class (filtered ranks).
The GNN gets two name signals per class (TF-IDF character n-gram similarity and MiniLM sentence-embedding similarity) and a
'has a name' flag; names are randomly withheld during training, so one model covers 0%, 50% and 100% of names withheld.
Fold sets: FOLDSET=grouped uses data/deletion/grouped_folds.json (ontologies linked by imports on one side of each split;
GO, Uberon and CL tested with fold 0); otherwise four folds over the pool, fold 0 = pato, so, po, envo, zfa, cmo, mco, pw
(the split used by the cross-encoder comparison). Baselines (seed 0, single split only): names only and look-alike vote.
usage: gnn_deletion.py FOLD SEED STEPS [extra,ontologies]"""
import sys, os, pickle, time, numpy as np, pandas as pd, torch, torch.nn as nn, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
torch.set_num_threads(2)
FOLD, SEED, STEPS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); EXTRA = [x for x in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if x]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
D = os.path.join(paths.DATA, "deletion"); MD = os.path.join(paths.RESULTS, "deletion", "models"); os.makedirs(MD, exist_ok=True)
O = pickle.load(open(f"{D}/ontos.pkl", "rb"))
POOL = sorted(n for n, o in O.items() if 800 <= len(o["ids"]) <= 8700 and len(o["isa"]) >= 0.8 * len(o["ids"]))
T0 = ["pato", "so", "po", "envo", "zfa", "cmo", "mco", "pw"]; REST = [n for n in POOL if n not in T0]
FOLDS = [T0] + [REST[i::3] for i in range(3)]
GROUPED = os.environ.get('FOLDSET') == 'grouped'
if GROUPED:
    import json; GF = json.load(open(f'{D}/grouped_folds.json')); FOLDS = [g['test'] for g in GF]; EXTRA = GF[FOLD]['extra']
TEST = FOLDS[FOLD]; TRAIN = [n for n in POOL if n not in TEST]
COND = [0.0, 0.5, 1.0]; E = {}; V = {}
print(f"fold {FOLD} seed {SEED}: test {TEST} extra {EXTRA}; {len(TRAIN)} training ontologies", flush=True)

def emb(n):
    if n not in E: E[n] = np.load(f"{D}/emb/{n}.npy").astype(np.float32)
    return E[n]
def tfidf(n):
    if n not in V: V[n] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform(O[n]["names"])
    return V[n]
def sims(n, q, named):
    """[B, N] TF-IDF and embedding similarity of each query to every class; zero where either side has no name"""
    t = (tfidf(n)[q] @ tfidf(n).T).toarray().astype(np.float32); e = emb(n)[q] @ emb(n).T
    m = named[None, :] & named[q][:, None]
    return np.where(m, t, 0).astype(np.float32), np.where(m, e, 0).astype(np.float32)
def chans(n, q, named):
    t, e = sims(n, q, named)
    return np.stack([t, e, np.broadcast_to(named[None, :], t.shape).astype(np.float32)], 2)
def split(o, seed, frac=0.10):
    rng = np.random.default_rng(seed); isa = np.array(o["isa"]); m = rng.random(len(isa)) < frac
    return isa[~m], isa[m]
def graph(o, obs):
    n = len(o["ids"]); rel = np.array([(a, b) for a, _, b in o["rel"]], dtype=int).reshape(-1, 2)
    Es = [(obs[:, 0], obs[:, 1]), (obs[:, 1], obs[:, 0]), (rel[:, 0], rel[:, 1]), (rel[:, 1], rel[:, 0])]
    deg = np.stack([np.log1p(np.bincount(d, minlength=n)) for _, d in Es], 1).astype(np.float32)
    return dict(n=n, E=[(torch.tensor(s), torch.tensor(d)) for s, d in Es], deg=torch.tensor(deg))

class NBF(nn.Module):
    def __init__(s, d=24, L=4, C=3):
        super().__init__(); s.d, s.L = d, L; s.q = nn.Parameter(torch.randn(d) * 0.5); s.f = nn.Linear(4, d); s.li = nn.Linear(C, d); s.lo = nn.Linear(C, d)
        s.W = nn.ModuleList(nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(4)) for _ in range(L)); s.S = nn.ModuleList(nn.Linear(d, d) for _ in range(L)); s.N = nn.ModuleList(nn.LayerNorm(d) for _ in range(L))
        s.out = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, 1))
    def forward(s, g, q, lx):
        B = len(q); f = s.f(g["deg"]).unsqueeze(1).expand(-1, B, -1); lz = torch.from_numpy(np.ascontiguousarray(lx)).transpose(0, 1)
        h = torch.zeros(g["n"], B, s.d); h[q, torch.arange(B)] = s.q; h = h + s.li(lz); h0 = h
        for l in range(s.L):
            m = torch.zeros_like(h)
            for t, (src, dst) in enumerate(g["E"]):
                if len(src): m = m.index_add(0, dst, s.W[l][t](h[src]))
            h = s.N[l](h + torch.relu(s.S[l](h) + m + h0))
        return s.out(torch.cat([h, f, s.lo(lz)], 2)).squeeze(2).t()

def train(steps, seed, B=8):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = NBF(); opt = torch.optim.Adam(net.parameters(), lr=3e-3); t0 = time.time()
    for st in range(steps):
        n = TRAIN[rng.integers(len(TRAIN))]; o = O[n]; obs, hid = split(o, int(rng.integers(1 << 30))); hid = hid[rng.choice(len(hid), min(B, len(hid)), replace=False)]
        u = rng.random(); p = 0.0 if u < 0.4 else (1.0 if u > 0.9 else rng.random()); named = rng.random(len(o["ids"])) >= p
        g = graph(o, obs); q = torch.tensor(hid[:, 0]); sc = net(g, q, chans(n, hid[:, 0], named)).clone(); sc[torch.arange(len(q)), q] = -1e9
        loss = nn.functional.cross_entropy(sc, torch.tensor(hid[:, 1])); opt.zero_grad(); loss.backward(); opt.step()
        if st % 250 == 249: print(f"  step {st + 1} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    return net.eval()

def ranks(sc, q, hid, o):
    P = sp.csr_matrix((np.ones(len(o["isa"])), (np.array(o["isa"])[:, 0], np.array(o["isa"])[:, 1])), shape=(len(o["ids"]),) * 2); r = []
    for i in range(len(q)):
        s = sc[i].astype(np.float64).copy(); t = hid[i, 1]; s[q[i]] = -1e18; oth = P[q[i]].indices; s[oth[oth != t]] = -1e18
        r.append(1 + (s > s[t]).sum() + 0.5 * ((s == s[t]).sum() - 1))
    return np.array(r)
z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
def vote(S, obs, N, k=10):
    """the k classes whose names most resemble the query vote (weighted by similarity) for their observed parents"""
    S = S.copy(); th = -np.sort(-S, 1)[:, k - 1:k]; S[(S < th) | (S <= 0)] = 0
    P = sp.csr_matrix((np.ones(len(obs)), (obs[:, 0], obs[:, 1])), shape=(N, N))
    return np.asarray((sp.csr_matrix(S) @ P).todense())

net = train(STEPS, SEED); TAG = ("grouped_" if GROUPED else "") + "gnn"; torch.save(net.state_dict(), f"{MD}/{TAG}_f{FOLD}_s{SEED}.pt"); rows = []
for n in TEST + EXTRA:
    o = O[n]; N = len(o["ids"]); obs, hid = split(o, 2026); rng = np.random.default_rng(5); hid = hid[rng.choice(len(hid), min(300, len(hid)), replace=False)]; q = hid[:, 0]
    g = graph(o, obs); t0 = time.time()
    for c in COND:
        named = np.random.default_rng(11).random(N) >= c; R = {}
        with torch.no_grad(): R["GNN + names (TF-IDF + embeddings), learned jointly"] = ranks(torch.cat([net(g, torch.tensor(q[s:s + 25]), chans(n, q[s:s + 25], named)) for s in range(0, len(q), 25)]).numpy(), q, hid, o)
        if SEED == 0 and not GROUPED:
            St, Se = sims(n, q, named)
            for lab, S in (("TF-IDF", St), ("embeddings", Se)):
                S = S.copy(); S[np.arange(len(q)), q] = 0
                R[f"names only ({lab})"] = ranks(S, q, hid, o); R[f"look-alike vote + names ({lab})"] = ranks(z(vote(S, obs, N)) + z(S), q, hid, o)
            S2 = St + Se; S2[np.arange(len(q)), q] = 0; R["look-alike vote + names (TF-IDF + embeddings)"] = ranks(z(vote(S2, obs, N)) + z(S2), q, hid, o)
        for m, r in R.items(): rows += [dict(ontology=n, fold=FOLD, seed=SEED, withheld=c, model=m, i=i, rank=float(x)) for i, x in enumerate(r)]
    OD = os.path.join(paths.RESULTS, "deletion", "grouped" if GROUPED else "single_split"); os.makedirs(OD, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{OD}/{TAG}_ranks_f{FOLD}_s{SEED}.csv", index=False)
    rr = pd.DataFrame(rows); rr = rr[rr.ontology == n]; rr["rr"] = 1 / rr["rank"]
    print(n, N, "classes", round(time.time() - t0), "s"); print(rr.pivot_table(index="model", columns="withheld", values="rr", aggfunc="mean").round(3).to_string(), flush=True)
