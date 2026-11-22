"""Cross-encoder comparison in the style of BERTSubs on the random-deletion benchmark (single four-fold split, fold 0).
BERTSubs fine-tunes a BERT cross-encoder to classify (subclass, superclass) pairs from class names, with an 'isolated class'
template (names only) and a 'path context' template (names plus names up the hierarchy). Here a small cross-encoder
(cross-encoder/ms-marco-MiniLM-L-6-v2) is fine-tuned on pairs from the 32 training ontologies of fold 0 and applied zero-shot
to the fold-0 test ontologies and to GO, Uberon and CL. Reference: the path-context model further fine-tuned on the target
ontology's own observed links (the original BERTSubs setting; not zero-shot). The GNN is gnn_deletion.py, fold 0, seeds 0-2.
Each query is ranked against 100 candidates; other true parents are excluded and all models score identical candidate sets.
This script trains both zero-shot cross-encoders and evaluates the hard-negative protocol (true parent + 49 most
name-similar classes + 50 random). Writes results/deletion/crossencoder/hard_ranks.csv.  usage: crossencoder_train.py"""
import sys, os, pickle, time, random, collections, numpy as np, pandas as pd, torch, torch.nn as nn, scipy.sparse as sp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
torch.set_num_threads(2)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
D = os.path.join(paths.DATA, "deletion"); MD = os.path.join(paths.RESULTS, "deletion", "models"); os.makedirs(MD, exist_ok=True)
OD = os.path.join(paths.RESULTS, "deletion", "crossencoder"); os.makedirs(OD, exist_ok=True)
O = pickle.load(open(f"{D}/ontos.pkl", "rb"))
POOL = sorted(n for n, o in O.items() if 800 <= len(o["ids"]) <= 8700 and len(o["isa"]) >= 0.8 * len(o["ids"]))
TEST = ["pato", "so", "po", "envo", "zfa", "cmo", "mco", "pw"]; LARGE = ["go", "uberon", "cl"]; TRAIN = [n for n in POOL if n not in TEST]
if os.environ.get("SMOKE"): TRAIN, TEST, LARGE = TRAIN[:2], TEST[:1], []
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"; V = {}; E = {}

def split(o, seed, frac=0.10):
    rng = np.random.default_rng(seed); isa = np.array(o["isa"]); m = rng.random(len(isa)) < frac
    return isa[~m], isa[m]
def tfidf(n):
    if n not in V: V[n] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform(O[n]["names"])
    return V[n]
def emb(n):
    if n not in E: E[n] = np.load(f"{D}/emb/{n}.npy").astype(np.float32)
    return E[n]
def parents(edges):
    up = collections.defaultdict(list)
    for a, b in edges: up[int(a)].append(int(b))
    return up
def text(o, up, c, ctx, skip=None):
    """isolated class: the name; path context: name < parent < grandparent < ... (first parent at each level, up to 3)"""
    s = o["names"][c] or o["ids"][c]
    if not ctx: return s
    x, path = c, []
    for _ in range(3):
        ps = [p for p in up.get(x, []) if not (x == c and p == skip)]
        if not ps: break
        x = ps[0]; path.append(o["names"][x] or o["ids"][x])
    return s + (" < " + " < ".join(path) if path else "")
def ancestors(up, c):
    seen, st = set(), [c]
    while st:
        x = st.pop()
        for y in up.get(x, []):
            if y not in seen: seen.add(y); st.append(y)
    return seen

def pairs_for(n, npos, seed, ctx, obs=None):
    """positives: observed links (sub context built without the target link); negatives: 2 name-similar non-ancestors + 1 random"""
    o = O[n]; rng = np.random.default_rng(seed); own = obs is not None
    if not own: obs, _ = split(o, seed)
    up = parents(obs); full = up if own else parents(o["isa"]); N = len(o["ids"])   # target ontology: only observed links, never hidden ones
    pos = obs[rng.choice(len(obs), min(npos, len(obs)), replace=False)]; S = (tfidf(n)[pos[:, 0]] @ tfidf(n).T).toarray(); out = []
    for k, (c, p) in enumerate(pos):
        c, p = int(c), int(p); bad = ancestors(full, c) | {c}; a = text(o, up, c, ctx, skip=p); out.append((a, text(o, up, p, ctx), 1.0))
        hard = [int(j) for j in np.argsort(-S[k])[:30] if int(j) not in bad][:2]; rnd = [int(j) for j in rng.integers(0, N, 5) if int(j) not in bad][:1]
        out += [(a, text(o, up, j, ctx), 0.0) for j in hard + rnd]
    return out

def finetune(model, tok, data, epochs=1, bs=32, lr=2e-5, seed=0, maxlen=96):
    random.Random(seed).shuffle(data); opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train(); bce = nn.BCEWithLogitsLoss(); t0 = time.time()
    for ep in range(epochs):
        for s in range(0, len(data), bs):
            b = data[s:s + bs]; x = tok([u for u, _, _ in b], [v for _, v, _ in b], truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
            loss = bce(model(**x).logits.squeeze(1), torch.tensor([y for _, _, y in b])); opt.zero_grad(); loss.backward(); opt.step()
            if (s // bs) % 200 == 199: print(f"    {s + bs}/{len(data)} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    return model.eval()
def score_pairs(model, tok, A, B, bs=128, maxlen=96):
    out = []
    with torch.no_grad():
        for s in range(0, len(A), bs):
            x = tok(A[s:s + bs], B[s:s + bs], truncation=True, max_length=maxlen, padding=True, return_tensors="pt"); out.append(model(**x).logits.squeeze(1).numpy())
    return np.concatenate(out)

# ---------- GNN of gnn_deletion.py (same architecture; fold 0 weights)
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
def chans(n, q):
    t = (tfidf(n)[q] @ tfidf(n).T).toarray().astype(np.float32); e = emb(n)[q] @ emb(n).T
    return np.stack([t, e, np.ones_like(t)], 2).astype(np.float32), t, e
def vote(S, obs, N, k=10):
    S = S.copy(); th = -np.sort(-S, 1)[:, k - 1:k]; S[(S < th) | (S <= 0)] = 0
    return np.asarray((sp.csr_matrix(S) @ sp.csr_matrix((np.ones(len(obs)), (obs[:, 0], obs[:, 1])), shape=(N, N))).todense())

# ---------- train the zero-shot cross-encoders on the training ontologies
tok = AutoTokenizer.from_pretrained(BASE); CE = {}
for ctx, lab in ((False, "isolated class"), (True, "path context")):
    data = [x for i, n in enumerate(TRAIN) for x in pairs_for(n, 10 if os.environ.get("SMOKE") else 400, 100 + i, ctx)]; print(f"training BERTSubs-style ({lab}) on {len(data)} pairs from {len(TRAIN)} ontologies", flush=True)
    CE[lab] = finetune(AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1), tok, data)
    torch.save(CE[lab].state_dict(), f"{MD}/crossencoder_{'pc' if ctx else 'ic'}.pt")
GNNS = []
for s in range(3):
    m = NBF(); m.load_state_dict(torch.load(f"{MD}/gnn_f0_s{s}.pt")); GNNS.append(m.eval())

z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9); rows = []
def crank(sc):   # rank of column 0 (the true parent) among the candidates
    return 1 + (sc[:, 1:] > sc[:, :1]).sum(1) + 0.5 * (sc[:, 1:] == sc[:, :1]).sum(1)
for n in TEST + LARGE:
    o = O[n]; N = len(o["ids"]); obs, hid = split(o, 2026); rng = np.random.default_rng(5); hid = hid[rng.choice(len(hid), min(300, len(hid)), replace=False)]; q = hid[:, 0]
    up = parents(obs); full = parents(o["isa"]); g = graph(o, obs); t0 = time.time(); crng = np.random.default_rng(17)
    lx, St, Se = chans(n, q); St[np.arange(len(q)), q] = 0; Se[np.arange(len(q)), q] = 0
    C = []                                                     # candidate sets
    for i, (c, p) in enumerate(hid):
        bad = set(full.get(int(c), [])) | {int(c)}; hard = [int(j) for j in np.argsort(-St[i])[:200] if int(j) not in bad][:49]
        pool = np.setdiff1d(np.arange(N), list(bad | set(hard))); C.append([int(p)] + hard + [int(j) for j in crng.choice(pool, 50, replace=False)])
    C = np.array(C); I = np.arange(len(q))[:, None]; SC = {}
    with torch.no_grad(): G = [torch.cat([m(g, torch.tensor(q[s:s + 25]), lx[s:s + 25]) for s in range(0, len(q), 25)]).numpy() for m in GNNS]
    for k, Gs in enumerate(G): SC[f"GNN + names (seed {k})"] = Gs[I, C]
    SC["names only (TF-IDF)"] = St[I, C]; S2 = St + Se; SC["look-alike vote + names"] = (z(vote(S2, obs, N)) + z(S2))[I, C]
    A_ic = [text(o, up, int(c), False) for c in q for _ in range(C.shape[1])]; B_ic = [text(o, up, int(j), False) for row in C for j in row]
    A_pc = [text(o, up, int(c), True) for c in q for _ in range(C.shape[1])]; B_pc = [text(o, up, int(j), True) for row in C for j in row]
    SC["BERTSubs-style, isolated class, zero-shot"] = score_pairs(CE["isolated class"], tok, A_ic, B_ic).reshape(C.shape)
    SC["BERTSubs-style, path context, zero-shot"] = score_pairs(CE["path context"], tok, A_pc, B_pc).reshape(C.shape)
    ref = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1); ref.load_state_dict(CE["path context"].state_dict())
    ref = finetune(ref, tok, pairs_for(n, 400, 999, True, obs=obs))                                       # target ontology's observed links only
    SC["BERTSubs-style, path context, fine-tuned on this ontology (reference)"] = score_pairs(ref, tok, A_pc, B_pc).reshape(C.shape)
    gm = np.mean([z(SC[f"GNN + names (seed {k})"]) for k in range(3)], 0); SC["GNN + BERTSubs zero-shot (summed standardised scores)"] = gm + z(SC["BERTSubs-style, path context, zero-shot"])
    for m, sc in SC.items():
        r = crank(sc); rows += [dict(ontology=n, model=m, i=i, rank=float(x)) for i, x in enumerate(r)]
    pd.DataFrame(rows).to_csv(f"{OD}/hard_ranks.csv", index=False)
    rr = pd.DataFrame(rows); rr = rr[rr.ontology == n]; rr["rr"] = 1 / rr["rank"]; print(n, round(time.time() - t0), "s", flush=True)
    print(rr.groupby("model").rr.mean().round(3).to_string(), flush=True)
