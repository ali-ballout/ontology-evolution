"""Hybrid heuristic on the random-deletion benchmark: look-alike name vote + names (TF-IDF + embeddings) plus the structural
vote, summed as standardised scores, for structural weights w in {0.25, 0.5, 1, 2} (w = 0.25 is the hybrid reported).
Same queries, name-withholding masks and filtered ranks as gnn_deletion.py; every pool ontology plus GO, Uberon and CL.
Writes results/deletion/grouped/hybrid_ranks.csv (the 'fold' column refers to the single four-fold split).
usage: hybrid_deletion.py"""
import os, sys, pickle, numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import paths, feats as SB
D = os.path.join(paths.DATA, "deletion"); O = pickle.load(open(f"{D}/ontos.pkl", "rb"))
POOL = sorted(n for n, o in O.items() if 800 <= len(o["ids"]) <= 8700 and len(o["isa"]) >= 0.8 * len(o["ids"]))
T0 = ["pato", "so", "po", "envo", "zfa", "cmo", "mco", "pw"]; REST = [n for n in POOL if n not in T0]; FOLDS = [T0] + [REST[i::3] for i in range(3)]
TESTS = [(n, f) for f, t in enumerate(FOLDS) for n in t] + [(n, 0) for n in ["go", "uberon", "cl"]]
def split(o, seed, frac=0.10):
    rng = np.random.default_rng(seed); isa = np.array(o["isa"]); m = rng.random(len(isa)) < frac
    return isa[~m], isa[m]
def ranks(sc, q, hid, o):
    Pf = sp.csr_matrix((np.ones(len(o["isa"])), (np.array(o["isa"])[:, 0], np.array(o["isa"])[:, 1])), shape=(len(o["ids"]),) * 2); r = []
    for i in range(len(q)):
        s = sc[i].astype(np.float64).copy(); t = hid[i, 1]; s[q[i]] = -1e18; oth = Pf[q[i]].indices; s[oth[oth != t]] = -1e18
        r.append(1 + (s > s[t]).sum() + 0.5 * ((s == s[t]).sum() - 1))
    return np.array(r)
z = lambda a: (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9); rows = []
for n, f in TESTS:
    o = O[n]; N = len(o["ids"]); obs, hid = split(o, 2026); rng = np.random.default_rng(5); hid = hid[rng.choice(len(hid), min(300, len(hid)), replace=False)]; q = hid[:, 0]
    M = SB.mats(o, obs); V = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True).fit_transform(o["names"]); E = np.load(f"{D}/emb/{n}.npy").astype(np.float32)
    P = sp.csr_matrix((np.ones(len(obs)), (obs[:, 0], obs[:, 1])), shape=(N, N))
    sv = np.concatenate([SB.feats(M, q[s:s + 50])[:, :, 4] for s in range(0, len(q), 50)])
    for c in (0.0, 0.5):
        named = np.random.default_rng(11).random(N) >= c; m = named[None, :] & named[q][:, None]
        S = np.where(m, (V[q] @ V.T).toarray() + E[q] @ E.T, 0); S[np.arange(len(q)), q] = 0
        Sk = S.copy(); th = -np.sort(-Sk, 1)[:, 9:10]; Sk[(Sk < th) | (Sk <= 0)] = 0; nv = z(np.asarray((sp.csr_matrix(Sk) @ P).todense())) + z(S)
        for w in (0.25, 0.5, 1.0, 2.0):
            r = ranks(nv + w * z(sv), q, hid, o); rows += [dict(ontology=n, fold=f, withheld=c, model=f"name vote + structural vote (w={w})", i=i, rank=float(x)) for i, x in enumerate(r)]
    print(n, flush=True)
OD = os.path.join(paths.RESULTS, "deletion", "grouped"); os.makedirs(OD, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OD, "hybrid_ranks.csv"), index=False)
