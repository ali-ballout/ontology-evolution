"""Builds data/studyB.pkl.xz (cutoff graphs, histories, queries, folds, families) and data/emb_1.npz ... emb_3.npz from the raw snapshots,
the HermiT hierarchies (data/hier.pkl.xz) and the name embeddings (data/names_emb.npz, from embed_names.py).
Per cutoff C (2019, 2021, 2023):
  graphs     rC of every ontology with a snapshot: ids, names, is-a index pairs, relation index pairs
  history    (2021, 2023 only) r1 = the snapshot two years earlier (2019 for 2021, 2021 for 2023), used when its release date
             is >= 180 days before rC's. A: links added between classes in r1 (kind moved/added, own, entailed by r1 via
             HermiT on r1); B: classes new in rC with a parent in r1 (own flag, name from rC); par: rC and r1 parents.
  queries    the shared query samples (code/study_a/queries.py; own namespace, still present in the current release; up to 150 per horizon
             bin and task), with targets, filtered parents, kind, horizon, entailment
  folds      std: 4 folds of the test ontologies (>= 10 queries in a task), GO / Uberon / CL in different folds;
             famA: test the connected block's test ontologies, train on every ontology outside the block;
             famB: test the independent test ontologies, train on every other ontology;
             famBsize: famB with training drawn at random (rng 11) to famA's training size
  families   0.5% class / is-a overlap, connected components, recomputed at the cutoff
Embeddings: MiniLM float16 for every name used. usage: build_data.py"""
import sys, os, pickle, lzma, itertools, datetime, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "study_a"))
import obo as obo2, paths
import queries as Q
HIER = paths.hierarchies()
CUTS = ["2019-07-01", "2021-07-01", "2023-07-01"]; R1 = {"2021-07-01": "2019-07-01", "2023-07-01": "2021-07-01"}
d = datetime.date.fromisoformat
IDX = paths.snapshot_index()
def parse(C, n):
    o = obo2.load(paths.snapshot_file(C, n)); return dict(ids=list(o["ids"]), names=list(o["names"]), isa=np.array(o["isa"], dtype=np.int32).reshape(-1, 2), rel=np.array([(a, b) for a, _, b in o["rel"]], dtype=np.int32).reshape(-1, 2))
def anc(H, c):
    seen, st = set(), [c]
    while st:
        x = st.pop()
        for y in H.get(x, []):
            if y not in seen: seen.add(y); st.append(y)
    return seen
DATA = dict(cuts=CUTS, r1=R1, graphs={}, history={}, queries={}, folds={}, families={}, test={})
for C in CUTS:
    names_ok = sorted(n for n, v in IDX[C].items() if v.get("ok")); G = {n: parse(C, n) for n in names_ok}; DATA["graphs"][C] = G
    # families
    ids = {n: set(G[n]["ids"]) for n in G}; E = {n: {(G[n]["ids"][a], G[n]["ids"][b]) for a, b in G[n]["isa"]} for n in G}; par = {n: n for n in G}
    def f(x):
        while par[x] != x: x = par[x]
        return x
    for a, b in itertools.combinations(names_ok, 2):
        if max(len(ids[a] & ids[b]) / max(1, min(len(ids[a]), len(ids[b]))), len(E[a] & E[b]) / max(1, min(len(E[a]), len(E[b])))) >= 0.005: par[f(a)] = f(b)
    comp = collections.defaultdict(list)
    for n in names_ok: comp[f(n)].append(n)
    block = sorted(max(comp.values(), key=len)); DATA["families"][C] = dict(block=block, independent=sorted(set(names_ok) - set(block)))
    # queries
    QD = {}
    for n in names_ok:
        r = Q.queries(C, n)
        if r is None: continue
        rC, qa, qb, B, _ = r
        QD[n] = dict(A=[dict(x=t["x"], y=t["y"], kind=t["kind"], h=t["h"], excl=sorted(t["excl"] - {t["y"]})) for t in qa],
                     B=[dict(x=x, name=B[x]["name"], targets=sorted(B[x]["tg"]), h=B[x]["h"]) for x in qb])
        Hc = HIER[C][n]
        for t in QD[n]["A"]: t["entailed"] = t["y"] in anc(Hc, t["x"])
    DATA["queries"][C] = QD; test = sorted(n for n, q in QD.items() if len(q["A"]) >= 10 or len(q["B"]) >= 10); DATA["test"][C] = test
    # folds
    rng = np.random.default_rng(7); big = [n for n in ["go", "uberon", "cl"] if n in test]; rest = [n for n in test if n not in big]; rng.shuffle(rest)
    std = [[] for _ in range(4)]
    for i, n in enumerate(big): std[i].append(n)
    for i, n in enumerate(rest): std[(len(big) + i) % 4].append(n)
    folds = {"std": [dict(test=sorted(t), train=[n for n in names_ok if n not in t]) for t in std]}
    tA = [n for n in test if n in block]; trA = [n for n in names_ok if n not in block]; tB = [n for n in test if n not in block]; trB = [n for n in names_ok if n not in tB]
    folds["famA"] = [dict(test=tA, train=trA)]; folds["famB"] = [dict(test=tB, train=trB)]
    folds["famBsize"] = [dict(test=tB, train=sorted(np.random.default_rng(11).choice(trB, min(len(trA), len(trB)), replace=False).tolist()))]
    DATA["folds"][C] = folds
    # history
    if C in R1:
        HI = {}; r1k = R1[C]
        for n in names_ok:
            if not IDX[r1k].get(n, {}).get("ok") or (d(IDX[C][n]["date"]) - d(IDX[r1k][n]["date"])).days < 180: continue
            g1 = DATA["graphs"][r1k][n]; c1 = set(g1["ids"]); gC = G[n]; own = lambda c: c.split(":")[0].lower() == n
            p1, pC = collections.defaultdict(set), collections.defaultdict(set)
            for a, b in g1["isa"]: p1[g1["ids"][a]].add(g1["ids"][b])
            for a, b in gC["isa"]: pC[gC["ids"][a]].add(gC["ids"][b])
            H1 = HIER[r1k][n]
            A = [dict(x=x, y=y, kind="moved" if p1[x] - pC[x] else "added", own=own(x), entailed=y in anc(H1, x)) for x in sorted(pC) if x in c1 for y in sorted(pC[x] - p1[x]) if y in c1]
            nmC = dict(zip(gC["ids"], gC["names"])); Bh = {x: dict(parents=sorted(pC[x] & c1), own=own(x), name=nmC.get(x, "")) for x in sorted(pC) if x not in c1 and pC[x] & c1}
            if len(A) + len(Bh) >= 5: HI[n] = dict(A=A, B=Bh, par={x: sorted(pC[x] | p1[x]) for x in set(a["x"] for a in A) | set(Bh)}, r1=r1k)
        DATA["history"][C] = HI
    print(C, f"{len(G)} graphs, {len(test)} test ontologies, block {len(block)}, folds:", {k: [len(v["test"]) for v in vs] for k, vs in folds.items()},
          f"train famA {len(trA)}; history ontologies {len(DATA['history'].get(C, {}))}", flush=True)
# embeddings
need = {x for C in CUTS for g in DATA["graphs"][C].values() for x in g["names"] if x.strip()}
need |= {t["name"] for C in CUTS for q in DATA["queries"][C].values() for t in q["B"] if t["name"].strip()}
need |= {b["name"] for C in DATA["history"] for h in DATA["history"][C].values() for b in h["B"].values() if b["name"].strip()}
z = np.load(os.path.join(paths.DATA, "names_emb.npz"), allow_pickle=True); ix = {x: i for i, x in enumerate(z["names"])}; names = sorted(x for x in need if x in ix)
print("names needed", len(need), "available", len(names))
names = np.array(names, dtype=object); E = z["E"][[ix[x] for x in names]]; cut = [0, len(names) // 3, 2 * len(names) // 3, len(names)]
for k in range(3): np.savez_compressed(os.path.join(paths.DATA, f"emb_{k + 1}.npz"), names=names[cut[k]:cut[k + 1]], E=E[cut[k]:cut[k + 1]])   # three parts, each under 50 MB
open(os.path.join(paths.DATA, "studyB.pkl.xz"), "wb").write(lzma.compress(pickle.dumps(DATA), preset=9))
