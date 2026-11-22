"""Study F1: the fixed deletion queries on the cutoff snapshots of every std-fold test ontology. In each snapshot 10% of
is-a links are hidden (rng seeded with the CRC-32 of "cutoff/ontology") and up to 300 hidden links are sampled; the child of
each sampled link is a query. Writes results/study_f/deletion_queries.pkl.xz:
{cutoff: {ontology: dict(fold, hidden=bool mask over the snapshot's is-a list, queries=indices into that list)}}.
usage: deletion_queries.py"""
import os, sys, zlib, lzma, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
D = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read())); OUT = {}
for C in D["cuts"]:
    OUT[C] = {}
    for f, F in enumerate(D["folds"][C]["std"]):
        for n in F["test"]:
            isa = np.asarray(D["graphs"][C][n]["isa"], dtype=int).reshape(-1, 2); rng = np.random.default_rng(zlib.crc32(f"{C}/{n}".encode()))
            m = rng.random(len(isa)) < 0.10; h = np.flatnonzero(m); q = np.sort(rng.choice(h, min(300, len(h)), replace=False))
            OUT[C][n] = dict(fold=f, hidden=m, queries=q)
    print(C, {n: len(v["queries"]) for n, v in OUT[C].items()})
od = os.path.join(paths.RESULTS, "study_f"); os.makedirs(od, exist_ok=True)
open(os.path.join(od, "deletion_queries.pkl.xz"), "wb").write(lzma.compress(pickle.dumps(OUT), preset=9))
