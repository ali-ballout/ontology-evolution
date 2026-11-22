"""MiniLM (sentence-transformers/all-MiniLM-L6-v2) embeddings, L2-normalised, float16, of every class name in the cutoff
snapshots (2019, 2021, 2023), the yearly snapshots and the current releases of the 41 ontologies. Writes data/names_emb.npz;
build_data.py keeps the names it needs in data/emb_1.npz ... emb_3.npz.  usage: embed_names.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, paths
from sentence_transformers import SentenceTransformer
IDX = paths.snapshot_index(); names = set()
for D in paths.YEARS:
    for n, v in IDX[D].items():
        if v.get("ok"): names |= {x for x in obo2.load(paths.snapshot_file(D, n))["names"] if x.strip()}
for n in IDX["2023-07-01"]: names |= {x for x in obo2.load(paths.snapshot_file("current", n))["names"] if x.strip()}
names = sorted(names); print(len(names), "names", flush=True)
E = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(names, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)
np.savez_compressed(os.path.join(paths.DATA, "names_emb.npz"), names=np.array(names, dtype=object), E=E)
