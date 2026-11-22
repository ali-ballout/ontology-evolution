"""Sentence embeddings (all-MiniLM-L6-v2, normalised, float16) of every class name of the random-deletion ontologies, one .npy
per ontology in data/deletion/emb/.  usage: embed_current.py"""
import os, sys, pickle, time, numpy as np
from sentence_transformers import SentenceTransformer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
D = os.path.join(paths.DATA, "deletion"); MD = os.path.join(paths.RESULTS, "deletion", "models"); os.makedirs(MD, exist_ok=True)
O = pickle.load(open(f"{D}/ontos.pkl", "rb")); os.makedirs(f"{D}/emb", exist_ok=True); m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
POOL = sorted(n for n, o in O.items() if 800 <= len(o["ids"]) <= 8700 and len(o["isa"]) >= 0.8 * len(o["ids"]))
for n in POOL + ["go", "uberon", "cl"]:
    f = f"{D}/emb/{n}.npy"
    if os.path.exists(f): continue
    t = time.time(); e = m.encode(O[n]["names"], batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    np.save(f, e.astype(np.float16)); print(n, len(e), round(time.time() - t), "s", flush=True)
