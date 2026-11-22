"""Parse the current releases (snapshots/current/*.obo) used by the random-deletion comparison into data/deletion/ontos.pkl.
usage: parse_current.py"""
import os, sys, glob, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo, paths
out = {}
for f in sorted(glob.glob(os.path.join(paths.SNAPSHOTS, "current", "*.obo"))):
    if os.path.getsize(f) < 20000: continue
    out[os.path.basename(f)[:-4]] = obo.load(f)
os.makedirs(os.path.join(paths.DATA, "deletion"), exist_ok=True); pickle.dump(out, open(os.path.join(paths.DATA, "deletion", "ontos.pkl"), "wb")); print(len(out), "ontologies")
