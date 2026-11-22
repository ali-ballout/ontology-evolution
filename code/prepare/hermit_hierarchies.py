"""HermiT classification of every cutoff snapshot. Each OBO file is translated to OWL (code/owl.py: is-a links, existential
restrictions, intersection definitions, disjointness axioms) and classified with HermiT (-c). Result per ontology:
{class id: [direct inferred superclass ids]} (equivalent classes listed as mutual superclasses), collected for the three
cutoffs into data/hier.pkl.xz.  usage: hermit_hierarchies.py"""
import sys, os, re, subprocess, collections, tempfile, lzma, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, owl as owlout, paths
HJ = paths.hermit_dir(); IDX = paths.snapshot_index(); OUT = {}
for D in paths.CUTS:
    OUT[D] = {}
    for n in sorted(k for k, v in IDX[D].items() if v.get("ok")):
        o = obo2.load(paths.snapshot_file(D, n)); f = os.path.join(tempfile.gettempdir(), f"{D}_{n}.ofn"); owlout.write_ofn(o, f); res = f + ".hier"
        r = subprocess.run(["java", "-Xmx3g", "-cp", f"{HJ}:{HJ}/HermiT.jar", "org.semanticweb.HermiT.cli.CommandLine", "-c", "-o", res, "file://" + f], capture_output=True, timeout=2400)
        if r.returncode != 0: OUT[D][n] = {"_failed": "HermiT error"}; print(D, n, "failed", flush=True); continue
        up = collections.defaultdict(set)
        for l in open(res):
            ids = [int(x) for x in re.findall(r"/c(\d+)>", l)]
            if l.startswith("SubClassOf") and len(ids) == 2: up[ids[0]].add(ids[1])
            elif l.startswith("EquivalentClasses"):
                for a in ids:
                    for b in ids:
                        if a != b: up[a].add(b)
        I = o["ids"]; OUT[D][n] = {I[a]: [I[b] for b in bs if b < len(I)] for a, bs in up.items() if a < len(I)}
        os.remove(f); os.remove(res); print(D, n, len(up), flush=True)
open(os.path.join(paths.DATA, "hier.pkl.xz"), "wb").write(lzma.compress(pickle.dumps(OUT), preset=9))
