"""HermiT classification wall time per 2023 cutoff snapshot (2 CPU cores, -Xmx3g), with ontology size; same procedure as
hermit_hierarchies.py. Writes results/study_c/hermit_timing.csv.  usage: hermit_timing.py"""
import sys, os, time, subprocess, csv, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, owl as owlout, paths
HJ = paths.hermit_dir(); D = "2023-07-01"; rows = []
for n in sorted(k for k, v in paths.snapshot_index()[D].items() if v.get("ok")):
    o = obo2.load(paths.snapshot_file(D, n)); f = os.path.join(tempfile.gettempdir(), f"t_{n}.ofn"); owlout.write_ofn(o, f); t = time.time()
    r = subprocess.run(["java", "-Xmx3g", "-cp", f"{HJ}:{HJ}/HermiT.jar", "org.semanticweb.HermiT.cli.CommandLine", "-c", "-o", f + ".hier", "file://" + f], capture_output=True, timeout=2400)
    rows.append(dict(ontology=n, classes=len(o["ids"]), isa=len(o["isa"]), relations=len(o["rel"]), hermit_s=round(time.time() - t, 2), ok=r.returncode == 0)); print(rows[-1], flush=True)
    os.remove(f); os.path.exists(f + ".hier") and os.remove(f + ".hier")
w = csv.DictWriter(open(os.path.join(paths.RESULTS, "study_c", "hermit_timing.csv"), "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
