"""Study D, validation of the logical-safety check (M3) against HermiT. For each sampled re-parenting suggestion (cutoff C, ontology n, query x,
suggestion s): the cutoff snapshot plus the axiom x SubClassOf s is written in OWL functional syntax and HermiT lists every
unsatisfiable class (-U). The suggestion is HermiT-unsafe if some class is unsatisfiable that was satisfiable at the cutoff.
HermiT -U is very slow on CL 2023 (10 min per check); MODE c classifies instead (17 s) and reads the classes equivalent to
owl:Nothing from the taxonomy. usage: hermit_check.py SAMPLE.csv OUT.csv [U|c]   (rows: cutoff, ontology, x, s, flag)
The sample used is results/study_d/hermit_check/validation.csv (columns cutoff, ontology, x, s, flag)."""
import sys, os, re, subprocess, tempfile, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, owl as owlout, paths
HJ = paths.hermit_dir()
S = pd.read_csv(sys.argv[1]); out = sys.argv[2]; MODE = sys.argv[3] if len(sys.argv) > 3 else "U"; done = set()
if os.path.exists(out): done = {tuple(r) for r in pd.read_csv(out)[["cutoff", "ontology", "x", "s"]].itertuples(index=False)}
def unsat(o, tag):
    f = os.path.join(tempfile.gettempdir(), f"hv_{tag}.ofn"); owlout.write_ofn(o, f)
    if MODE == "U":
        r = subprocess.run(["java", "-Xmx3g", "-Xss256m", "-cp", f"{HJ}:{HJ}/HermiT.jar", "org.semanticweb.HermiT.cli.CommandLine", "-U", "file://" + f], capture_output=True, text=True, timeout=1800)
        os.remove(f); assert "owl:Nothing" in r.stdout, r.stdout[-500:] + r.stderr[-500:]
        return {int(m) for m in re.findall(r":c(\d+)", r.stdout)}
    h = f + ".h"; r = subprocess.run(["java", "-Xmx3g", "-Xss256m", "-cp", f"{HJ}:{HJ}/HermiT.jar", "org.semanticweb.HermiT.cli.CommandLine", "-c", "-o", h, "file://" + f], capture_output=True, text=True, timeout=1800)
    os.remove(f); assert r.returncode == 0, r.stderr[-500:]
    u = set()
    for l in open(h):
        if "owl:Nothing" in l or "#Nothing" in l: u |= {int(m) for m in re.findall(r"/c(\d+)>|:c(\d+)\b", l) for m in m if m}
    os.remove(h); return u
base = {}
for (C, n), g in S.groupby(["cutoff", "ontology"]):
    o = obo2.load(paths.snapshot_file(C, n)); ix = {c: i for i, c in enumerate(o["ids"])}; isa0 = list(o["isa"])
    b = unsat(o, f"{os.getpid()}"); rows = []
    for r in g.itertuples(index=False):
        if (C, n, r.x, r.s) in done: continue
        o["isa"] = isa0 + [(ix[r.x], ix[r.s])]; u = unsat(o, f"{os.getpid()}") - b
        rows.append(dict(cutoff=C, ontology=n, x=r.x, s=r.s, flag=r.flag, hermit_unsafe=bool(u), n_new_unsat=len(u), x_unsat=ix[r.x] in u, base_unsat=len(b)))
    o["isa"] = isa0
    if rows: pd.DataFrame(rows).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    print(C, n, len(rows), flush=True)
