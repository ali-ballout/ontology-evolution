"""Study D, validation of the logical-safety check for CL 2023, where HermiT -U and -c take 5-10 minutes per check when a class becomes unsatisfiable.
HermiT -k checks the satisfiability of the query x after adding x SubClassOf s; if x stays satisfiable, its inferred descendants
(up to 30, nearest first) are checked the same way. usage: hermit_check_k.py SAMPLE.csv OUT.csv"""
import sys, os, subprocess, collections, tempfile, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, owl as owlout, paths
HJ = paths.hermit_dir(); HIER = paths.hierarchies()
S = pd.read_csv(sys.argv[1]); rows = []
for (C, n), g in S.groupby(["cutoff", "ontology"]):
    o = obo2.load(paths.snapshot_file(C, n)); ix = {c: i for i, c in enumerate(o["ids"])}; isa0 = list(o["isa"])
    H = HIER[C][n]; DN = collections.defaultdict(set)
    for a, bs in H.items():
        for b in bs: DN[b].add(a)
    def desc(c):
        out, fr, seen = [], [c], {c}
        while fr and len(out) < 30:
            nx = []
            for x in fr:
                for y in sorted(DN.get(x, ())):
                    if y not in seen: seen.add(y); out.append(y); nx.append(y)
            fr = nx
        return out[:30]
    for r in g.itertuples(index=False):
        o["isa"] = isa0 + [(ix[r.x], ix[r.s])]; f = os.path.join(tempfile.gettempdir(), f"hk_{os.getpid()}.ofn"); owlout.write_ofn(o, f); bad = []; checked = 0
        for c in [r.x] + desc(r.x):
            p = subprocess.run(["java", "-Xmx3g", "-Xss256m", "-cp", f"{HJ}:{HJ}/HermiT.jar", "org.semanticweb.HermiT.cli.CommandLine", f"-k:c{ix[c]}", "file://" + f], capture_output=True, text=True, timeout=600)
            checked += 1; assert "satisfiable" in p.stdout, p.stdout + p.stderr[-300:]
            if "not satisfiable" in p.stdout: bad.append(c); break
        os.remove(f); rows.append(dict(cutoff=C, ontology=n, x=r.x, s=r.s, flag=r.flag, hermit_unsafe=bool(bad), n_new_unsat=len(bad), x_unsat=r.x in bad, base_unsat=-1, n_checked=checked, method="k"))
        print(r.x, r.s, r.flag, bool(bad), checked, flush=True)
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
