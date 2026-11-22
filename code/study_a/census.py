"""Study A census of subclass edits after each cutoff (1 July 2019, 2021, 2023).
Snapshots: rC = latest release <= cutoff; later yearly snapshots 1 July of each following year up to 2025, then
the current release (Nov 2026). For every is-a link absent from rC, its first later snapshot gives the horizon (years after
the cutoff; current release = 3.4 years after 2023, 5.4 after 2021, 7.4 after 2019). Edits are classified by:
  kind      new_placeable: child class absent from rC, at least one parent present in rC (counted per class, first appearance)
            new_outside:   child absent from rC, every parent also new (outside the placement task; per class)
            moved:         child in rC, parent in rC, child lost at least one rC parent by that snapshot
            added:         child in rC, parent in rC, child kept every rC parent
  entailed  (moved/added) parent already an inferred ancestor of the child in rC (HermiT, data/hier.pkl.xz)
  own       child in the ontology's own namespace
  persists  link (or, for classes, the class) still present in the current release
Also per ontology: class counts at rC and current, and a flag when the release file's class count jumps by > 50% between
consecutive snapshots (a change in what the file bundles). Writes results/study_a/census.csv and census_ontology.csv.
usage: census.py"""
import sys, os, collections, datetime, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import obo as obo2, paths
CUTS = paths.CUTS; YEARS = paths.YEARS; CUR = paths.CURRENT
yr = lambda a, b: round((datetime.date.fromisoformat(b) - datetime.date.fromisoformat(a)).days / 365.25, 1)
IDX = paths.snapshot_index(); HIER = paths.hierarchies()
NAMES = sorted(IDX["2023-07-01"])
CACHE = {}
def load(D, n):
    k = (D, n)
    if k not in CACHE:
        o = obo2.load(paths.snapshot_file("current" if D == "cur" else D, n)); I = o["ids"]
        par = collections.defaultdict(set)
        for a, b in o["isa"]: par[I[a]].add(I[b])
        CACHE[k] = (set(I), par)
    return CACHE[k]
def ancestors(h, c):
    seen, st = set(), [c]
    while st:
        x = st.pop()
        for y in h.get(x, []):
            if y not in seen: seen.add(y); st.append(y)
    return seen
rows, orows = [], []
for C in CUTS:
    for n in NAMES:
        if not IDX[C].get(n, {}).get("ok"): continue
        own = lambda c: c.split(":")[0].lower() == n
        later = [D for D in YEARS if D > C and IDX[D].get(n, {}).get("ok")] + ["cur"]
        cC, pC = load(C, n); cur_c, cur_p = load("cur", n)
        H = HIER.get(C, {}).get(n, {"_failed": "missing"}); hok = "_failed" not in H
        seen_link, seen_cls = set(), set(); cnt = collections.Counter(); sizes = [len(cC)]
        for D in later:
            cD, pD = load(D, n); sizes.append(len(cD)); h = yr(C, CUR if D == "cur" else D)
            for x, ps in pD.items():
                if x not in cC:
                    if x in seen_cls: continue
                    seen_cls.add(x); kind = "new_placeable" if ps & cC else "new_outside"
                    cnt[(kind, own(x), None, h, x in cur_c)] += 1; continue
                for y in ps - pC.get(x, set()):
                    if y not in cC or (x, y) in seen_link: continue
                    seen_link.add((x, y)); kind = "moved" if pC.get(x, set()) - ps else "added"
                    ent = (y in ancestors(H, x)) if hok else None
                    cnt[(kind, own(x), ent, h, y in cur_p.get(x, set()))] += 1
        for (kind, o_, ent, h, per), k in cnt.items(): rows.append(dict(cutoff=C, ontology=n, kind=kind, own=o_, entailed=ent, horizon=h, persists=per, count=k))
        jumps = [i for i in range(1, len(sizes)) if sizes[i - 1] and abs(sizes[i] / sizes[i - 1] - 1) > 0.5]
        orows.append(dict(cutoff=C, ontology=n, rC_date=IDX[C][n]["date"], classes_rC=len(cC), classes_cur=len(cur_c), reasoner_ok=hok, file_scope_jumps=len(jumps), later_snapshots=len(later)))
        own_by_kind = collections.Counter()
        for kk, v in cnt.items():
            if kk[1]: own_by_kind[kk[0]] += v
        print(C, n, dict(own_by_kind), 'reasoner' if hok else 'NO REASONER', flush=True); CACHE.clear()
OD = os.path.join(paths.RESULTS, "study_a"); os.makedirs(OD, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OD, "census.csv"), index=False); pd.DataFrame(orows).to_csv(os.path.join(OD, "census_ontology.csv"), index=False)
