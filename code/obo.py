"""Minimal OBO parser: classes with is_a, relationship (A SubClassOf r some B), intersection_of definitions, disjoint_from."""
import re
def load(path):
    T = {}; cur = None; typedef = False
    for line in open(path, encoding="utf8", errors="ignore"):
        line = line.rstrip("\n")
        if line.startswith("["):
            cur = None; typedef = line != "[Term]"
            if not typedef: cur = dict(id=None, name="", is_a=[], rel=[], inter=[], disj=[], obs=False)
            continue
        if cur is None or not line or ":" not in line: continue
        k, v = line.split(":", 1); lab = v.split(" ! ", 1)[1].strip() if " ! " in v else ""; v = v.split(" ! ")[0].strip(); v = re.sub(r"\s*\{.*?\}\s*", " ", v).strip()
        if k == "id": cur["id"] = v; T[v] = cur; cur["lab"] = lab
        elif k == "name": cur["name"] = v
        elif k == "is_a": cur["is_a"].append(v.split()[0])
        elif k == "relationship" and len(v.split()) >= 2: cur["rel"].append(tuple(v.split()[:2]))
        elif k == "intersection_of":
            p = v.split(); cur["inter"].append((None, p[0]) if len(p) == 1 else (p[0], p[1]))
        elif k == "disjoint_from": cur["disj"].append(v.split()[0])
        elif k == "is_obsolete" and v == "true": cur["obs"] = True
    T = {k: t for k, t in T.items() if not t["obs"]}
    ids = sorted(T); ix = {c: i for i, c in enumerate(ids)}
    isa = sorted({(ix[c], ix[p]) for c in ids for p in T[c]["is_a"] if p in ix and p != c})
    rel = sorted({(ix[c], r, ix[p]) for c in ids for r, p in T[c]["rel"] if p in ix and p != c})
    defs = {ix[c]: [(r, ix[p]) for r, p in T[c]["inter"] if p in ix] for c in ids if len(T[c]["inter"]) >= 2 and all(p in ix for _, p in T[c]["inter"])}
    disj = sorted({(ix[c], ix[p]) for c in ids for p in T[c]["disj"] if p in ix})
    return dict(ids=ids, names=[T[c]["name"] or T[c].get("lab", "") for c in ids], isa=isa, rel=rel, defs=defs, disj=disj)
