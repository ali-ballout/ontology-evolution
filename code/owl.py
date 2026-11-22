"""OBO-to-OWL translation used for HermiT: is-a links, existential restrictions, intersection definitions, disjointness."""
import re
def write_ofn(o, path, keep=None):
    """Write an ontology (or the sub-ontology induced by a set of class indices) in OWL functional syntax."""
    K = set(range(len(o["ids"]))) if keep is None else set(keep); P = lambda r: "p_" + re.sub(r"\W", "_", r); L = ["Prefix(:=<http://x/>)", "Ontology(<http://x/o>"]
    L += [f"Declaration(Class(:c{i}))" for i in sorted(K)]
    L += [f"SubClassOf(:c{a} :c{b})" for a, b in o["isa"] if a in K and b in K]
    L += [f"SubClassOf(:c{a} ObjectSomeValuesFrom(:{P(r)} :c{b}))" for a, r, b in o["rel"] if a in K and b in K]
    for a, d in o["defs"].items():
        if a in K and all(b in K for _, b in d):
            L.append(f"EquivalentClasses(:c{a} ObjectIntersectionOf(" + " ".join(f":c{b}" if r is None else f"ObjectSomeValuesFrom(:{P(r)} :c{b})" for r, b in d) + "))")
    L += [f"DisjointClasses(:c{a} :c{b})" for a, b in o["disj"] if a in K and b in K]
    open(path, "w").write("\n".join(L) + "\n)\n"); return len(L) - 2 - len(K)
