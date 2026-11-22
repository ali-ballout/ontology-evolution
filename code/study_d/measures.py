"""Study D measures per (cutoff, ontology), as fixed in the protocol (protocols/study_d.md).
Inputs: top-20 lists per query from export_d.py (GBR_del_c, L0, GNN_R) and baseline_lists.py (names only, hybrid, R0).
Reference (from the snapshots):
  correct(x)  = told parents of x in the Nov 2026 release, existing at the cutoff, not parents of x at the cutoff
  M1 hit@k    = any correct class in the top k (k = 5, 10, 20); M2 effort = position of the first correct one, 21 if none
  M3 unsafe   = suggestion s is a descendant of x in the cutoff HermiT hierarchy (cycle), or for x or one of its inferred
                descendants d, {d, s} plus their inferred ancestors contain a told disjoint pair (d becomes
                unsatisfiable); share of the top 10
  M4 anatomy  = top-10 wrong suggestions: redundant (inferred ancestor at the cutoff) > near miss (parent, child or sibling
                of a correct class in the Nov 2026 release) > transient (parent of x in an intermediate yearly snapshot,
                gone in Nov 2026) > unsupported
Writes results/study_d/measures/q_<C>_<n>.csv.gz (one row per query and model) and s_<C>_<n>.csv.gz (top-10 suggestions
with flags). Needs the raw snapshots (code/prepare/fetch_snapshots.py).
usage: measures.py C n"""
import sys, os, glob, collections, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "study_a"))
import paths, queries as AQ
import lzma, pickle
C, n = sys.argv[1], sys.argv[2]
OUT = os.path.join(paths.RESULTS, "study_d", "measures"); os.makedirs(OUT, exist_ok=True); LD = os.path.join(paths.RESULTS, "study_d", "lists")
D = pickle.loads(lzma.decompress(open(os.path.join(paths.DATA, "studyB.pkl.xz"), "rb").read()))
Q = D["queries"][C][n]; del D
L = pd.concat([pd.read_csv(f) for f in glob.glob(f"{LD}/{C}_std_*_{n}.csv.gz")] + [pd.read_csv(f"{LD}/{C}_training_free_{n}.csv.gz")], ignore_index=True)
rC, pC, _ = AQ.load(C, n); cC = set(rC["ids"]); _, pcur, _ = AQ.load("cur", n)
I = rC["ids"]; DJ = collections.defaultdict(set)
for a, b in rC["disj"]: DJ[I[a]].add(I[b]); DJ[I[b]].add(I[a])
H = AQ.HIER[C][n]; assert "_failed" not in H
DN = collections.defaultdict(set)
for a, bs in H.items():
    for b in bs: DN[b].add(a)
def clos(c, E):
    d, fr = set(), [c]
    while fr:
        nx = []
        for x in fr:
            for y in E.get(x, ()):
                if y not in d and y != c: d.add(y); nx.append(y)
        fr = nx
    return d
_anc = {}
def anc(c):
    if c not in _anc: _anc[c] = clos(c, H)
    return _anc[c]
inter = collections.defaultdict(set)   # parents of x in intermediate yearly snapshots
for Dd in [d for d in AQ.YEARS if d > C and AQ.IDX[d].get(n, {}).get("ok")]:
    _, pD, _ = AQ.load(Dd, n)
    for x in {t["x"] for k in "AB" for t in Q[k]}:
        inter[x] |= pD.get(x, set())
chcur = collections.defaultdict(set)
for a, bs in pcur.items():
    for b in bs: chcur[b].add(a)
qrows, srows = [], []
for task in "AB":
    for i, t in enumerate(Q[task]):
        x = t["x"]; corr = (pcur.get(x, set()) & cC) - pC.get(x, set())
        near = set()
        for c in corr:
            near |= pcur.get(c, set()) | chcur.get(c, set())
            for p in pcur.get(c, set()): near |= chcur.get(p, set())
        near -= corr
        trans = (inter.get(x, set()) - pcur.get(x, set())) & cC
        ax = anc(x) | {x} if x in cC else {x}; dx = clos(x, DN) if x in cC else set()
        up = set(ax)
        for d in dx: up |= anc(d) | {d}
        for m, s20 in L[(L.task == task) & (L.i == i)][["model", "top20"]].itertuples(index=False):
            S = s20.split("|"); first = next((j + 1 for j, s in enumerate(S) if s in corr), 21)
            r = dict(cutoff=C, ontology=n, task=task, i=i, x=x, model=m, kind=t.get("kind", "new"), entailed=t.get("entailed"), h=t["h"], n_correct=len(corr),
                     hit5=first <= 5, hit10=first <= 10, hit20=first <= 20, effort=first, unsafe=0, a_red=0, b_near=0, c_trans=0, d_uns=0, n_wrong=0)
            for j, s in enumerate(S[:10]):
                cyc = s in dx; Ss = {s} | anc(s); DS = set().union(*(DJ[u] for u in Ss if u in DJ)); dis = bool(DS & up)
                ok = s in corr
                cat = "" if ok else ("a" if s in ax else "b" if s in near else "c" if s in trans else "d")
                r["unsafe"] += cyc or dis
                if not ok:
                    r["n_wrong"] += 1; r[{"a": "a_red", "b": "b_near", "c": "c_trans", "d": "d_uns"}[cat]] += 1
                srows.append(dict(cutoff=C, ontology=n, task=task, i=i, x=x, model=m, pos=j + 1, s=s, correct=ok, cat=cat, cycle=cyc, disjoint=dis))
            r["unsafe"] /= 10; qrows.append(r)
pd.DataFrame(qrows).to_csv(f"{OUT}/q_{C}_{n}.csv.gz", index=False, compression="gzip")
pd.DataFrame(srows).to_csv(f"{OUT}/s_{C}_{n}.csv.gz", index=False, compression="gzip")
print(C, n, len(qrows), len(srows), flush=True)
