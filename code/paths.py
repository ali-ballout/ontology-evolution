"""Locations used by the scripts outside the learned-ranker code, relative to the package root.
Raw OBO releases are not part of the package; code/prepare/fetch_snapshots.py downloads them into snapshots/ (or the
directory given by SNAPSHOT_DIR) as snapshots/<snapshot>/<ontology>.obo, with <snapshot> = 2019-07-01 ... 2025-07-01 or
current. HermiT is taken from HERMIT_DIR (a directory containing HermiT.jar) or from the copy bundled with owlready2."""
import os, csv, lzma, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data"); RESULTS = os.path.join(ROOT, "results"); REPORTS = os.path.join(ROOT, "reports")
SNAPSHOTS = os.environ.get("SNAPSHOT_DIR", os.path.join(ROOT, "snapshots"))
YEARS = [f"{y}-07-01" for y in range(2019, 2026)]; CUTS = ["2019-07-01", "2021-07-01", "2023-07-01"]; CURRENT = "2026-11-18"
def snapshot_file(snapshot, n): return os.path.join(SNAPSHOTS, snapshot, f"{n}.obo")
def snapshot_index():
    """{snapshot: {ontology: {"ok": True, "date": release date, "classes": n}}} from data/snapshots.csv (yearly snapshots)"""
    idx = {D: {} for D in YEARS}
    for r in csv.DictReader(open(os.path.join(DATA, "snapshots.csv"))):
        if r["snapshot"] in idx: idx[r["snapshot"]][r["ontology"]] = dict(ok=True, date=r["release_date"], classes=int(r["classes"]))
    return idx
def hierarchies():
    """HermiT hierarchies of the cutoff snapshots: {cutoff: {ontology: {class id: [direct inferred superclass ids]}}}"""
    return pickle.loads(lzma.decompress(open(os.path.join(DATA, "hier.pkl.xz"), "rb").read()))
def hermit_dir():
    d = os.environ.get("HERMIT_DIR")
    if d: return d
    import owlready2; return os.path.join(os.path.dirname(owlready2.__file__), "hermit")
