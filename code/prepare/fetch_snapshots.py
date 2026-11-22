"""Download the raw OBO releases listed in data/snapshots.csv into snapshots/<snapshot>/<ontology>.obo (or SNAPSHOT_DIR)
and check each file against its recorded SHA-256.
Sources (column 'source'):
  obo-purl           versioned OBO PURL (url); for the current releases the PURL of the recorded data-version
  go-archive         Gene Ontology release archive (url)
  github-release     release asset of the ontology's GitHub repository at the recorded tag (url)
  git-tag            release file at the recorded tag (url; the upper-case file name is tried as well)
  git-commit         file 'path' at commit 'reference' of the repository
  git-latest-before  file 'path' at the latest commit of the repository on or before the snapshot date
A mismatching checksum is reported; the analyses in this package need only the derived data in data/ and results/.
usage: fetch_snapshots.py [--only ontology,...] [--snapshot 2019-07-01,...]"""
import os, sys, csv, hashlib, argparse, subprocess, tempfile, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
ap = argparse.ArgumentParser(); ap.add_argument("--only", default=""); ap.add_argument("--snapshot", default=""); a = ap.parse_args()
ONLY = set(filter(None, a.only.split(","))); SNAP = set(filter(None, a.snapshot.split(",")))
REPOS = os.path.join(tempfile.gettempdir(), "obo_repos")
def http(url, dst):
    try:
        with urllib.request.urlopen(url, timeout=300) as r, open(dst, "wb") as f: f.write(r.read())
        return os.path.getsize(dst) > 20000
    except Exception: return False
def repo(r):
    d = os.path.join(REPOS, r.replace("/", "__"))
    if not os.path.exists(d): subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", f"https://github.com/{r}.git", d], check=False)
    return d
def gitshow(r, commit, path, dst):
    out = subprocess.run(["git", "-C", repo(r), "show", f"{commit}:{path}"], capture_output=True).stdout
    if len(out) < 20000: return False
    open(dst, "wb").write(out); return True
bad = []
for row in csv.DictReader(open(os.path.join(paths.DATA, "snapshots.csv"))):
    n, S = row["ontology"], row["snapshot"]
    if (ONLY and n not in ONLY) or (SNAP and S not in SNAP): continue
    dst = paths.snapshot_file(S, n); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and hashlib.sha256(open(dst, "rb").read()).hexdigest() == row["sha256"]: continue
    src, ok = row["source"], False
    if src in ("obo-purl", "go-archive", "github-release"): ok = http(row["url"], dst)
    elif src == "git-tag": ok = http(row["url"], dst) or http(row["url"][:-len(n + ".obo")] + n.upper() + ".obo", dst)
    elif src == "git-commit": ok = gitshow(row["repository"], row["reference"], row["path"], dst)
    elif src == "git-latest-before":
        h = subprocess.run(["git", "-C", repo(row["repository"]), "log", "--all", f"--before={S}T23:59:59", "-1", "--format=%H", "--", row["path"]], capture_output=True, text=True).stdout.strip()
        ok = bool(h) and gitshow(row["repository"], h, row["path"], dst)
    match = ok and hashlib.sha256(open(dst, "rb").read()).hexdigest() == row["sha256"]
    if not match: bad.append((S, n, "download failed" if not ok else "checksum differs"))
    print(S, n, "ok" if match else bad[-1][2], flush=True)
print(f"{len(bad)} problems" + (":\n" + "\n".join(" ".join(b) for b in bad) if bad else ""))
