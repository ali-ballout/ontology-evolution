"""Run the whole Study B programme as a resumable job queue. usage: run_all.py [--workers N] [--only deletion,history,ablation,ranker]
Jobs (each skipped if its output exists, so the script can be stopped and restarted at any time):
  deletion  regime R, 3 cutoffs x (4 std folds + famA + famB + famBsize) x 3 seeds          63 trainings
  history   regimes M, E, N, X, cutoffs 2021 and 2023 x 4 std folds x 3 seeds               96 trainings
  ablation  NAMES, STRUCT, NOREL, cutoff 2023 x 4 std folds x 3 seeds                       36 trainings
  ranker    tuned non-neural ranker, 3 cutoffs x 4 std folds                                 12 jobs
  evaluate  one per (cutoff, foldset, fold), started once all its trainings are finished    21 jobs
Each job is a separate process with one thread; logs in results/study_b/logs/; progress in results/study_b/progress.txt."""
import os, sys, subprocess, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 3)); ap.add_argument("--only", default="deletion,history,ablation,ranker"); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
ONLY = set(a.only.split(",")); CUTS = ["2019-07-01", "2021-07-01", "2023-07-01"]; SEEDS = [0, 1, 2]
import lzma, pickle
FOLDS = pickle.loads(lzma.decompress(open(os.path.join(ROOT, "data", "studyB.pkl.xz"), "rb").read()))["folds"]
MD, RD, LD = (os.path.join(ROOT, "results", "study_b", x) for x in ("models", "ranks", "logs"))
for d in (MD, RD, LD): os.makedirs(d, exist_ok=True)
train, evals, rankers = [], {}, []
def T(c, fs, f, r, s):
    j = dict(kind="train", args=["train.py", c, fs, str(f), r, str(s)], out=os.path.join(MD, f"{c}_{fs}_{f}_{r}_s{s}.pt"), log=os.path.join(LD, f"train_{c}_{fs}_{f}_{r}_s{s}.txt"))
    train.append(j); evals.setdefault((c, fs, f), []).append(j)
if "deletion" in ONLY:
    for s in SEEDS:
        for c in CUTS:
            for fs in ("std", "famA", "famB", "famBsize"):
                for f in range(len(FOLDS[c][fs])): T(c, fs, f, "R", s)
if "history" in ONLY:
    for s in SEEDS:
        for c in CUTS[1:]:
            for r in ("M", "E", "N", "X"):
                for f in range(4): T(c, "std", f, r, s)
if "ablation" in ONLY:
    for s in SEEDS:
        for r in ("NAMES", "STRUCT", "NOREL"):
            for f in range(4): T(CUTS[2], "std", f, r, s)
if "ranker" in ONLY:
    for c in CUTS:
        for f in range(4): rankers.append(dict(kind="ranker", args=["ranker.py", c, str(f)], out=None, done=os.path.join(LD, f"ranker_{c}_{f}.done"), log=os.path.join(LD, f"ranker_{c}_{f}.txt")))
EV = [dict(kind="eval", key=k, deps=v, args=["evaluate.py", k[0], k[1], str(k[2])], done=os.path.join(LD, f"eval_{k[0]}_{k[1]}_{k[2]}.done"), log=os.path.join(LD, f"eval_{k[0]}_{k[1]}_{k[2]}.txt")) for k, v in evals.items()]
if a.dry: print(f"trainings {len(train)}, evaluations {len(EV)}, rankers {len(rankers)}"); sys.exit(0)
def finished(j):
    if j["kind"] == "train": return os.path.exists(j["out"]) or os.path.exists(j["log"] + ".skip")
    return os.path.exists(j["done"])
queue = [j for j in rankers[:2] + train + rankers[2:]] ; running = []; t0 = time.time(); failed = []
def launch(j):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.Popen([sys.executable] + j["args"], cwd=HERE, stdout=open(j["log"], "w"), stderr=subprocess.STDOUT, env=env); running.append((p, j))
while True:
    for p, j in list(running):
        if p.poll() is not None:
            running.remove((p, j))
            if p.returncode == 2 and j["kind"] == "train": open(j["log"] + ".skip", "w").write("no history examples")
            elif p.returncode != 0: failed.append(" ".join(j["args"]))
            elif j["kind"] in ("eval", "ranker"): open(j["done"], "w").write("ok")
    todo = [j for j in queue if not finished(j) and all(j is not r for _, r in running) and " ".join(j["args"]) not in failed]
    ready_ev = [e for e in EV if not finished(e) and all(finished(d) for d in e["deps"]) and all(e is not r for _, r in running) and " ".join(e["args"]) not in failed]
    for j in (ready_ev + todo)[: max(0, a.workers - len(running))]: launch(j)
    ntr = sum(finished(j) for j in train); nev = sum(finished(e) for e in EV); nrk = sum(finished(r) for r in rankers)
    msg = f"{time.strftime('%H:%M:%S')}  elapsed {(time.time() - t0) / 3600:.2f} h | trainings {ntr}/{len(train)} | evaluations {nev}/{len(EV)} | rankers {nrk}/{len(rankers)} | running {len(running)} | failed {len(failed)}"
    open(os.path.join(ROOT, "results", "study_b", "progress.txt"), "w").write(msg + ("\nFAILED:\n" + "\n".join(failed) if failed else "") + "\n")
    if not running and not ready_ev and not todo: print(msg); break
    time.sleep(20)
print("finished;", "failures listed in results/study_b/progress.txt" if failed else "no failures")
