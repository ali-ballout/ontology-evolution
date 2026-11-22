"""Run the GNN size study as a resumable job queue. usage: run_e.py [--workers N] [--dry] [--cutoffs 2023-07-01[,...]]
Jobs (each skipped if its output exists, so the script can be stopped and restarted at any time):
  val trainings    5 sizes x 4 std folds x seed 0, on the training ontologies minus the held-out 20%        20
  val evaluations  one per fold, once its 5 val trainings are finished                                       4
  full trainings   5 sizes x 4 std folds x seeds 0-2, on all training ontologies                            60
  test evaluations one per fold, once its 15 full trainings are finished                                     4
(per cutoff; default cutoff 2023-07-01). Sizes (layers L, width d): L2d24, L4d24 (the Study B network), L6d24, L4d48, L4d96.
At most one width-96 training or evaluation runs at a time. Logs in results/study_e/logs/, progress in results/study_e/progress.txt. Device: E_DEVICE (default cuda when available)."""
import os, sys, subprocess, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--dry", action="store_true"); ap.add_argument("--cutoffs", default="2023-07-01"); ap.add_argument("--steps", type=int, default=3000, help=argparse.SUPPRESS); a = ap.parse_args()   # --steps: smoke tests only
CONFIGS = [(2, 24), (4, 24), (6, 24), (4, 48), (4, 96)]; COST = {(2, 24): 0.5, (4, 24): 1, (6, 24): 1.5, (4, 48): 2, (4, 96): 4}
MD, LD = os.path.join(ROOT, "results", "study_e", "models"), os.path.join(ROOT, "results", "study_e", "logs")
for d in (MD, LD): os.makedirs(d, exist_ok=True)
train, evals = [], []
def T(c, f, L, D, sp, s):
    j = dict(kind="train", cost=COST[(L, D)], args=["train_e.py", c, str(f), str(L), str(D), sp, str(s)] + ([str(a.steps)] if a.steps != 3000 else []), out=os.path.join(MD, f"{c}_std_{f}_L{L}d{D}_{sp}_s{s}.pt"), log=os.path.join(LD, f"train_{c}_{f}_L{L}d{D}_{sp}_s{s}.txt"))
    train.append(j); return j
for c in a.cutoffs.split(","):
    for f in range(4):
        deps = [T(c, f, L, D, "val", 0) for L, D in CONFIGS]
        evals.append(dict(kind="eval", deps=deps, args=["eval_val_e.py", c, str(f)], done=os.path.join(LD, f"val_{c}_{f}.done"), log=os.path.join(LD, f"val_{c}_{f}.txt")))
    for f in range(4):
        deps = [T(c, f, L, D, "full", s) for s in (0, 1, 2) for L, D in CONFIGS]
        evals.append(dict(kind="eval", deps=deps, args=["eval_test_e.py", c, str(f)], done=os.path.join(LD, f"test_{c}_{f}.done"), log=os.path.join(LD, f"test_{c}_{f}.txt")))
if a.dry: print(f"{len(train)} trainings ({sum(j['args'][5] == 'val' for j in train)} for selection), {len(evals)} evaluations; relative cost {sum(j['cost'] for j in train):.0f} Study-B trainings"); sys.exit(0)
def finished(j): return os.path.exists(j["out"]) if j["kind"] == "train" else os.path.exists(j["done"])
queue = sorted(train, key=lambda j: (j["args"][5] != "val", -j["cost"])); running = []; t0 = time.time(); failed = []
def launch(j):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.Popen([sys.executable] + j["args"], cwd=HERE, stdout=open(j["log"], "w"), stderr=subprocess.STDOUT, env=env); running.append((p, j))
while True:
    for p, j in list(running):
        if p.poll() is not None:
            running.remove((p, j))
            if p.returncode != 0: failed.append(" ".join(j["args"]))
            elif j["kind"] == "eval": open(j["done"], "w").write("ok")
    busy = lambda j: any(j is r for _, r in running) or " ".join(j["args"]) in failed
    ready = [e for e in evals if not finished(e) and not busy(e) and all(finished(d) for d in e["deps"])] + [j for j in queue if not finished(j) and not busy(j)]
    heavy = lambda j: j["kind"] == "eval" or j.get("cost", 0) >= 4   # at most one width-96 training or evaluation at a time (GPU memory)
    slots = max(0, a.workers - len(running)); nheavy = sum(heavy(r) for _, r in running)
    for j in ready:
        if slots == 0: break
        if heavy(j) and nheavy: continue
        launch(j); slots -= 1; nheavy += heavy(j)
    ntr = sum(finished(j) for j in train); nev = sum(finished(e) for e in evals)
    msg = f"{time.strftime('%H:%M:%S')}  elapsed {(time.time() - t0) / 3600:.2f} h | trainings {ntr}/{len(train)} | evaluations {nev}/{len(evals)} | running {len(running)} | failed {len(failed)}"
    open(os.path.join(ROOT, "results", "study_e", "progress.txt"), "w").write(msg + ("\nFAILED:\n" + "\n".join(failed) if failed else "") + "\n")
    if not running and not ready: print(msg + ("\nFAILED (run again to retry):\n" + "\n".join(failed) if failed else "\nno failures")); break
    time.sleep(5)
