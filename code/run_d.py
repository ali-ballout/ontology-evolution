"""Run the learned-ranker part of Study D as a resumable queue. usage: run_d.py [--workers N] [--dry]
  12 jobs: ranker_c.py h1 <cutoff> std <fold> (tuned ranker on the std folds; Study C's code and grid)
  12 jobs: export_d.py <cutoff> <fold> (top-20 lists), each started when its ranker job has finished
The training-free lists come from baseline_lists.py."""
import os, sys, subprocess, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=5); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
LD = os.path.join(ROOT, "results", "study_d", "logs"); os.makedirs(LD, exist_ok=True)
CUTS = ["2019-07-01", "2021-07-01", "2023-07-01"]
J = [dict(args=["ranker_c.py", "h1", c, "std", str(f)], key=f"rank_{c}_{f}", deps=[]) for c in CUTS for f in range(4)]
J += [dict(args=["export_d.py", c, str(f)], key=f"export_{c}_{f}", deps=[f"rank_{c}_{f}"]) for c in CUTS for f in range(4)]
if a.dry: print(f"{len(J)} jobs ({sum(j['key'].startswith('rank') for j in J)} ranker, {sum(j['key'].startswith('export') for j in J)} export)"); sys.exit(0)
done = lambda k: os.path.exists(os.path.join(LD, k + ".done")); running, failed, t0 = [], [], time.time()
while True:
    for p, j in list(running):
        if p.poll() is not None:
            running.remove((p, j))
            if p.returncode == 0: open(os.path.join(LD, j["key"] + ".done"), "w").write("ok")
            else: failed.append(j["key"])
    busy = {j["key"] for _, j in running}
    ready = [j for j in J if not done(j["key"]) and j["key"] not in busy and j["key"] not in failed and all(done(d) for d in j["deps"])]
    ready.sort(key=lambda j: not j["key"].startswith("export"))
    for j in ready[: max(0, a.workers - len(running))]:
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", RANKS_DIR=os.path.join(ROOT, "results", "study_d", "ranker_ranks"))
        running.append((subprocess.Popen([sys.executable] + j["args"], cwd=HERE, stdout=open(os.path.join(LD, j["key"] + ".txt"), "w"), stderr=subprocess.STDOUT, env=env), j))
    nd = sum(done(j["key"]) for j in J)
    msg = f"{time.strftime('%H:%M:%S')}  elapsed {(time.time() - t0) / 3600:.2f} h | jobs {nd}/{len(J)} | running {len(running)} | failed {len(failed)}"
    open(os.path.join(ROOT, "results", "study_d", "progress.txt"), "w").write(msg + ("\nFAILED: " + " ".join(failed) if failed else "") + "\n")
    if not running and not ready: print(msg); break
    time.sleep(20)
print("finished;", "failures listed in results/study_d/progress.txt (run again to retry them)" if failed else "no failures")
