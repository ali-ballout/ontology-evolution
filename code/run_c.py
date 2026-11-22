"""Run Study C as a resumable job queue. usage: run_c.py [--workers N] [--dry]
  h1      tuned ranker on famA, famB, famBsize at 2019, 2021, 2023                 9 jobs
  h6      reasoner + ranker strategies, std folds 0-3 at 2021 and 2023            8 jobs
  timing  runtime per 2023 test ontology (after the 2023 h6 jobs)                 1 job
Finished jobs are skipped on restart. Logs in results/study_c/logs/, progress in results/study_c/progress.txt. Each job uses 2 threads."""
import os, sys, subprocess, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 4)); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
LD = os.path.join(ROOT, "results", "study_c", "logs"); os.makedirs(LD, exist_ok=True)
J = [dict(args=["ranker_c.py", "h1", c, fs, "0"], key=f"h1_{c}_{fs}_0") for c in ["2023-07-01", "2021-07-01", "2019-07-01"] for fs in ["famA", "famB", "famBsize"]]
J += [dict(args=["ranker_c.py", "h6", c, "std", str(f)], key=f"h6_{c}_std_{f}") for c in ["2023-07-01", "2021-07-01"] for f in range(4)]
T = dict(args=["timing_c.py"], key="timing", deps=[f"h6_2023-07-01_std_{f}" for f in range(4)])
if a.dry: print(f"{len(J)} ranker jobs + timing"); sys.exit(0)
done = lambda k: os.path.exists(os.path.join(LD, k + ".done")); running, failed, t0 = [], [], time.time()
while True:
    for p, j in list(running):
        if p.poll() is not None:
            running.remove((p, j))
            if p.returncode == 0: open(os.path.join(LD, j["key"] + ".done"), "w").write("ok")
            else: failed.append(j["key"])
    busy = {j["key"] for _, j in running}
    ready = [j for j in J if not done(j["key"]) and j["key"] not in busy and j["key"] not in failed]
    if not done("timing") and "timing" not in busy and "timing" not in failed and all(done(d) for d in T["deps"]): ready = [T] + ready
    for j in ready[: max(0, a.workers - len(running))]:
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        running.append((subprocess.Popen([sys.executable] + j["args"], cwd=HERE, stdout=open(os.path.join(LD, j["key"] + ".txt"), "w"), stderr=subprocess.STDOUT, env=env), j))
    nd = sum(done(j["key"]) for j in J)
    msg = f"{time.strftime('%H:%M:%S')}  elapsed {(time.time() - t0) / 3600:.2f} h | ranker jobs {nd}/{len(J)} | timing {'done' if done('timing') else 'pending'} | running {len(running)} | failed {len(failed)}"
    open(os.path.join(ROOT, "results", "study_c", "progress.txt"), "w").write(msg + ("\nFAILED: " + " ".join(failed) if failed else "") + "\n")
    if not running and not ready: print(msg); break
    time.sleep(20)
print("finished;", "failures listed in results/study_c/progress.txt (run again to retry them)" if failed else "no failures")
