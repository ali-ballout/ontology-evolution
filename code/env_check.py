"""Check the Python environment and estimate run time. usage: env_check.py"""
import sys, os, time, importlib
print("Python", sys.version.split()[0], "| CPU cores", os.cpu_count())
missing = []
for m in ["numpy", "scipy", "pandas", "sklearn", "torch"]:
    try: mod = importlib.import_module(m); print(f"  {m:8s} {getattr(mod, '__version__', '?')}")
    except ImportError: print(f"  {m:8s} MISSING"); missing.append(m)
try: import lightgbm; print(f"  lightgbm {lightgbm.__version__}")
except ImportError: print("  lightgbm missing (optional, ~2 MB: pip install lightgbm); the ranker falls back to scikit-learn")
try:
    import psutil; gb = psutil.virtual_memory().total / 2**30; print(f"RAM {gb:.0f} GB")
except ImportError: gb = None; print("RAM unknown (psutil not installed)")
if missing: print("Install before running:", " ".join("scikit-learn" if m == "sklearn" else m for m in missing)); sys.exit(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
t = time.time(); from lib import DATA; print(f"data loaded in {time.time() - t:.0f}s")
import subprocess
t = time.time(); r = subprocess.run([sys.executable, "train.py", "2023-07-01", "std", "3", "R", "99", "100"], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
dt = time.time() - t; root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try: os.remove(os.path.join(root, "results", "study_b", "models", "2023-07-01_std_3_R_s99.pt"))
except OSError: pass
if r.returncode != 0: print("test training failed:\n", r.stdout[-2000:], r.stderr[-2000:]); sys.exit(1)
per = dt * 30 / 60; w = max(1, min((os.cpu_count() or 4) - 4, int(gb // 3) if gb else 8))
print(f"100 training steps: {dt:.0f}s -> about {per:.0f} min per full training run (3,000 steps)")
print(f"suggested workers: {w} (one thread and up to ~2.5 GB each); 195 trainings -> about {195 * per / w / 60:.1f} h, plus evaluation and rankers")
