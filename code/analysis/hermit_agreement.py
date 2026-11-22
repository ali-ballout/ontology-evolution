"""Agreement of the structural safety check (cycle or declared-disjoint pair) with HermiT on the validation sample
(results/study_d/hermit_check/validation.csv), with the Wilson 95% lower bound on agreement and, because the sample was
drawn by flag, the exact (Clopper-Pearson) two-sided 95% lower bound within each group.  usage: hermit_agreement.py"""
import os, sys, math, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
v = pd.read_csv(os.path.join(paths.RESULTS, "study_d", "hermit_check", "validation.csv")); n = len(v); k = int((v.flag == v.hermit_unsafe).sum()); p = k / n; zz = 1.959964
lo = (p + zz * zz / (2 * n) - zz * math.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n))) / (1 + zz * zz / n)
print(f"sample {n} ({int(v.flag.sum())} flagged, {int((~v.flag).sum())} not flagged) from {v.groupby(['cutoff', 'ontology']).ngroups} cutoff-ontology pairs; agreement {k}/{n}, Wilson 95% lower bound {lo:.1%}")
from scipy.stats import beta
for name, g in (("flagged", v[v.flag]), ("not flagged", v[~v.flag])):
    m = len(g); a = int((g.flag == g.hermit_unsafe).sum()); print(f"  {name}: agreement {a}/{m}, exact 95% lower bound {(beta.ppf(0.025, a, m - a + 1) if a else 0.0):.1%}")
u = v[v.hermit_unsafe]; print(f"HermiT-unsafe cases where the query stays satisfiable and a descendant does not: {int((~u.x_unsat).sum())} of {len(u)}")
