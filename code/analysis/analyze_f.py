"""Study F analysis against protocols/study_f.md.
F1: per-ontology MRR under random deletion (cutoff snapshot, 10% hidden) against MRR on real future edits, same test
ontologies, snapshots and models (Study B regime R, std folds, mean over 3 seeds); hybrid from the same views.
F2: gain of the sentence encoder in name search (sum vs TF-IDF only) by cutoff, and its difference between cutoffs.
Ontologies with >= 10 queries in every setting compared; ontology bootstrap 95% intervals (4,000 resamples).
usage: analyze_f.py   writes reports/study_f_comparisons.csv; the printed report goes to stdout."""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
RES = paths.RESULTS; os.makedirs(paths.REPORTS, exist_ok=True); OUT = []
def boot(x, B=4000):
    r = np.random.default_rng(0); return np.percentile([r.choice(x, len(x)).mean() for _ in range(B)], [2.5, 97.5])
def per_ont(d, col="rr"):
    nq = d.groupby("ontology").i.nunique(); return d[d.ontology.isin(nq[nq >= 10].index)].groupby("ontology")[col].mean()
def report(label, C, d):
    d = d.dropna(); lo, hi = boot(d.values)
    print(f"  {C[:4]} {label}: mean {d.mean():+.3f}, median {d.median():+.3f}, positive {int((d > 0).sum())}/{len(d)}, CI {lo:+.3f}..{hi:+.3f}")
    OUT.append(dict(analysis=label, cutoff=C, n=len(d), mean=d.mean(), median=d.median(), positive=int((d > 0).sum()), ci_lo=lo, ci_hi=hi))
# ---------- F1
D = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_f/deletion_ranks/*.csv.gz")], ignore_index=True); D["rr"] = 1 / D["rank"]
G = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_b/ranks/*_std_*.csv.gz") if "__ranker" not in f], ignore_index=True)
G = G[G.regime == "R"].copy(); G["rr"] = 1 / G["rank"]
A = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_a/ranks/*.csv") if os.path.getsize(f) > 20], ignore_index=True)
A = A[A.model == "hybrid"].drop(columns=["kind", "h", "entailed"]); A["rr"] = 1 / A["rank"]
meta = G.drop_duplicates(["cutoff", "ontology", "task", "i"])[["cutoff", "ontology", "task", "i", "entailed", "in_training"]]
A = A.merge(meta, on=["cutoff", "ontology", "task", "i"])
FV = {"future non-entailed re-parenting": lambda d: d[(d.task == "A") & (d.entailed == False)], "future re-parenting (all)": lambda d: d[d.task == "A"], "future new classes": lambda d: d[d.task == "B"]}
for excl in (True, False):
    print(f"\n== F1 {'primary: queries whose link is in training material excluded' if excl else 'sensitivity: all queries'}")
    for C in sorted(D.cutoff.unique()):
        d = D[(D.cutoff == C) & (~D.in_training if excl else True)]; g = G[(G.cutoff == C) & (~G.in_training if excl else True)]; a = A[(A.cutoff == C) & (~A.in_training if excl else True)]
        gd = per_ont(d[d.model.str.startswith("GNN")].groupby(["ontology", "i"]).rr.mean().reset_index()); hd = per_ont(d[d.model == "hybrid"]); nd = per_ont(d[d.model == "names only"])
        print(f"  {C[:4]} deletion: GNN {gd.mean():.3f}, hybrid {hd.mean():.3f}, name search {nd.mean():.3f} ({len(gd)} ontologies)")
        OUT.append(dict(analysis=f"deletion levels{'' if excl else ' (all)'}", cutoff=C, n=len(gd), gnn=gd.mean(), hybrid=hd.mean(), names=nd.mean()))
        for v, f in FV.items():
            gf = per_ont(f(g).groupby(["ontology", "i"]).rr.mean().reset_index()); hf = per_ont(f(a))
            ok = gd.index.intersection(gf.index).intersection(hf.index)
            print(f"  {C[:4]} {v}: GNN {gf[ok].mean():.3f} (deletion {gd[ok].mean():.3f}), GNN−hybrid {(gf - hf)[ok].mean():+.3f} (deletion {(gd - hd)[ok].mean():+.3f})")
            sfx = "" if excl else " (all)"
            report(f"GNN deletion − {v}{sfx}", C, (gd - gf)[ok]); report(f"(GNN−hybrid) deletion − {v}{sfx}", C, ((gd - hd) - (gf - hf))[ok])
# ---------- F1 reconciliation with the main results (Study B report, all queries, all ontologies with >= 10 queries)
print("\n== F1 reconciliation: future-edit GNN results in the main analysis and in the paired set (overlapping queries removed)")
for C in sorted(D.cutoff.unique()):
    d = D[D.cutoff == C]; g = G[G.cutoff == C]; nd = d.drop_duplicates(["ontology", "i"]); x = nd.groupby("ontology").in_training.agg(["size", "sum"])
    left = (x["size"] - x["sum"]); lost = sorted(left[left < 10].index)
    print(f"  {C[:4]} deletion queries: {len(nd)} in {nd.ontology.nunique()} ontologies; {int(nd.in_training.sum())} ({nd.in_training.mean():.1%}) hide a link that is also in a training ontology; ontologies left with < 10 queries: {', '.join(f'{o} ({int(x.loc[o, chr(115) + chr(117) + chr(109)])} of {int(x.loc[o, chr(115) + chr(105) + chr(122) + chr(101)])} removed)' for o in lost) or 'none'}")
    gd = per_ont(d[~d.in_training & d.model.str.startswith("GNN")].groupby(["ontology", "i"]).rr.mean().reset_index())
    for v, f in FV.items():
        fa = f(g).groupby(["ontology", "i", "in_training"]).rr.mean().reset_index(); main = per_ont(fa); fr = fa[~fa.in_training]; pr = per_ont(fr); ok = pr.index.intersection(gd.index)
        nqm = int(fa[fa.ontology.isin(main.index)].shape[0]); nqp = int(fr[fr.ontology.isin(ok)].shape[0])
        print(f"  {C[:4]} {v}: main {main.mean():.3f} ({len(main)} ontologies, {nqm} queries) -> paired set {pr[ok].mean():.3f} ({len(ok)} ontologies, {nqp} queries); future queries removed as overlapping {int(fa[fa.ontology.isin(main.index)].in_training.sum())}; ontologies not in the paired set: {', '.join(sorted(set(main.index) - set(ok))) or 'none'}; main result on the paired ontologies {main[main.index.intersection(ok)].mean():.3f}")
        OUT.append(dict(analysis=f"reconciliation, {v}", cutoff=C, n=len(ok), main_mrr=main.mean(), main_ontologies=len(main), main_queries=nqm, paired_mrr=pr[ok].mean(), paired_queries=nqp))
# ---------- F2
E = pd.read_csv(f"{RES}/study_f/encoder_ranks.csv.gz"); E["rr"] = 1 / E["rank"]
N = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_a/ranks/*.csv") if os.path.getsize(f) > 20], ignore_index=True)
N = N[N.model == "names only"][["cutoff", "ontology", "task", "i", "rank"]]
chk = E[E.signal == "sum"].merge(N, on=["cutoff", "ontology", "task", "i"], suffixes=("", "_a"))
print(f"\n== F2 check: sum signal against Study A name search, {len(chk)} queries, max |rank difference| {np.abs(chk['rank'] - chk['rank_a']).max():.1f}")
EV = {"non-entailed re-parenting": lambda d: d[(d.task == "A") & (d.entailed == False)], "new classes": lambda d: d[d.task == "B"]}
gains = {}
for v, f in EV.items():
    for C in sorted(E.cutoff.unique()):
        e = f(E[E.cutoff == C]); s = per_ont(e[e.signal == "sum"]); t = per_ont(e[e.signal == "tfidf"]); m = per_ont(e[e.signal == "encoder"])
        print(f"  {C[:4]} {v}: TF-IDF {t.mean():.3f}, encoder {m.mean():.3f}, sum {s.mean():.3f}")
        gains[(v, C)] = (s - t).dropna(); report(f"encoder gain (sum − TF-IDF), {v}", C, gains[(v, C)])
    Cs = sorted(E.cutoff.unique()); last = Cs[-1]
    for C in Cs[:-1]:
        a, b = gains[(v, C)].values, gains[(v, last)].values; r = np.random.default_rng(0)
        bs = [r.choice(a, len(a)).mean() - r.choice(b, len(b)).mean() for _ in range(4000)]; lo, hi = np.percentile(bs, [2.5, 97.5]); dm = a.mean() - b.mean()
        print(f"  gain {C[:4]} − gain {last[:4]}, {v}: {dm:+.3f}, CI {lo:+.3f}..{hi:+.3f}")
        OUT.append(dict(analysis=f"encoder gain difference {C[:4]}−{last[:4]}, {v}", cutoff=C, n=len(a), mean=dm, ci_lo=lo, ci_hi=hi))
# ---------- F3
T = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_f/tfidf_ranks/*.csv.gz")], ignore_index=True); T["rr"] = 1 / T["rank"]
KB = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RES}/study_b/ranks/*_std_*__ranker.csv.gz")], ignore_index=True); KB = KB[KB.regime == "GBR_del"].copy(); KB["rr"] = 1 / KB["rank"]
HA = A.copy()   # Study A hybrid with the encoder, entailment merged above
print("\n== F3: tuned ranker minus hybrid, std folds, without the sentence encoder (TF-IDF only) and with it (Study B)")
def wins(d): return f"wins {int((d > 0).sum())}/{len(d)}"
for C in sorted(T.cutoff.unique()):
    t = T[T.cutoff == C]
    for v, f in EV.items():
        rt, ht = per_ont(f(t[t.regime == "GBR_del"])), per_ont(f(t[t.regime == "hybrid"])); re_, he = per_ont(f(KB[KB.cutoff == C])), per_ont(f(HA[HA.cutoff == C]))
        d = (rt - ht).dropna(); print(f"  {C[:4]} {v}: TF-IDF only: ranker {rt[d.index].mean():.3f}, hybrid {ht[d.index].mean():.3f}, {wins(d)}; with encoder: ranker {re_[d.index].mean():.3f}, hybrid {he[d.index].mean():.3f}")
        report(f"ranker − hybrid, TF-IDF only, {v}", C, d); report(f"ranker − hybrid, with encoder, {v}", C, (re_ - he).dropna()); report(f"TF-IDF-only ranker − hybrid with encoder, {v}", C, (rt - he).dropna())
pd.DataFrame(OUT).to_csv(os.path.join(paths.REPORTS, "study_f_comparisons.csv"), index=False)
