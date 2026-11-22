"""Cross-encoder comparison (100 candidates per query): MRR, Hits@1, Hits@10 per ontology set, and per-ontology wins of the
GNN (mean over its three seeds) against each text model. Writes reports/crossencoder_<PROTO>.csv.
usage: crossencoder_summary.py [hard|random]"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); import paths
PROTO = sys.argv[1] if len(sys.argv) > 1 else "hard"; R_ = os.path.join(paths.RESULTS, "deletion", "crossencoder")
R = pd.concat([pd.read_csv(f) for f in (([f"{R_}/hard_ranks.csv"] if PROTO == "hard" else []) + sorted(glob.glob(f"{R_}/{PROTO}_*_ranks.csv")))]).drop_duplicates(["ontology", "model", "i"])
R["model"] = R.model.str.replace(r"GNN \+ names \(seed \d\)", "GNN + names, learned jointly (3 seeds)", regex=True)
R["rr"] = 1 / R["rank"]; R["h1"] = R["rank"] <= 1; R["h10"] = R["rank"] <= 10; LARGE = ["go", "uberon", "cl"]
R["set"] = np.where(R.ontology.isin(LARGE), "GO, Uberon, CL", "fold 0 (8 ontologies)")
po = R.groupby(["set", "ontology", "model"])[["rr", "h1", "h10"]].mean().reset_index()
s = po.groupby(["set", "model"])[["rr", "h1", "h10"]].mean().round(3); s.columns = ["MRR", "Hits@1", "Hits@10"]
print(f"\n=== {PROTO} negatives (100 candidates per query; mean over ontologies)\n", s.sort_values(["set", "MRR"]).to_string())
w = po.pivot_table(index="ontology", columns="model", values="rr"); G = "GNN + names, learned jointly (3 seeds)"
for m in [c for c in w.columns if c.startswith("BERTSubs") or c.startswith("look")]:
    d = w[G] - w[m]; print(f"GNN vs {m}: wins {int((d > 0).sum())}/{len(d)}, median margin {d.median():.3f}, smallest {d.min():.3f}")
os.makedirs(paths.REPORTS, exist_ok=True); w.round(3).to_csv(os.path.join(paths.REPORTS, f"crossencoder_{PROTO}.csv"))
