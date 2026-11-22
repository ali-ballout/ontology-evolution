# Study B protocol: what transfers

Fixed before any Study B experiment was run. Tests H1 (transfer, including to ontology families absent from training),
H4 (history supervision) and H5 (graph architecture against a tuned non-neural ranker), with input ablations. Any
deviation is reported as such.

## Fixed inputs (data/studyB.pkl.xz, data/emb_*.npz)

- Cutoffs 1 July 2019, 2021, 2023; for each ontology, the latest release on or before the cutoff (34 / 41 / 41 ontologies).
- Test queries: the shared Study A samples (own namespace; links still present in the November 2026 release; up to 150
  per horizon bin and task). Re-parenting targets carry kind (moved / added) and entailment by the cutoff snapshot
  (HermiT). New classes enter as isolated named nodes; parents must exist at the cutoff.
- Test ontologies: at least 10 queries in a task (28 / 30 / 25).
- Histories (2021, 2023): r1 = the snapshot two years before the cutoff (at least 180 days older than rC); links added
  between existing classes (moved / added, entailed by r1 or not) and new-class placements. Nothing dated after the cutoff.
- Families: 0.5% class / is-a overlap, connected components, at each cutoff (block 17 / 23 / 23 ontologies).
- Folds: std = 4 folds (GO, Uberon, CL in different folds); famA = the block's test ontologies, trained only outside the
  block; famB = the independent test ontologies, trained on all others; famBsize = famB with training cut to famA's size.

## Models

- GNN (d 24, 4 layers, Adam 3·10⁻³, batch 8, 3,000 steps, random name withholding): R deletion only (all cutoffs, all
  fold sets); M / E / N / X history regimes (2021, 2023, std folds); NAMES / STRUCT / NOREL input ablations (2023, std
  folds). Three seeds each (195 trainings).
- Tuned non-neural ranker: LightGBM LambdaRank reranking the hybrid's top 300 with the same inputs; tuned on 20% held-out
  training ontologies (8-setting grid); signals deletion (GBR_del) and deletion + history (GBR_delhist).
- Training-free baselines: the Study A ranks on the same queries.

## Comparisons (per ontology, then mean over ontologies; ontology bootstrap intervals; per-seed values)

1. H1: GNN R vs hybrid on non-entailed re-parenting and new-class placement, at each cutoff, for std, famA and famB.
   Rejected if the famA interval includes zero at a majority of cutoffs for both tasks. famB vs famBsize separates the
   effect of related training ontologies from training size.
2. H4: M, E, N, X vs R on non-entailed re-parenting and new classes (2021, 2023). Rejected if no history regime beats R on
   non-entailed edits at a majority of cutoffs. X vs E tests whether history helps only through entailed edits.
3. H5: GNN R vs GBR_del and GNN M vs GBR_delhist on both tasks. Rejected if the ranker is within the GNN's interval on both.
4. Ablations: NAMES, STRUCT, NOREL vs R.

All results are reported, including negative ones; per-ontology tables and seed spread are kept.
