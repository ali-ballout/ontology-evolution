# Study E protocol: does a larger graph neural network beat the tuned ranker?

Fixed before any Study E training. Tests whether the rejection of H5 (Study B: the tuned non-neural ranker matches the
GNN) depends on the one GNN size used in Studies B–D. Any deviation is reported as such.

## Fixed inputs

- Cutoff 1 July 2023; the four std folds; the shared Study A queries; data/studyB.pkl.xz and the name embeddings (data/emb_*.npz) unchanged.
- Training regime R of Study B (deletion only): 10% of is-a links hidden on a training snapshot, batch 8, Adam
  (learning rate 3e-3), 3,000 steps, random name withholding, the same sampling code. Only depth and width vary.
- Comparator: the Study B tuned ranker GBR_del on the same folds and queries (its existing rank files; not rerun).

## Sizes

(layers L, hidden width d): L2d24, L4d24 (the Study B network), L6d24, L4d48, L4d96. Every other setting is as in Study B.

## Selection, without test ontologies

For each fold, the 20% of training ontologies that the tuned ranker held out for its own tuning (same split, rng 100 + fold)
are held out again. Each size is trained once (seed 0) on the remaining training ontologies and scored on the held-out
ones: in each, 10% of is-a links hidden (fixed rng), up to 100 hidden links, the child as query, every class a candidate,
other parents removed, ties counted half. The size with the highest validation MRR (mean over held-out ontologies) is the
**selected size** for that fold; ties go to the smaller network.

## Test models

Each size is trained on all of the fold's training ontologies with seeds 0, 1 and 2 (60 trainings) and evaluated on the
fold's test ontologies with the Study B evaluation (filtered ranks; MRR per ontology, mean over the three seeds).
The L4d24 models retrain the Study B network with the same code path; their agreement with the Study B results is reported
as a check.

## Comparisons (per ontology, ontology bootstrap 95% interval, 4,000 resamples; cluster bootstrap where >= 8 clusters)

Views: non-entailed re-parenting and new-class placement (the H5 views); entailed re-parenting reported descriptively.

1. **Primary:** the selected-size GNN (per fold) against the tuned ranker.
2. The best size on the test set, chosen after the fact, against the tuned ranker (reported, labelled as not deployable).
3. Each size against L4d24.

## Criterion

H5 stays rejected if, for the selected-size GNN, the 95% interval of (GNN − tuned ranker) includes zero or lies below
zero on both views. If the interval lies above zero on either view, the article reports that a larger network beats
the tuned ranker on that view and revises the H5 statement and the claims built on it. Comparison 2 cannot change the
verdict on its own; it is reported to show how much size could matter at best.

## Code and outputs

code/run_e.py (queue), code/elib.py, code/train_e.py, code/eval_val_e.py, code/eval_test_e.py; results/study_e/val and
results/study_e/ranks; analysis code/analysis/analyze_e.py.

## Not covered

One cutoff; one training budget and learning rate for every size (a larger network might need more steps or a different
rate); no other architectures.
