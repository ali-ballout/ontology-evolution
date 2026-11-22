# Study C protocol: combining reasoning with learned ranking

Fixed before any Study C experiment was run. Tests H6 (combination) and H1 for the tuned ranker. No strategy uses a
target's entailment status; only signals available when a suggestion is made.

## Part 1: H1 for the tuned ranker (deletion-trained)

- Design: the tuned ranker (GBR_del_c, the grid below) on the family-held-out folds (famA, famB, famBsize) at all three
  cutoffs, with the shared queries.
- Criterion: rejected if the famA interval against the hybrid includes zero at a majority of cutoffs for both tasks
  (non-entailed re-parenting, new classes).

## Part 2: H6, reasoner plus learned ranker (re-parenting)

- Scope: re-parenting only, because a reasoner says nothing about a class that arrives with only a name.
- Cutoffs and folds: 2021 and 2023, std folds (calibration needs pre-cutoff edits with known entailment, from the
  r1 → cutoff histories classified by HermiT on r1; 2019 has no earlier history).
- Reasoner signal at prediction time (HermiT on the cutoff snapshot): for each candidate, whether it is an inferred ancestor
  of the query, its inferred distance, and whether it is an inferred descendant (adding it would create a cycle).
- Candidates: the hybrid's top 300 plus every inferred ancestor of the query; coverage is reported.
- Ranker grid: num_leaves {7, 15, 63} × learning rate {0.02, 0.05, 0.1} × trees {100, 200, 500} (27 settings), tuned only on
  20% held-out training ontologies.

| Label | Strategy (all rules fixed before any test result) |
|---|---|
| R0 | Reasoner only: inferred ancestors nearest first, then the hybrid (the Study A reasoner ranker) |
| L0 | Learned only: tuned ranker, deletion + history signal (L0d: deletion only) |
| S1 | Reasoner first: inferred ancestors (nearest first, ties by learned score), then the learned order |
| S2 | Learned order with inferred descendants (cycles) removed |
| **S3 (primary)** | Reasoner-aware ranker: the tuned ranker with the three reasoner features added, trained on deletion + history. For deletion examples the features come from the transitive closure of the observed graph, so the hidden link cannot leak; for history examples from HermiT on r1 |
| S4 | Calibrated fusion: standardised learned score + w × reasoner ancestor score, w chosen on held-out training ontologies' history edits |
| Oracle | Not deployable: R0's rank if the target is entailed, else L0's |

- H6 criterion for S3, at each cutoff: (1) on all re-parenting edits, S3 beats both R0 and L0 (both ontology-bootstrap
  intervals exclude 0); (2) on non-entailed edits, the lower interval bound of S3 − L0 is above −0.01 MRR. H6 is supported
  if S3 succeeds at both cutoffs. S1, S2 and S4 are reported against the same criterion; the oracle gap is reported.
- Metrics: MRR, Hits@1, Hits@10, coverage; all, entailed and non-entailed edits; per ontology.

## Part 3: runtime

For each 2023 test ontology: HermiT classification time of the snapshot; feature extraction and inference per query for
the ranker and the GNN (CPU and GPU).

## Statistics

Per-ontology paired differences, ontology bootstrap; cluster bootstrap over overlap clusters (at least 20% shared classes
or links) at each cutoff where there are at least 8 clusters; block and independent ontologies reported separately. Every
outcome is reported, including failure of S3.
