# Study design: research question and hypotheses

Fixed before any experiment of Studies A–D was run. Each study's queries, folds, comparisons and criteria were then fixed in
its own protocol (study_a.md … study_d.md) before that study was run.

## Research question

To what extent can the evolution of an ontology be anticipated by models trained on other ontologies, what can logical
reasoning already establish, and how should the two support ontology maintenance?

## Hypotheses and rejection criteria

A hypothesis is rejected when its criterion fails. Rejections are reported, and the evaluation is not redesigned to
recover them.

| Hypothesis | Test | Rejected if |
|---|---|---|
| **H1 Transfer.** A model trained on other ontologies ranks real future parents better than the best training-free heuristic, including in ontology families absent from training. | Study B (GNN) and Study C (tuned ranker): family-held-out folds at every cutoff; learned ranker vs hybrid and names only, non-entailed re-parenting and new-class placement. | The ontology-level 95% interval of the difference from the hybrid includes zero on the held-out block at a majority of cutoffs, for both tasks. |
| **H2 Entailment structure.** A substantial, edit-type-dependent share of future subclass edits is already entailed by the release available at the time. | Study A: census by edit type, ontology, cutoff and horizon; HermiT on every cutoff snapshot. | The entailed share is below 10% for every edit type, or does not differ between edit types. |
| **H3 Complementarity.** Reasoning dominates on entailed edits; learning is the best tool on non-entailed edits. | Studies A–C: reasoner rankers vs learned rankers, split by entailment. | On non-entailed edits no learned model beats both the reasoner rankers and the heuristics at a majority of cutoffs. |
| **H4 History supervision.** Training on other ontologies' past edits improves prediction of non-entailed future edits beyond random-deletion training. | Study B: history regimes, including history restricted to non-entailed edits. | No history regime beats random deletion on non-entailed edits at a majority of cutoffs. |
| **H5 Architecture.** A graph neural ranker beats a tuned non-neural learner given the same inputs and training examples. | Study B: gradient-boosted ranker tuned only on the training ontologies. | The tuned non-neural ranker is within the GNN's ontology-level interval on both tasks. |
| **H6 Combination.** A combination with a selection rule fixed in advance beats both of its components on all edits, without losing on non-entailed edits. | Study C. | It does not beat the better component on all edits, or it loses on non-entailed edits. |
| **H7 Practical value.** For new classes, a curator reviewing a short list finds a correct parent often enough to save effort, with suggestions that rarely make the ontology incoherent. | Study D: correct suggestions in the top 5, 10 and 20; unsafe suggestions; runtime. | Recall@10 is not at least 10 points above name search, or more than 5% of top-10 suggestions are unsafe, at a majority of cutoffs. |

## Evaluation framework (shared by all studies)

- 41 public OBO ontologies with retrievable release histories (data/snapshots.csv). Cutoffs 1 July 2019, 2021 and 2023;
  for each ontology and cutoff the latest release dated on or before the cutoff. Everything a model sees is dated on or
  before its cutoff.
- Targets: is-a links absent at the cutoff, present in a later yearly snapshot (1 July 2020 … 2025) and still present in
  the current release (November 2026). Axes: edit type (new-class placement, moved parent, added parent), entailment by
  HermiT on the cutoff snapshot (re-parenting), namespace (own or imported; own is the main view), horizon (1, 2, 3+ years).
- Families: two ontologies are linked when they share at least 0.5% of their classes or is-a links; connected components
  at each cutoff give one connected block and a set of independent ontologies.
- Ranking over every class existing at the cutoff; filtered ranks; ties count half. Per-ontology metrics, averaged over
  ontologies; views with fewer than 10 queries in an ontology leave that ontology out. Paired per-ontology comparisons with
  ontology bootstrap intervals (4,000 resamples); cluster bootstrap where there are at least 8 overlap clusters.
- Nothing is tuned on test ontologies. Every result is reported, including negative ones.
