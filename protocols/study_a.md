# Study A protocol: how ontologies evolve

Tests H2 and describes how predictable each kind of edit is without learning.

## Census (code/study_a/census.py, census_report.py)

- For each cutoff C and ontology: rC = the latest release on or before C; later yearly snapshots (1 July of each following
  year up to 2025) and the current release (November 2026).
- Every is-a link absent from rC is classified at its first appearance: edit type (new-class placement, new class outside
  the task because all its parents are new, moved parent, added parent), own namespace or imported, entailed by rC
  (HermiT, re-parenting only), horizon, and whether the link (or, for new classes, the class) is still present in the
  current release.
- Reported per cutoff: counts by edit type; entailed share pooled, without GO and per ontology (median and range over
  ontologies with at least 10 edits of that kind); paired moved − added entailed share; share appearing within one year;
  share later removed; ontologies with test material; families.

## Query samples (code/study_a/queries.py)

- Own namespace, still present in the current release. Up to 150 queries per horizon bin (1 year, 2 years, 3+ years) and
  task, sampled once with a fixed seed (31) and shared by every ranker in every study.
- Re-parenting (task A): a link (x, y) with x and y in rC; kind moved or added; entailed by rC or not.
- New-class placement (task B): a class absent from rC whose first parents include a class of rC; added to the rC graph as
  an isolated node carrying its name. Every parent existing at the cutoff (at first appearance or in the current release)
  is a target and the best-ranked one counts.
- A test ontology for a task at a cutoff has at least 10 queries in that task.

## Training-free rankers (code/study_a/training_free.py)

Names only (TF-IDF character 2–4-grams + MiniLM sentence embeddings), name vote, structural vote, hybrid (standardised
name vote + standardised names-only score + 0.25 × standardised structural vote) and, for re-parenting, the reasoner ranker
(inferred ancestors of the query in rC first, nearest first, then the hybrid order). Filtered ranks over all classes of rC.

## Reported

MRR and Hits@10 per view (cutoff × edit type × entailment × horizon), mean over ontologies with at least 10 queries; paired
per-ontology change from the 1-year to the 3+-year horizon.
