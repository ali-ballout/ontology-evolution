# Study D protocol: what a curator would get

Fixed before any Study D result was seen. Short lists of suggested parents at a historical cutoff, judged by what the
ontology's editors later did. Tests H7 and answers two pre-declared questions. No new models; the learned models are those
already tested.

## Suggestion lists (top 20 per query)

- Queries: the shared Study A queries at all three cutoffs (re-parenting and new classes).
- Lists from: names only, hybrid, reasoner ranker R0 (code/baseline_lists.py); the tuned ranker GBR_del_c (deletion-trained,
  Study C's procedure and grid, on the std folds at all three cutoffs); the history-trained ranker L0 (Study C models;
  re-parenting at 2021 and 2023); the GNN (Study B regime R, seed 0).
- Exclusions: the query itself, its parents at the cutoff, and classes absent at the cutoff never appear in a list.

## Measures

- M1, recall@k (k = 5, 10, 20): a list hits if it contains any class that is a parent of the query in the November 2026
  release, was not its parent at the cutoff, and existed at the cutoff.
- M2, review effort: suggestions read up to and including the first correct one; 21 if none is in the top 20.
- M3, logical safety: share of top-10 suggestions whose addition would create a cycle (the suggestion is an inferred
  descendant of the query) or make the query or one of its inferred descendants unsatisfiable (their inferred ancestors
  after the addition contain a declared disjoint pair); computed from the cutoff snapshot's HermiT hierarchy and validated
  against HermiT on a sample.
- M4, anatomy of wrong top-10 suggestions, first match wins: (a) redundant, already an inferred ancestor at the cutoff;
  (b) near miss, parent, child or sibling of a correct class in the November 2026 release; (c) transient, a parent of the
  query in an intermediate yearly snapshot, later removed; (d) unsupported.
- Breakdowns: by edit type, ontology size (< 3,000; 3,000–10,000; > 10,000 classes) and family.

## H7

For the tuned ranker, new classes, at each cutoff: (1) recall@10 exceeds names only by at least 0.10 (ontology-mean
difference, 95% interval excluding zero); (2) at most 5% of its top-10 suggestions fail M3. Supported if both hold at a
majority of cutoffs. Criterion 2 is weak on its own, because about half of the ontologies have no disjointness axioms.

## Pre-declared questions

- Q1, side-by-side list for re-parenting (2021, 2023): the reasoner's top 5 and L0's top 5, interleaved, duplicates skipped,
  filled to 10 from the two lists in turn; against L0's and the reasoner's top 10 on recall@10, for all, entailed and
  non-entailed edits.
- Q2: M4 shares for the tuned ranker, names only and the reasoner at each cutoff.

## Statistics

Per-ontology values; ontology bootstrap; cluster bootstrap where there are at least 8 clusters at 20% overlap; block and
independent ontologies separately. All results reported.

## Deviations and additions

The validation of M3 against HermiT was planned on every flagged suggestion; with 10,009 distinct disjointness flags and
5,885 cycle flags this was not feasible, and HermiT checked a random sample of 150 flagged and 150 unflagged re-parenting
suggestions from the cutoff-ontology pairs with disjointness axioms (results/study_d/hermit_check/validation.csv).

The two kinds of unsafe suggestion under M3, subclass cycles and unsatisfiable classes, are also reported separately. This breakdown was added after the protocol was fixed; M3 and the H7 criterion are computed as defined above.
