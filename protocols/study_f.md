# Study F protocol: paired deletion check and encoder-date check

Fixed before either check was run. Two analyses that test claims of the article directly. Neither trains a new model.

## F1. Random deletion against real future edits on the same ontologies

The earlier comparison (Section 10) set deletion results on November 2026 releases against future-edit results on cutoff
snapshots: different ontologies, snapshots and queries. F1 pairs them.

- Cutoffs 2019, 2021, 2023; std folds; each fold's test ontologies; the cutoff snapshot of each (data/studyB.pkl.xz).
- Deletion queries: in each cutoff snapshot, 10% of is-a links hidden (rng seeded with the CRC-32 of "cutoff/ontology"),
  up to 300 hidden links sampled; the child is the query, every class of the snapshot a candidate, its other parents
  removed (filtered rank, ties half). The queries are fixed in one file before any model is scored.
- A deletion query whose hidden link is an is-a link in a training ontology of its fold (shared imported content) is
  flagged, as the future-edit queries are; the primary comparison excludes flagged queries in both settings, and the
  comparison with them included is reported beside it.
- Models: the Study B GNNs of regime R (deletion-trained; std folds; seeds 0–2), unchanged, scored on the snapshot with the
  hidden links removed; name search and the hybrid heuristic computed on the same view (Study A definitions).
- Future-edit results: the existing Study A and Study B ranks on the same ontologies.
- Comparisons per cutoff, per ontology (ontologies with >= 10 queries in both settings), ontology bootstrap 95% intervals:
  1. GNN MRR under deletion minus GNN MRR on future non-entailed re-parenting, on all re-parenting, and on new classes.
  2. The GNN's gain over the hybrid under deletion minus its gain on future non-entailed re-parenting and on new classes.
- Reading: the claim that deletion overstates the usefulness of learned rankers is supported for a view if both 1 and 2 are
  positive with intervals above zero at a majority of cutoffs; if only 1 holds, the article says deletion inflates the
  level but not the learned advantage; otherwise the claim is withdrawn.

## F2. Does the post-2019 sentence encoder help more at the earlier cutoffs?

The sentence encoder (all-MiniLM-L6-v2) was trained in 2021, after the 2019 cutoff. If its training data carried knowledge
of later ontology content, its contribution should be larger at 2019 (and 2021) than at 2023, which it predates.

- Shared Study A queries at all three cutoffs; training-free name search with three name signals: character n-gram TF-IDF
  only, MiniLM only, and their sum (the name search reported in the paper; its ranks must equal the Study A ranks).
- Measure per cutoff and view (non-entailed re-parenting, new classes): the per-ontology gain of the sum over TF-IDF only,
  with ontology bootstrap intervals; and the difference of that gain between 2019 and 2023 (and 2021 and 2023), with
  intervals from resampling ontologies within each cutoff.
- Reading: if the 2019 − 2023 difference in gain has an interval that includes zero or lies below zero, the data show no
  sign that the encoder's date favours the earlier cutoffs, and the article says so; if it lies above zero, the
  article reports it and the 2019 results are qualified accordingly.

## Not covered

F1 uses one deletion rate and one deletion protocol; the tuned ranker is not scored under deletion. F2 tests one way an
encoder trained later could help; it cannot exclude every form of leakage.

## F3. The learned-ranker advantage without the sentence encoder (added after F1 and F2, fixed before it was run)

F2 compares the encoder's contribution to name search between cutoffs. F3 removes the encoder from the whole pipeline at
the two cutoffs it does not predate, so that every component is dated on or before the cutoff.

- Cutoffs 2019 and 2021; std folds; the shared Study A queries.
- TF-IDF-only pipeline: the sentence-encoder similarity is set to zero everywhere. Name search is character n-gram TF-IDF;
  the name vote uses the ten classes most similar by TF-IDF; the hybrid keeps its weights (name vote + name search +
  0.25 x structural vote). The tuned ranker is the Study B ranker (deletion signal, 8-setting grid, same folds, tuning split
  and seeds), retrained and tuned with the encoder features at zero, reranking the TF-IDF-only hybrid's top 300.
- Comparison per cutoff and view (non-entailed re-parenting, new classes): tuned ranker minus hybrid, both TF-IDF only,
  per ontology (>= 10 queries), ontology bootstrap 95% intervals; the same comparison with the encoder (Study B) beside it.
- Reading: where the interval lies above zero, the learned ranker's advantage over the best heuristic at that cutoff does
  not depend on the encoder, and the article says so; where it does not, the article reports it and restricts the claim of
  a strictly chronological result to the 2023 cutoff.
- Not covered: the GNN is not retrained without the encoder.
