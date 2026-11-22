# Temporal ontology completion across historical releases: data, code and results

Materials for the article *Learning and Reasoning for Temporal Ontology Completion: Cross-Ontology Transfer Across
Historical Releases*. The package holds the snapshot list, the derived data, the protocols, all code, the model outputs
(rank files and suggestion lists) and the analysis scripts that produce every number, table and figure of the article.

## Reproducing the reported numbers

```sh
pip install -r requirements.txt
./reproduce.sh            # about one minute; writes reports/
```

`reproduce.sh` needs only the files in this package: no raw ontology files, training or reasoning.

| Article | Report (reports/) | Script |
|---|---|---|
| Table 2, Section 6 census statements | `study_a_census.txt` | `code/study_a/census_report.py` |
| Table 3, horizon effects | `study_a_training_free.txt` | `code/study_a/aggregate_training_free.py` |
| Table 4, Table 5 (GNN rows), Sections 7.1–7.3 | `study_b.txt`, `study_b_comparisons.csv` | `code/analysis/analyze_b.py` |
| Table 5 (tuned-ranker rows), Table 6, runtime | `study_c.txt`, `study_c_comparisons.csv` | `code/analysis/analyze_c.py` |
| Tables 7–9, Figures 2 and 3, Section 9 | `study_d.txt`, `study_d_*.csv` | `code/analysis/analyze_d.py` |
| GNN size study (Section 7.1) | `study_e.txt`, `study_e_comparisons.csv` | `code/analysis/analyze_e.py` |
| Table 11 and its reconciliation with Table 4, encoder checks (Sections 7.1, 10 and 12) | `study_f.txt`, `study_f_comparisons.csv` | `code/analysis/analyze_f.py` |
| Overlap of test targets with training material, families per analysis, queries shared between cutoffs, candidate-list coverage, new-class target definitions (Sections 4.3, 4.4, 4.5, 5.2, 7.1) | `dependence_audit.txt`, `dependence_audit.csv` | `code/analysis/audit_dependence.py` |
| Agreement of the safety check with HermiT (Section 9.3) | `study_d_hermit_check.txt` | `code/analysis/hermit_agreement.py` |
| Table 10 | `deletion_grouped.txt`, `deletion_grouped.json` | `code/deletion/summary_grouped.py` |
| Cross-encoder comparison (Section 10) | `crossencoder_hard.txt`, `crossencoder_random.txt` | `code/deletion/crossencoder_summary.py` |
| Figures 1–3 | `figures/` | `code/analysis/figures.py` |

## Contents

```
data/
  snapshots.csv        every release used: ontology, snapshot (1 July 2019 ... 2025, or current = November 2026),
                       release date, data-version, source, repository, tag or commit, file path, URL, classes, SHA-256
  studyB.pkl.xz        cutoff graphs (ids, names, is-a and relation links), edit histories, query samples, folds, families
  hier.pkl.xz          HermiT hierarchies of the cutoff snapshots {cutoff: {ontology: {class: [direct inferred superclasses]}}}
  emb_1.npz … emb_3.npz  MiniLM name embeddings (float16) of every name in studyB.pkl.xz, in three parts; model
                       sentence-transformers/all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
  deletion/            folds and overlap clusters of the random-deletion comparison
protocols/             study design (hypotheses, rejection criteria) and the protocols of Studies A–F
code/                  learned rankers and job queues (top level), study_a/, study_d/, study_f/, analysis/, deletion/, prepare/
results/
  study_a/             census.csv, census_ontology.csv; ranks/ (training-free rankers, every query);
                       newclass_target_status.csv (per new-class query: targets, and how many are parents in the
                       current release; from code/study_a/newclass_target_status.py)
  study_b/ranks/       GNN ranks per regime and seed; tuned-ranker ranks (*__ranker.csv.gz)
  study_c/ranks/       family-held-out tuned ranker (h1_*) and reasoning/learning combinations (h6_*); timing.csv,
                       hermit_timing.csv
  study_d/             lists/ (top-20 suggestions per query and ranker), ranker_ranks/ (filtered ranks of the tuned ranker
                       on the std folds), measures/ (q_*: per query and ranker; s_*: top-10 suggestions with flags),
                       hermit_check/validation.csv (HermiT verdicts on the validation sample)
  study_e/             GNN size study (2023, std folds): val/ (size-selection ranks on held-out training ontologies),
                       ranks/ (test ranks, every size and seed; config = L<layers>d<width>)
  study_f/             deletion_queries.pkl.xz (hidden links on the cutoff snapshots of the std test ontologies),
                       deletion_ranks/ (GNN per seed, name search, hybrid), encoder_ranks.csv.gz (name search by name signal),
                       tfidf_ranks/ (tuned ranker and hybrid without the sentence encoder, 2019 and 2021)
  deletion/            grouped/ (GNN and hybrid ranks), crossencoder/ (ranks under the 100-candidate protocols)
reproduce.sh
```

## Names used in files and code

| In files | In the article |
|---|---|
| task A / task B | re-parenting / new-class placement |
| `names only` | name search |
| `hybrid` | hybrid heuristic |
| `reasoner: nearest inferred ancestor`, `R0` | reasoner ranker |
| `R` (GNN regime), `GNN_R` | GNN, deletion-trained |
| `M`, `E`, `N`, `X` | GNN trained with all history, existing-class history, new-class history, non-entailed existing-class history |
| `NAMES`, `STRUCT`, `NOREL` | GNN input ablations: names only, structure only, no relation edges |
| `RGPU` | regime R retrained on a GPU with the same seed (device check; excluded from all comparisons) |
| `GBR_del`, `GBR_delhist` | tuned ranker (8-setting grid), deletion-trained / deletion + history (Study B) |
| `GBR_del_c` | tuned ranker (27-setting grid), deletion-trained (Studies C and D) |
| `L0`, `L0d` | history-trained ranker / the same ranker trained on deletion only (Study C) |
| `S1`, `S2`, `S3`, `S4` | reasoner first, cycles removed, reasoner-aware ranker, calibrated fusion |
| `std`, `famA`, `famB`, `famBsize` | fold sets (Section 4.4) |
| `entailed` | target already entailed by the cutoff snapshot (HermiT) |
| `h` = `1y`, `2y`, `3y+` | horizon bin |

Rank files have one row per query, ranker (and seed): `cutoff, ontology, task, i` identify the query (index into the
sample in `studyB.pkl.xz`), `rank` is the filtered rank of the best target (ties count half).

## Running everything from the start

Raw releases are not redistributed; each ontology keeps its own licence. Steps (Python ≥ 3.10; Java for HermiT):

1. `python3 code/prepare/fetch_snapshots.py` downloads every release in `data/snapshots.csv` into `snapshots/` (or
   `SNAPSHOT_DIR`) and checks it against its SHA-256. Current releases are fetched through the PURL of their
   data-version; ten releases without a data-version line are fetched from the unversioned PURL and will no longer match.
2. `code/prepare/hermit_hierarchies.py`, `embed_names.py`, `build_data.py` rebuild `data/` (HermiT from `HERMIT_DIR` or
   from owlready2).
3. Study A: `code/study_a/census.py`; `code/study_a/training_free.py CUTOFF ontology[,...]`.
4. Study B: `python3 code/env_check.py`, then `python3 code/run_all.py --workers N` (195 GNN trainings, tuned rankers,
   evaluation).
5. Study C: `python3 code/run_c.py --workers N`; HermiT timing with `code/prepare/hermit_timing.py`.
6. Study D: `python3 code/baseline_lists.py`, `python3 code/run_d.py --workers N`, then
   `code/study_d/measures.py CUTOFF ontology` per test set; HermiT validation with `code/study_d/hermit_check.py`
   (and `hermit_check_k.py` for CL 2023).
7. Study E (GNN size): `python3 code/run_e.py --workers N` (80 trainings; uses the GPU when available).
8. Paired deletion and encoder checks: `code/study_f/deletion_queries.py`, `score_deletion.py CUTOFF FOLD` for every
   cutoff and std fold (uses the Study B models of regime R), `encoder_check.py`, `ranker_tfidf.py CUTOFF FOLD` for the
   2019 and 2021 std folds.
9. Random deletion on November 2026 releases: `code/deletion/parse_current.py`, `embed_current.py`, `run_grouped.sh`; cross-encoders with
   `gnn_deletion.py 0 SEED 3000` for seeds 0–2, then `crossencoder_train.py` and `crossencoder_eval.py PROTO ontologies`.

Training uses fixed seeds. On other hardware or library versions retrained models can differ slightly; the device check in
`reports/study_b.txt` compares a GPU retraining with the CPU run against the spread between seeds.

## Licence

Code: MIT licence (LICENSE). Data and results: CC BY 4.0. The ontologies remain under their own licences.

## Citation

Please cite the accompanying article: *Learning and Reasoning for Temporal Ontology Completion: Cross-Ontology Transfer
Across Historical Releases*.
