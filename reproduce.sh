#!/bin/sh
# Recomputes every reported number from the materials in this package (no raw snapshots, training or reasoning needed).
# Reports and figures are written to reports/.
set -e
cd "$(dirname "$0")"; mkdir -p reports
python3 code/study_a/census_report.py                     > /dev/null
python3 code/study_a/aggregate_training_free.py           > reports/study_a_training_free.txt
python3 code/analysis/analyze_b.py                         > reports/study_b.txt
python3 code/analysis/analyze_c.py                         > reports/study_c.txt
python3 code/analysis/analyze_d.py                         > reports/study_d.txt
python3 code/analysis/hermit_agreement.py                  > reports/study_d_hermit_check.txt
python3 code/analysis/analyze_e.py                         > reports/study_e.txt
python3 code/analysis/analyze_f.py                         > reports/study_f.txt
python3 code/analysis/audit_dependence.py                  > reports/dependence_audit.txt
python3 code/deletion/summary_grouped.py                   > reports/deletion_grouped.txt
python3 code/deletion/crossencoder_summary.py hard         > reports/crossencoder_hard.txt
python3 code/deletion/crossencoder_summary.py random       > reports/crossencoder_random.txt
python3 code/analysis/figures.py                           > /dev/null
echo "reports written to reports/"
