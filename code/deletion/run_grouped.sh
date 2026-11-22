#!/bin/sh
# Random-deletion benchmark on grouped folds: three folds x three seeds, 3,000 training steps each; then the hybrid and the summary.
cd "$(dirname "$0")/../.."
for f in 0 1 2; do for s in 0 1 2; do
  [ -f results/deletion/grouped/grouped_gnn_ranks_f${f}_s${s}.csv ] || FOLDSET=grouped python3 code/deletion/gnn_deletion.py $f $s 3000
done; done
[ -f results/deletion/grouped/hybrid_ranks.csv ] || python3 code/deletion/hybrid_deletion.py
python3 code/deletion/summary_grouped.py
