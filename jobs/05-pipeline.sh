#!/bin/bash
# 05: two-stage pipeline with --dependency=afterok. Stage 2 only starts if stage 1 exits 0.
# Pass FAIL=1 to make stage 1 fail and watch stage 2 sit at Reason=DependencyNeverSatisfied.
#   bash jobs/05-pipeline.sh            (or FAIL=1 bash jobs/05-pipeline.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/pipeline

J1=$(sbatch --parsable --job-name=stage1 --output=logs/%x-%j.out --wrap \
    "echo prep on \$(hostname); seq 1 100 > runs/pipeline/data.txt; sleep 10; exit ${FAIL:-0}")
J2=$(sbatch --parsable --job-name=stage2 --output=logs/%x-%j.out --dependency=afterok:$J1 --kill-on-invalid-dep=no --wrap \
    "echo train on \$(hostname); awk '{s+=\$1} END {print \"sum =\", s}' runs/pipeline/data.txt")
echo "stage1=$J1 stage2=$J2 (afterok:$J1)"
squeue -j "$J1,$J2" -o "%.8i %.8j %.10T %.25r"
