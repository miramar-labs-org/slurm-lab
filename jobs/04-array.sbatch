#!/bin/bash
# 04: job array as a toy sweep. It runs 8 tasks, at most 2 at a time (%2). Each task gets its own
# $SLURM_ARRAY_TASK_ID and lands on whichever GPU is free.
#   sbatch jobs/04-array.sbatch; squeue    (pending tasks show as JOBID_[2-7%2])
#SBATCH --job-name=sweep
#SBATCH --array=0-7%2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%A_%a.out

LRS=(0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
PY=~/.pyenv/versions/pySlurm/bin/python
echo "array job $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID on $(hostname): lr=$LR"
# Toy "training": fit y=3x with SGD and report the final loss for this lr.
$PY -W ignore - "$LR" <<'PY'
import sys, torch
lr = float(sys.argv[1]); torch.manual_seed(0)
x = torch.randn(1024, 1, device="cuda"); y = 3 * x
w = torch.zeros(1, device="cuda", requires_grad=True)
for _ in range(50):
    loss = ((x * w - y) ** 2).mean(); loss.backward()
    with torch.no_grad(): w -= lr * w.grad; w.grad.zero_()
print(f"lr={lr} w={w.item():.4f} loss={loss.item():.6f}")
PY
