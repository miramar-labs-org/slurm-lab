#!/bin/bash
# 01: what Slurm tells a job about itself. One task per node, on both nodes.
#   sbatch jobs/01-env.sbatch        (submit from the repo root; -o paths are relative to it)
#SBATCH --job-name=env
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x-%j.out

echo "== batch script runs once, on the first node: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST  SLURM_NNODES=$SLURM_NNODES  SLURM_NTASKS=$SLURM_NTASKS"
echo "expanded hostnames: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')"
echo "== srun launches one task per allocated task slot, on every node:"
# Single quotes: expand per task, not in the batch shell.
srun bash -c 'echo "  task $SLURM_PROCID/$SLURM_NTASKS on $(hostname): SLURM_NODEID=$SLURM_NODEID SLURM_LOCALID=$SLURM_LOCALID cpus=$(nproc)"'
