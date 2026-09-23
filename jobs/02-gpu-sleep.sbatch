#!/bin/bash
# 02: a single-GPU job to practise watching and cancelling.
#   J=$(sbatch --parsable jobs/02-gpu-sleep.sbatch); squeue; scontrol show job $J; scancel $J
#SBATCH --job-name=gpu-sleep
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

PY=~/.pyenv/versions/pySlurm/bin/python
echo "$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
$PY -W ignore -c 'import torch; print("gpu:", torch.cuda.get_device_name(0))'
echo "sleeping; cancel me with scancel $SLURM_JOB_ID"
sleep 600
