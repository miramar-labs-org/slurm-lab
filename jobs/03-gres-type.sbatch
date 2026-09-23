#!/bin/bash
# 03: pick a GPU *type*. The GRES type (gb10 / orin, see slurm/gres.conf) steers the job to a node
# without naming it. Submit two orin jobs at once: there's only one orin GPU, so the second
# pends with Reason=Resources until the first finishes.
#   sbatch --gres=gpu:gb10:1 jobs/03-gres-type.sbatch
#   sbatch --gres=gpu:orin:1 jobs/03-gres-type.sbatch; sbatch --gres=gpu:orin:1 jobs/03-gres-type.sbatch; squeue
#SBATCH --job-name=gres-type
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

PY=~/.pyenv/versions/pySlurm/bin/python
echo "$(hostname) got $SLURM_JOB_GPUS (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
$PY -W ignore -c 'import torch; print("gpu:", torch.cuda.get_device_name(0))'
sleep 30
