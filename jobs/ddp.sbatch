#!/bin/bash
# DDP across both nodes: one torchrun per node, one GPU each, gloo over WiFi.
#   sbatch jobs/ddp.sbatch                 (extra args go to ddp_train.py, e.g. sbatch jobs/ddp.sbatch --steps 100)
# Resumes from ckpt/ddp.pt if present. To practice requeue: scontrol requeue <jobid> mid-run; rm ckpt/ddp.pt to start over.
#SBATCH --job-name=ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

# Rendezvous store on the first node of the allocation (nodelists sort by name, so usually orin).
# Use its NodeAddr from slurm.conf: the DGX resolves its own hostname to 127.0.0.1.
export PYTHONWARNINGS=ignore  # numpy/GB10-arch/nested-tensor warnings, harmless
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_ADDR=$(scontrol show node "$MASTER" | grep -oP 'NodeAddr=\K\S+')
echo "== $(date -Is) job $SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} nodes=$SLURM_JOB_NODELIST master=$MASTER($MASTER_ADDR)"

# Per node: gloo must use the WiFi NIC (name differs per host), and torchrun must advertise the LAN IP.
# Without --local-addr, whichever node wins rank 0 advertises its hostname, and the DGX's resolves to 127.0.0.1.
srun bash -c '
  case $(hostname) in
    spark-79b7) export GLOO_SOCKET_IFNAME=wlP9s9   LOCAL_ADDR=192.168.1.200 ;;
    orin)       export GLOO_SOCKET_IFNAME=wlP1p1s0 LOCAL_ADDR=192.168.1.202 ;;
  esac
  exec ~/.pyenv/versions/pySlurm/bin/torchrun --nnodes=2 --nproc-per-node=1 --local-addr=$LOCAL_ADDR \
    --rdzv-backend=c10d --rdzv-id=$SLURM_JOB_ID --rdzv-endpoint='"$MASTER_ADDR"':29500 \
    train/ddp_train.py "$@"' _ "$@"
