#!/usr/bin/env bash
# Start the slurm-lab cluster. Run on the DGX (controller). Units are disabled at boot, so run this each session.
set -euo pipefail
AGX=aaron@192.168.1.202

sudo systemctl start munge
ssh "$AGX" 'sudo systemctl start munge' 2>&1 | grep -v openshell || true
munge -n | ssh "$AGX" unmunge 2>&1 | grep -E '^STATUS' || { echo "munge cross-node check failed"; exit 1; }

sudo systemctl start slurmctld slurmd
ssh "$AGX" 'sudo systemctl start slurmd' 2>&1 | grep -v openshell || true

# Nodes that were down when slurmctld last saw them stay DOWN; bring them back.
sleep 3
sudo scontrol update nodename=spark-79b7,orin state=resume 2>/dev/null || true
sinfo -N -l
