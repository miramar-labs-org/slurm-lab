#!/usr/bin/env bash
# Stop the slurm-lab cluster on both nodes (frees the boxes for other projects).
set -uo pipefail
AGX=aaron@192.168.1.202

ssh "$AGX" 'sudo systemctl stop slurmd munge' 2>&1 | grep -v openshell
sudo systemctl stop slurmd slurmctld munge
echo "stopped. DGX: $(systemctl is-active slurmctld slurmd munge | tr '\n' ' ')"
echo "        AGX: $(ssh "$AGX" 'systemctl is-active slurmd munge' 2>&1 | grep -v openshell | tr '\n' ' ')"
