#!/usr/bin/env bash
# Install Slurm + munge on this node and put the repo's configs in /etc/slurm.
# Run from the repo root on each node:  sudo scripts/install-node.sh
# Idempotent. Services are left stopped and disabled at boot; use cluster-up.sh / cluster-down.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

UNITS="munge slurmd slurmctld"

if ! dpkg -s slurm-wlm munge >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y slurm-wlm munge
fi

# The packages start their daemons on install; we only run them per session.
systemctl disable --now $UNITS 2>/dev/null || true

install -d -m 755 /etc/slurm
install -m 644 slurm/slurm.conf slurm/gres.conf slurm/cgroup.conf /etc/slurm/
install -d -o slurm -g slurm -m 755 /var/spool/slurmd /var/spool/slurmctld /var/log/slurm

echo "installed on $(hostname): $(dpkg-query -W -f='${Version}' slurm-wlm), units disabled"
