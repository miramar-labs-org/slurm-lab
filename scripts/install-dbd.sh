#!/usr/bin/env bash
# Accounting on the DGX: MariaDB + slurmdbd, host-native. Run from the repo root on the DGX:  sudo bash scripts/install-dbd.sh
# Idempotent. Stop the cluster first (cluster-down.sh). Units are left stopped and disabled at boot; cluster-up.sh / cluster-down.sh start and stop them.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! dpkg -s mariadb-server slurmdbd >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y mariadb-server slurmdbd
fi

install -m 644 slurm/mariadb-slurm.cnf /etc/mysql/mariadb.conf.d/99-slurm.cnf
install -o slurm -g slurm -m 600 slurm/slurmdbd.conf /etc/slurm/slurmdbd.conf

# Create the DB and a password-less socket-auth user for the slurm OS user.
systemctl start mariadb
mysql <<'SQL'
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED VIA unix_socket;
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
SQL

# Register the cluster and an account for the lab user. slurm.conf has AccountingStorageEnforce=associations,
# so a user without an association can't submit. "Nothing new added" on a rerun is fine.
systemctl start munge slurmdbd
sleep 2
sacctmgr -i add cluster miramar || true
sacctmgr -i add account lab Description=slurm-lab Organization=miramar || true
sacctmgr -i add user "${SUDO_USER:-aaron}" Account=lab DefaultAccount=lab || true
sacctmgr -n show assoc format=cluster,account,user

systemctl disable --now slurmdbd mariadb 2>/dev/null || true
echo "installed: mariadb $(dpkg-query -W -f='${Version}' mariadb-server), slurmdbd $(dpkg-query -W -f='${Version}' slurmdbd), units disabled"
