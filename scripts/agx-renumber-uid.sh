#!/usr/bin/env bash
# Renumber the AGX `aaron` user/group 2002 -> 1000 so Slurm's numeric uid matches the DGX.
# Runs as root, detached from any aaron session (it kills them all):
#   scp scripts/agx-renumber-uid.sh aaron@192.168.1.202:/tmp/ && \
#   ssh aaron@192.168.1.202 'sudo install -m 700 /tmp/agx-renumber-uid.sh /root/ && \
#     sudo systemd-run --unit=uid-renumber bash /root/agx-renumber-uid.sh'
# Then reconnect after ~30 s and check /root/uid-renumber.log.
set -euo pipefail

USER_NAME=aaron
OLD=2002
NEW=1000
HOME_REAL=/mnt/nvme/home/aaron   # bind-mounted at /home/aaron

exec >>/root/uid-renumber.log 2>&1
echo "=== $(date -Is) start: $USER_NAME $OLD -> $NEW"

if [[ "$(id -u "$USER_NAME")" == "$NEW" ]]; then
    echo "already uid $NEW, nothing to do"; exit 0
fi
if getent passwd "$NEW" >/dev/null || getent group "$NEW" >/dev/null; then
    echo "uid/gid $NEW already taken, aborting"; exit 1
fi

sleep 5   # let the launching ssh return

# Linger would restart user@$OLD as soon as we kill it, so turn it off first.
loginctl disable-linger "$USER_NAME"
loginctl terminate-user "$USER_NAME" || true
systemctl stop "user@$OLD.service" || true
for _ in 1 2 3 4 5; do
    pkill -KILL -u "$OLD" || true
    sleep 1
    pgrep -u "$OLD" >/dev/null || break
done
if pgrep -u "$OLD" -a; then
    echo "processes still running as $OLD, aborting"; exit 1
fi

# Unmount CIFS so the recursive chown stays off the share.
systemctl stop home-aaron-shared.automount home-aaron-shared.mount mnt-nvme-home-aaron-shared.mount || true
umount -l /home/aaron/shared "$HOME_REAL/shared" 2>/dev/null || true

groupmod -g "$NEW" "$USER_NAME"
usermod -u "$NEW" -g "$NEW" "$USER_NAME"   # also chowns the home dir tree it owns
echo "passwd: $(getent passwd "$USER_NAME")"

# usermod only fixes files under the home it knows about; sweep the real path + tmp explicitly.
for d in "$HOME_REAL" /tmp /var/tmp; do
    find "$d" -xdev -uid "$OLD" -exec chown -h "$NEW" {} +
    find "$d" -xdev -gid "$OLD" -exec chgrp -h "$NEW" {} +
done
echo "leftover $OLD-owned files in home: $(find "$HOME_REAL" -xdev \( -uid "$OLD" -o -gid "$OLD" \) | wc -l)"

sed -i "s/uid=$OLD,gid=$OLD/uid=$NEW,gid=$NEW/" /etc/fstab
grep cifs /etc/fstab
systemctl daemon-reload
systemctl start home-aaron-shared.automount

loginctl enable-linger "$USER_NAME"   # brings user@$NEW (jupyterlab, runner) back up
echo "=== $(date -Is) done"
