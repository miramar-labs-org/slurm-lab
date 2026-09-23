# slurm-lab — Plan

**Goal:** learn Slurm in a day by joining the DGX Spark and the AGX Orin into a
two-node Slurm GPU cluster, then running a real multi-node PyTorch DDP job on it.

**Non-goals:** performance. The cluster runs over WiFi and the two GPUs are very
different sizes, so the job *will* be slower than the DGX alone. That is fine: the
point is the Slurm mechanics, not throughput.

**Operating rule:** this project only runs when nothing else is running on either box.
Slurm daemons are installed but **not enabled at boot**. Start them for a session, stop
them after (`scripts/cluster-up.sh` / `cluster-down.sh`).

---

## Verified facts (2026-09-23)

| | DGX Spark | AGX Orin |
|---|---|---|
| Hostname (`hostname -s`) | `spark-79b7` | `orin` |
| LAN IP / iface | `192.168.1.200` / `wlP9s9` (WiFi) | `192.168.1.202` / `wlP1p1s0` (WiFi) |
| OS / arch | Ubuntu 24.04.4, aarch64 | Ubuntu 24.04.4 (L4T R39.2.1 / JetPack 7.2.1), aarch64 |
| GPU / compute cap | GB10 / sm_121 | Orin (nvgpu) / sm_87 |
| Driver / CUDA toolkit | 580.173.02 / 13.0 | 595.78 / 13.2 |
| `slurm-wlm` apt candidate | 23.11.4-1.2ubuntu5+esm1 | 23.11.4-1.2ubuntu5 |
| `aaron` uid:gid | 1000:1000 | 1000:1000 (renumbered from 2002 on 2026-09-23) |
| `~/shared` | local dir, exported over SMB | CIFS mount of `//spark-79b7.local/shared` (automount, fstab) |

- Wired ports are down on both boxes, so everything uses WiFi. Ping RTT between them is 5–88 ms. Accepted.
- Both Slurm packages are the same upstream 23.11.4. The DGX one is an Ubuntu Pro ESM rebuild. Slurm requires matching versions across nodes, and these match.
- CUDA toolkit versions differ, and that's fine. The PyTorch wheel bundles its own CUDA runtime + NCCL. **Decision: don't upgrade the DGX driver/toolkit.**

## Python env: `pySlurm` (done)

It's a pyenv virtualenv at the **same path on both nodes**, which Slurm needs because the job
command runs on every node: `~/.pyenv/versions/pySlurm/bin/python`.

- Python 3.12.2, `torch 2.14.0+cu130`, NCCL 2.30.7 (identical on both nodes)
- GPU matmul smoke test passes on both
- The Orin warns `_warn_unsupported_code`: the wheel ships sm_80/90/100/110/120, not sm_87.
  It runs anyway because sm_80 binaries are compatible with sm_87. Harmless.
- pyenv was installed fresh on the AGX for this (git clone + pyenv-virtualenv, init lines
  appended to `~/.zshrc`, matching the DGX).
- The scaffold's `.venv` (DGX only) is for the notebook. **Slurm jobs use `pySlurm`**, never `.venv`.

---

## Phase 0: prerequisites

### 0.1 Normalize the AGX `aaron` uid/gid 2002 → 1000 ✅ done 2026-09-23
Slurm runs each job step as the *numeric* uid that submitted it, on every node. uid/gid
1000 are free on the AGX. A filesystem scan found nothing owned by 2002 outside the home dir
(`/mnt/nvme/home/aaron`, bind-mounted at `/home/aaron`, ~28.7k files).

The uid can't be changed while `aaron` has processes (jupyterlab user service, GNOME session,
your own ssh). So run it as a **detached root unit** and reconnect afterwards:

1. Write `scripts/agx-renumber-uid.sh` (runs as root on the AGX):
   - `sleep 5`; `loginctl terminate-user aaron`; `pkill -KILL -u aaron`
   - stop `home-aaron-shared.automount`/`.mount` and `mnt-nvme-home-aaron-shared.*` (keeps the
     recursive chown off CIFS)
   - `groupmod -g 1000 aaron && usermod -u 1000 -g 1000 aaron`
   - `find /mnt/nvme/home/aaron -xdev -uid 2002 -exec chown -h 1000 {} +` and the same for `-gid 2002 → chgrp`
   - `sed -i 's/uid=2002,gid=2002/uid=1000,gid=1000/' /etc/fstab`; `systemctl daemon-reload`; restart the automount
   - log everything to `/root/uid-renumber.log`
2. Launch it: `scp` the script over, then `ssh agx 'sudo systemd-run --unit=uid-renumber bash /root/agx-renumber-uid.sh'`.
3. Wait ~30 s, then `ssh aaron@192.168.1.202 id` should show `uid=1000`. Also check `ls -ln ~ | head`,
   `ls ~/shared`, `systemctl --user status jupyterlab`, `cat /root/uid-renumber.log`.

**Result:** ran in ~11 s, 0 leftover 2002-owned files, fstab/automount, jupyterlab, ollama, and pySlurm torch all OK.
Needed one extra step: the AGX has `Linger=yes`, so the script turns linger off before the kill
(otherwise `user@2002` respawns and `usermod` fails), then back on at the end.
4. Fallback if ssh breaks: the AGX has a local console (monitor + keyboard), with passwordless sudo.

### 0.2 Name resolution ✅ done 2026-09-23
`orin.local` mDNS is unreliable (platform rule: reach the AGX by IP). Add to `/etc/hosts` on **both**:
```
192.168.1.200 spark-79b7
192.168.1.202 orin
```

### 0.3 Firewall / ports ✅ checked 2026-09-23
**Result:** no LAN filtering on either node. DGX `ufw` is inactive. The AGX has no ufw; its only
DROP rule is Tailscale's `ts-input` (drops 100.64/10 not on tailscale0).

**Name-resolution gotchas found:**
- The DGX `/etc/hosts` line 2 maps `127.0.0.1 spark-79b7`. It's pre-existing and left alone (k3s and the
  platform may depend on it), so on the DGX its own hostname resolves to loopback. Mitigation: always set
  `NodeAddr`/`SlurmctldHost` IPs explicitly in slurm.conf, and pass `--rdzv-endpoint=192.168.1.200` by IP.
- On the AGX, `getent hosts orin` returns IPv6 addresses (nss-myhostname). `ahostsv4` returns the right
  IPv4, which is another reason to use explicit IPs.
- Backups of the original files: `/etc/hosts.bak-slurm-lab` on both nodes.

Check `ufw status` on both. Slurm needs 6817 (slurmctld), 6818 (slurmd), and the `srun` port
range (`SrunPortRange=60001-60100` in slurm.conf). The PyTorch rendezvous needs 29500, plus the ports NCCL picks at random.

---

## Phase 1: test the network path first (no Slurm yet)

Prove the two GPUs can do an NCCL all-reduce over WiFi *before* adding Slurm on top.

- `train/nccl_test.py`: `init_process_group("nccl")`, all-reduce a tensor, print the result, time 20 iterations.
- Launch manually with `torchrun` on each node:
  ```
  # DGX (rank 0)
  NCCL_SOCKET_IFNAME=wl NCCL_IB_DISABLE=1 NCCL_DEBUG=INFO \
    ~/.pyenv/versions/pySlurm/bin/torchrun --nnodes=2 --node-rank=0 --nproc-per-node=1 \
    --master-addr=192.168.1.200 --master-port=29500 train/nccl_test.py
  # AGX: same, with --node-rank=1
  ```
  `NCCL_SOCKET_IFNAME=wl` prefix-matches both WiFi interfaces (`wlP9s9` and `wlP1p1s0`) and keeps
  NCCL off `tailscale0`/`docker0`/`cni0`.
- **If NCCL fails across the two GPU architectures:** fall back to `backend="gloo"`
  (all-reduce goes through the CPU). `GLOO_SOCKET_IFNAME` needs the exact interface name per node, so set it by hostname.
  Record which backend works. Phase 4 uses it.

## Phase 2: install Slurm

The controller (`slurmctld`) runs on the DGX. The compute daemon (`slurmd`) runs on both. No accounting DB to start.

1. On both: `sudo apt install slurm-wlm munge` (installs slurmd + slurmctld + client tools), then **`systemctl disable`** every Slurm/munge unit (start them per session only).
2. munge: generate the key on the DGX (`/etc/munge/munge.key`), copy it to the AGX (same content,
   `munge:munge`, mode 0400). Test: `munge -n | ssh aaron@192.168.1.202 unmunge`.
3. Keep the configs in the repo under `slurm/` and `install -m644` them to `/etc/slurm/` on both nodes (identical files):
   - `slurm.conf`: `ClusterName=miramar`, `SlurmctldHost=spark-79b7`,
     `NodeName=spark-79b7 CPUs=… RealMemory=… Gres=gpu:gb10:1`,
     `NodeName=orin CPUs=… RealMemory=… Gres=gpu:orin:1` (get the real values from `slurmd -C` on each node),
     `GresTypes=gpu`, `SelectType=select/cons_tres`, `ProctrackType=proctrack/cgroup`,
     `TaskPlugin=task/cgroup,task/affinity`, `SrunPortRange=60001-60100`,
     partitions: `dgx` (spark-79b7), `agx` (orin), `all` (both, Default=YES).
     Set `RealMemory` well below the physical total. Both are unified-memory boxes.
   - `gres.conf`: set `AutoDetect=off` and declare the GPUs by hand
     (`NodeName=spark-79b7 Name=gpu Type=gb10 File=/dev/nvidia0`,
     `NodeName=orin Name=gpu Type=orin File=<device node>`). Check which `/dev/nvidia*` device
     the Orin actually exposes on JP7.2.1. If there isn't a clean one, drop `File=` (count-only GRES).
   - `cgroup.conf`: start with `ConstrainDevices=no` (simplest). Try `yes` as an exercise later.
4. Start: DGX `munge slurmctld slurmd`, AGX `munge slurmd`. Check that `sinfo` shows both nodes `idle`
   and that `srun -N2 hostname` prints both hostnames.
5. Write `scripts/cluster-up.sh` / `cluster-down.sh` (start/stop the services on both nodes over ssh) and `scripts/install-node.sh` (apt, disable units, install configs).

## Phase 3: Slurm fundamentals (the actual learning)

One small `jobs/*.sbatch` per exercise, output to `~/shared/slurm-lab/logs/%x-%j.out`
(the same path on both nodes):

- [ ] `sinfo`, `sinfo -N -l`, `scontrol show node`, `scontrol show partition`
- [ ] `srun` interactive: `srun -p agx --gres=gpu:1 --pty bash`, `nvidia-smi`/`tegrastats` inside
- [ ] `sbatch` a single-GPU job, then watch it with `squeue`/`scontrol show job`, cancel it with `scancel`
- [ ] GRES: request `--gres=gpu:gb10:1` vs `--gres=gpu:orin:1`, and see a job pending on resources
- [ ] Job arrays (`--array=0-7%2`) doing a toy sweep, `$SLURM_ARRAY_TASK_ID`
- [ ] Dependencies (`--dependency=afterok:<jobid>`) as a two-stage pipeline
- [ ] Node states: `scontrol update nodename=orin state=drain reason=test`, then `resume`
- [ ] Environment: `SLURM_JOB_NODELIST`, `scontrol show hostnames`, `SLURM_PROCID`, `SLURM_LOCALID`
- [ ] Stretch: accounting (`slurmdbd` + MariaDB on the DGX; it doesn't support Postgres), then `sacct`/`sreport`

## Phase 4: distributed training under Slurm

- `train/ddp_train.py`: a small model on a public dataset. **Public data only (PHI rule):**
  e.g. ResNet-18 on CIFAR-10, or a char-level GPT on tiny-shakespeare. Pre-download the dataset to
  `~/shared/slurm-lab/data` so each node doesn't fetch it itself.
- `jobs/ddp.sbatch`: `-N2 --ntasks-per-node=1 --gres=gpu:1 -p all`. Set the master to the first node
  in `$SLURM_JOB_NODELIST` (use its LAN IP), then `srun ~/.pyenv/versions/pySlurm/bin/torchrun --nnodes=2
  --nproc-per-node=1 --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29500 train/ddp_train.py`.
  Use the backend that passed Phase 1.
- Log per-rank step time. Expect the Orin to hold back the DGX. Optional exercise: give the
  Orin a smaller per-rank batch and compare.
- Save checkpoints to `~/shared/slurm-lab/ckpt`, then practice `scancel` followed by resume-from-checkpoint (`--requeue`).

## Phase 5: wrap-up

- `cluster-down.sh`; confirm the Slurm units are disabled at boot on both nodes.
- `NOTES.md`: a Slurm cheat-sheet built from what was actually run, plus gotchas hit.
- Blog draft (the Create Project workflow already opened a blog PR).

---

## Repo layout (target)

```
PLAN.md              this file
NOTES.md             cheat-sheet + gotchas (Phase 5)
slurm/               slurm.conf, gres.conf, cgroup.conf (installed to /etc/slurm on both)
scripts/             agx-renumber-uid.sh, install-node.sh, cluster-up.sh, cluster-down.sh
train/               nccl_test.py, ddp_train.py
jobs/                *.sbatch exercises
```

## Risks

| Risk | Mitigation |
|---|---|
| uid renumber locks out ssh | detached root unit, logs in /root; local console fallback |
| NCCL fails between sm_121 ↔ sm_87 | gloo backend fallback (Phase 1 decides) |
| CIFS `~/shared` quirks (0600 perms, no locking) | fine for single-user logs/ckpts; keep the dataset read-only |
| Orin GRES device node unclear under nvgpu | count-only GRES (no `File=`) |
| WiFi drops mid-job | small job, frequent checkpoints; that's what `--requeue` is for |
| Slurm doesn't see k3s/Ollama GPU use | the operating rule above: only run when nothing else is running |
