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

## Phase 1: test the network path first (no Slurm yet) ✅ done 2026-09-23: **gloo**

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

**Result (2026-09-23): NCCL fails on the Orin, gloo works. Phase 4 uses `backend="gloo"`.**
- **NCCL root cause (verified 2026-09-23):** the JP7.2.1 NVML (`nvidia-l4t-nvml` 39.2.1, driver 595.78,
  `Orin (nvgpu)`) returns `NVML_ERROR_NOT_SUPPORTED` for `nvmlDeviceGetP2PStatus`, even for (0,0).
  NCCL's `commAlloc` → `ncclNvmlDeviceGetHandleByPciBusId` → `ncclNvmlEnsureInitialized` checks the
  P2P status of every device pair and returns `ncclSystemError` on any failure. It reproduces with 1 rank on the AGX
  alone, so it isn't a network problem. The DGX NVML (580) returns Success.
  - `NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` / `NCCL_NVLS_ENABLE` can't help: the check runs before transports are chosen.
  - Older or newer NCCL won't help. 2.27.7–2.30.7 fail on any error. 2.32.3 only tolerates it for GPUs that aren't CUDA-visible.
  - NCCL hard-requires NVML (a missing lib is also fatal). Nothing else here needs it: gloo, torch CUDA,
    and Slurm with `AutoDetect=off` don't.
- gloo: correctness OK. A 16 MiB fp32 all-reduce takes **~2.5 s/iter (busbw ~6.3 MiB/s)** over WiFi.
  Run with `GLOO_SOCKET_IFNAME=wlP9s9` (DGX) / `wlP1p1s0` (AGX) and `train/nccl_test.py --backend gloo`.
- **Decision: gloo-only.** Stretch goal if NCCL is ever wanted: a ~20-line `libnvidia-ml.so.1` shim on the AGX
  (it links to the real lib and overrides only `nvmlDeviceGetP2PStatus` to return Success with status
  NOT_SUPPORTED), put first on `LD_LIBRARY_PATH` for the job. Expect little speedup, because WiFi TCP is the bottleneck either way.
- **The repo lives on the shared filesystem (2026-09-23):** the only working copy is `~/shared/slurm-lab`
  (a local dir on the DGX, the CIFS mount of `//spark-79b7.local/shared` on the AGX). The same path works on both nodes,
  and `~/git-miramar-labs-org/projects/slurm-lab` is a symlink to it on both. `logs/`, `data/`, `ckpt/`
  and `runs/` sit inside it (gitignored). CIFS forces mode 0600 on the AGX, so invoke scripts via
  `bash`/`python` (no exec bit). `sbatch file` is fine. Verified: a 2-node sbatch job ran from here and both nodes wrote to `logs/`.

## Phase 2: install Slurm ✅ done 2026-09-23

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

**Result (2026-09-23):** `sinfo` shows both nodes idle, `srun -N2 hostname` works, and
`srun -N2 --gres=gpu:1` runs a torch matmul on GB10 + Orin (both uid 1000, `CUDA_VISIBLE_DEVICES=0`).
Configs are in `slurm/`. Use `sudo scripts/install-node.sh` (per node), then `scripts/cluster-up.sh` / `cluster-down.sh` (on the DGX).
Two things differed from the plan:
- **Orin GRES needs a `File=`.** Slurm 23.11 logs `Ignoring file-less GPU` and the node goes INVAL
  (`gres/gpu count reported lower than configured (0 < 1)`). `/dev/nvidia0`/`nvidia1` exist, but CUDA
  actually opens `/dev/nvgpu/igpu0/ctrl` + `/dev/nvmap` (checked via `/proc/self/fd`), so `File=/dev/nvgpu/igpu0/ctrl`.
- **`LaunchParameters=disable_send_gids` is required.** By default slurmctld resolves supplementary groups on
  the DGX and sends numeric gids. DGX aaron isn't in video/render, and gid numbers differ across hosts, so orin jobs
  failed with `NvRmMemInitNvmap failed: Permission denied` → `No CUDA GPUs are available`.
- `RealMemory` is set to 32000 (DGX) and 16000 (AGX) out of 124609/62878, to leave room for k3s/Ollama.
- The munge key was generated by the package on the DGX and copied to the AGX (md5 match).

## Phase 3: Slurm fundamentals (the actual learning) ✅ exercises run 2026-09-23

One small `jobs/*.sbatch` per exercise, output to `~/shared/slurm-lab/logs/%x-%j.out`
(the same path on both nodes):

- [x] `sinfo`, `sinfo -N -l`, `scontrol show node`, `scontrol show partition`
- [ ] `srun` interactive: `srun -p agx --gres=gpu:1 --pty bash`, `nvidia-smi`/`tegrastats` inside
- [x] `sbatch` a single-GPU job, then watch it with `squeue`/`scontrol show job`, cancel it with `scancel`
- [x] GRES: request `--gres=gpu:gb10:1` vs `--gres=gpu:orin:1`, and see a job pending on resources
- [x] Job arrays (`--array=0-7%2`) doing a toy sweep, `$SLURM_ARRAY_TASK_ID`
- [x] Dependencies (`--dependency=afterok:<jobid>`) as a two-stage pipeline
- [x] Node states: `scontrol update nodename=orin state=drain reason=test`, then `resume`
- [x] Environment: `SLURM_JOB_NODELIST`, `scontrol show hostnames`, `SLURM_PROCID`, `SLURM_LOCALID`
- [x] Stretch: accounting (`slurmdbd` + MariaDB on the DGX; it doesn't support Postgres), then `sacct`/`sreport`

**Result (2026-09-23):** exercises are in `jobs/01-05` (submit from the repo root). All ran on both nodes.
The interactive `srun --pty` item is left for hands-on use (it needs a real terminal). Observations:
- **Node order is alphabetical:** `orin,spark-79b7`, so the batch script, rank 0 and `SLURM_NODEID=0` land on the **orin**.
  Phase 4 must take the master address from `scontrol show hostnames | head -1`, never hard-code the DGX.
- `task/cgroup` pins each task to its allocated CPUs (`nproc`=1 for a default 1-CPU task).
- **InvalidAccount without slurmdbd:** every job first shows `PENDING InvalidAccount` and only starts when the
  backfill scheduler runs (default every 30 s). Cause: 23.11's main scheduler calls `assoc_mgr_validate_assoc_id`,
  which calls `assoc_mgr_refresh_lists`, and that fails because there's no accounting plugin to return a QOS list
  (`_refresh_assoc_mgr_qos_list: no new list given back`). Backfill skips that check. Still unfixed in 23.11.11.
  `PriorityType=priority/basic` doesn't help (tested). Fix: `SchedulerParameters=bf_interval=2`, so jobs start in ~1–2 s.
  The proper fix is slurmdbd (stretch). Side effect: pending jobs show Reason `None` rather than `Resources`.
  **Superseded 2026-09-23:** with slurmdbd running, jobs start in 0.4–3 s at the default bf_interval, so the line is now commented out.

**Accounting result (2026-09-23):** host-native MariaDB 10.11 + slurmdbd on the DGX (`scripts/install-dbd.sh`,
`slurm/slurmdbd.conf`, `slurm/mariadb-slurm.cnf`). MariaDB listens on 127.0.0.1 only, and the `slurm` DB user uses unix_socket auth, so there's no password.
Associations: cluster `miramar` → account `lab` → user `aaron`. `AccountingStorageEnforce=associations` rejects unknown accounts and users.
Not on k3s: slurmctld depends on it, and it must come up before k3s does. See guide §3.11. What `sacct` exposed:
- No `DefMemPerCPU` → every job booked the node's entire memory, so jobs on a node ran serially. Set `DefMemPerCPU=1024`.
- GPUs were missing from AllocTRES until `AccountingStorageTRES=gres/gpu`.
- `jobacct_gather/cgroup` reported MaxRSS=0 and CPU=0 (cause not isolated). `jobacct_gather/linux` gives real numbers (MaxRSS ~2.1 GB/rank for ddp).
- slurmdbd logs an `innodb_buffer_pool_size` warning (hard-coded 4 GB advice); 256M is kept deliberately on unified memory.
- Array `04`: 8 tasks, 2 at a time, split 4/4 across nodes. lr 0.1–0.3 converge, ≤0.01 underfit in 50 steps.
- `05` with `FAIL=1`: stage2 goes to `DependencyNeverSatisfied` (kept because of `--kill-on-invalid-dep=no`).
- Drain: a job pinned to `-p agx` pends with "Nodes required for job are DOWN, DRAINED…" and runs right after `resume`.

## Phase 4: distributed training under Slurm ✅ done 2026-09-23

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

**Results (2026-09-23):**
- `train/ddp_train.py`: a char-level GPT (0.83M params, 4 layers, dim 128) on tiny-shakespeare (`data/tinyshakespeare.txt`, public),
  gloo, rank 0 saving atomically to `ckpt/ddp.pt` every 25 steps, auto-resume. It's tiny on purpose, since each step all-reduces ~3.3 MB of grads over WiFi.
- **Hang #1: rendezvous ok, then the workers hang in init.** Ranks come from c10d rendezvous join order, not the nodelist, so the DGX can be rank 0.
  The rank-0 agent advertises its hostname for the worker store, `spark-79b7` → 127.0.0.1, and the workers got
  `MASTER_ADDR=localhost`. Fix: `torchrun --local-addr=<LAN IP>` per host (in `jobs/ddp.sbatch`).
- 2-node step ≈ **450 ms**. Solo (`--standalone`, one GPU) ≈ **17 ms DGX, ~45 ms Orin**. So ~90% of each step is the
  gloo gradient all-reduce over WiFi (~8 MiB/s). The Orin's slower compute is noise. The smaller-Orin-batch exercise is moot.
- 300 steps: loss 2.9 (step 10) → 2.08, val 2.11, and the sample is Shakespeare-shaped gibberish.
- Requeue: `scontrol requeue <id>` after the step-75 checkpoint → the step is CANCELLED "DUE TO JOB REQUEUE", then the job pends with
  **Reason=BeginTime, ~2 min** (Slurm's built-in requeue delay), restarts with `SLURM_RESTART_COUNT=1`, prints "resumed at step 75" on both ranks, and
  finishes. `--open-mode=append` keeps both runs in one log. (`scancel` removes the job for good. Requeue is `scontrol requeue`.)

## Phase 5: wrap-up ✅ done 2026-09-23

- `cluster-down.sh`; confirm the Slurm units are disabled at boot on both nodes.
- `NOTES.md`: a Slurm cheat-sheet built from what was actually run, plus gotchas hit.
- Blog draft (the Create Project workflow already opened a blog PR).

**Results:** the cheat-sheet and gotchas live in `docs/learning-slurm.md` §5–6, not a separate NOTES.md. The blog draft is filled in on
miramar-labs-org.github.io PR #80 (unmerged: merging publishes it). The cluster was stopped with `cluster-down.sh`, and all units are disabled at boot.

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
