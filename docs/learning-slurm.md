# Learning Slurm on a Two-Node Home Cluster

A hands-on guide to Slurm, the job scheduler that runs most of the world's supercomputers. It's built
around a real (if unusual) cluster: an **NVIDIA DGX Spark** and an **NVIDIA Jetson AGX Orin** on home
WiFi. Everything here was actually run, and every error shown is one we actually hit.

> **Who this is for:** someone comfortable with Linux, ssh, and PyTorch who has never administered or
> used Slurm. By the end you'll have built a cluster from bare hosts, submitted every common kind of
> job, and debugged the failure modes that tutorials usually skip.

**Contents**

1. [Slurm in five minutes](#1-slurm-in-five-minutes)
2. [The lab](#2-the-lab)
3. [Building the cluster](#3-building-the-cluster)
4. [Exercises: using the cluster](#4-exercises-using-the-cluster)
5. [Troubleshooting playbook](#5-troubleshooting-playbook)
6. [Cheat sheet](#6-cheat-sheet)
7. [What's next](#7-whats-next)
8. [Glossary](#8-glossary)

---

## 1. Slurm in five minutes

Slurm answers one question: *who gets which machines, when?* You describe what a job needs
(nodes, CPUs, memory, GPUs, time), Slurm queues it, and when the resources are free it starts your
program on them and cleans up afterwards.

### The moving parts

| Piece | Runs on | Job |
|---|---|---|
| **`slurmctld`** | the controller (one node) | The brain. Holds the queue, decides what runs where, and tracks node state. |
| **`slurmd`** | every compute node | The hands. Receives work from `slurmctld`, launches processes, and reports back. |
| **`munge`** | every node | Authentication. Every Slurm message is signed with a shared secret key, so nodes can trust each other's claims about who a user is. |
| `slurmdbd` | optional, one node | Accounting database (job history, fair-share, quotas). **We don't run it**, and section 5 shows what that costs. |
| `slurm.conf` | identical on every node | The single description of the whole cluster: nodes, partitions, plugins. |

### The vocabulary

- **Node:** one machine (`spark-79b7`, `orin`).
- **Partition:** a named group of nodes, like a queue. Ours are `all` (default), `dgx` and `agx`.
- **GRES (Generic RESource):** anything countable that isn't CPU or memory. Here that means GPUs, declared as
  `gpu:<type>:<count>`, e.g. `gpu:orin:1`.
- **Job:** a resource allocation plus the thing you run in it. It has a job ID.
- **Job step:** each `srun` inside a job. A job can run many steps, one after another or in parallel.
- **Task:** one process in a step. `srun -N2 --ntasks-per-node=1` means two tasks, one per node.
  Each gets a rank in `$SLURM_PROCID`.

### The three ways to run something

| Command | Blocks your shell? | Use it for |
|---|---|---|
| `srun <cmd>` | yes | Run a command on allocated nodes right now, or a step inside a batch job. |
| `sbatch script.sh` | no, returns a job ID | The normal way: queue a script and collect its output file later. |
| `salloc` | yes, gives you a shell | Grab an allocation, then run several `srun`s interactively inside it. |

A batch script is just a shell script with `#SBATCH` comment lines that act as default flags:

```bash
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --nodes=2
#SBATCH --gres=gpu:1          # per node
#SBATCH --output=logs/%x-%j.out   # %x = job name, %j = job id
srun hostname                 # runs once per task, on every node
```

> **Key idea:** the batch script itself runs **once**, on the first allocated node. Only commands
> launched with `srun` fan out to every node. Forgetting this is the most common beginner bug.

---

## 2. The lab

### Hardware

| | DGX Spark | AGX Orin |
|---|---|---|
| Hostname | `spark-79b7` | `orin` |
| Role | controller **and** compute | compute |
| LAN IP / interface | `192.168.1.200` / `wlP9s9` (WiFi) | `192.168.1.202` / `wlP1p1s0` (WiFi) |
| OS | Ubuntu 24.04, aarch64 | Ubuntu 24.04 (JetPack 7.2.1), aarch64 |
| GPU / compute capability | GB10 / sm_121 | Orin iGPU (`nvgpu` driver) / sm_87 |
| CPUs / RAM | 20 / 124.6 GB unified | 12 / 62.9 GB unified |
| Slurm | 23.11.4 | 23.11.4 |

Two very different GPUs, joined over WiFi, is a *terrible* training cluster, and that's the point.
Every assumption a real cluster hides (identical nodes, a fast interconnect, a shared filesystem,
a directory service) we had to provide by hand. That's where the learning is.

### Architecture

```
                 ┌──────────────────────── DGX Spark (spark-79b7, .200) ─────────────────────┐
 you ── sbatch ─►│ slurmctld  (queue + scheduler, :6817)                                     │
                 │ slurmd     (compute, :6818)      GPU: gb10  File=/dev/nvidia0             │
                 │ munge      (auth)                ~/shared/slurm-lab  ← local dir, SMB export│
                 └──────────────┬──────────────────────────────────┬─────────────────────────┘
                                │ Slurm RPC 6817/6818, srun 60001-60100 │ SMB (CIFS)
                                │ gloo all-reduce (TCP)             │
                 ┌──────────────▼──────────────────────────────────▼─────────────────────────┐
                 │ slurmd  (compute, :6818)     GPU: orin  File=/dev/nvgpu/igpu0/ctrl         │
                 │ munge   (auth, same key)     ~/shared/slurm-lab  ← CIFS mount, same path   │
                 └──────────────────────── AGX Orin (orin, .202) ────────────────────────────┘
```

### Repository layout

```
slurm-lab/                 lives at ~/shared/slurm-lab (same path on both nodes)
├── PLAN.md                phase plan + verified facts + decisions (the lab notebook)
├── docs/learning-slurm.md this guide
├── slurm/                 slurm.conf, gres.conf, cgroup.conf → /etc/slurm on both nodes
├── scripts/
│   ├── agx-renumber-uid.sh   one-off: make the Orin's uid match the DGX
│   ├── install-node.sh       apt install + disable units + install configs (per node)
│   ├── cluster-up.sh         start munge/slurmctld/slurmd on both (run on DGX)
│   └── cluster-down.sh       stop everything
├── train/nccl_test.py     cross-node all-reduce test (nccl | gloo)
├── jobs/                  exercises 01–05
└── logs/ data/ ckpt/ runs/   job artifacts (gitignored)
```

---

## 3. Building the cluster

Slurm itself installs in one `apt` line. Almost all the real work is making the two machines
**look the same** to it. A job is a command line that gets executed *verbatim on every node*, so
anything that differs between nodes (user IDs, paths, group numbers, device names) is a bug waiting to
happen.

### 3.1 Same user, same numeric ID

**Why:** Slurm ships jobs around by **numeric uid**, not username. Job 42 submitted by uid 1000 on the
DGX runs as uid 1000 on the Orin. Originally `aaron` was 1000 on the DGX but **2002** on the Orin, so
jobs there would have run as a nonexistent user.

**Fix:** `scripts/agx-renumber-uid.sh` renumbers the user and group, re-owns the home directory, and
fixes the CIFS mount options in `/etc/fstab`. You can't change a user's uid while it has running
processes, including the ssh session you'd run the command from. So the script runs as a
**detached root systemd unit** that kills every session of that user, does the work, and logs to `/root`:

```bash
scp scripts/agx-renumber-uid.sh aaron@192.168.1.202:/tmp/
ssh aaron@192.168.1.202 'sudo install -m 700 /tmp/agx-renumber-uid.sh /root/ &&
                         sudo systemd-run --unit=uid-renumber bash /root/agx-renumber-uid.sh'
# reconnect ~30 s later
ssh aaron@192.168.1.202 id        # uid=1000(aaron) gid=1000(aaron) ...
```

> **Lesson: linger.** The Orin had `loginctl` *linger* enabled for `aaron`, meaning systemd keeps a
> user manager running even with nobody logged in. Kill it and systemd immediately restarts it, and
> `usermod` fails with "user is currently used by process". The script disables linger first and
> re-enables it at the end.

Real clusters avoid all this with a directory service (LDAP/FreeIPA/SSSD) so uids are consistent everywhere.

### 3.2 Name resolution

Each node must be able to reach every other node by the name in `slurm.conf`. We added to `/etc/hosts`
on both:

```
192.168.1.200 spark-79b7
192.168.1.202 orin
```

> **Lesson: hostnames lie; use IPs.** The DGX's `/etc/hosts` already maps its own name to `127.0.0.1`
> (common on Ubuntu, and k3s may rely on it), and the Orin's resolver answers its own name with IPv6
> addresses first. So `slurm.conf` pins every address explicitly:
> `SlurmctldHost=spark-79b7(192.168.1.200)` and `NodeName=... NodeAddr=192.168.1.x`.

Firewalls: DGX `ufw` is inactive, and the Orin only drops Tailscale-range traffic. Slurm needs 6817
(controller), 6818 (slurmd), and the `SrunPortRange` (60001–60100) open between nodes.

### 3.3 Same software, same path

Every node runs the *same command line*, so the interpreter must live at the same path everywhere.
We use a pyenv virtualenv, `~/.pyenv/versions/pySlurm` (Python 3.12.2, torch 2.14.0+cu130), on
both nodes. The torch wheel bundles its own CUDA runtime and NCCL, so the different system CUDA
toolkits (13.0 vs 13.2) don't matter.

### 3.4 A shared filesystem

Jobs need to see the same code and write output where you can find it. Our DGX already exported
`~/shared` over SMB, and the Orin mounts it at the same path. So the repo's **only working copy is
`~/shared/slurm-lab`**: edit on the DGX, and the Orin sees the change instantly. Logs from both nodes land
in one `logs/` directory. (The old project path is a symlink to it on both nodes.)

> **Lesson:** our first attempt rsync'ed the repo to the Orin before every run. It worked, but it's
> exactly the kind of drift a shared filesystem exists to prevent. Production clusters use NFS,
> Lustre, GPFS or similar for `/home` and scratch.

The CIFS mount forces file mode 0600, so there's no exec bit. Run scripts as `bash x.sh`/`python x.py`.
`sbatch x.sbatch` is fine because sbatch reads the file itself.

### 3.5 Test the network *before* Slurm

Before adding a scheduler, prove the two GPUs can actually talk. `train/nccl_test.py` all-reduces a
tensor across both nodes with plain `torchrun`:

```bash
# on the DGX (rank 0); run the same on the Orin with --node-rank=1
GLOO_SOCKET_IFNAME=wlP9s9 ~/.pyenv/versions/pySlurm/bin/torchrun --nnodes=2 --node-rank=0 \
  --nproc-per-node=1 --master-addr=192.168.1.200 --master-port=29500 \
  train/nccl_test.py --backend gloo
```

```
[rank 0/2] host=spark-79b7 gpu=NVIDIA GB10 backend=gloo
[rank 0] correctness OK: [3.0, 3.0, 3.0, 3.0]
all_reduce 16 MiB x 20: 2534.9 ms/iter, busbw 6.3 MiB/s
```

**gloo works. NCCL doesn't**, and why is a good lesson in reading error messages.

#### Case study: why NCCL fails on the Orin

NCCL (NVIDIA's GPU collective library) and gloo (Meta's CPU/TCP one) are both PyTorch
`torch.distributed` backends. NCCL is normally far faster because it knows the GPU topology (NVLink,
PCIe, InfiniBand). To learn that topology it queries **NVML**, the GPU management library behind `nvidia-smi`.

The failure:

```
NCCL WARN nvmlDeviceGetP2PStatus(0,0,NVML_P2P_CAPS_INDEX_READ) failed: Not Supported
ncclSystemError: System call ... or external library call failed or device error.
```

The debugging path:

1. **Reproduce minimally.** The same error happens with **one rank on the Orin alone**, so it's not the
   network, and not the GB10↔Orin mismatch.
2. **Trace it.** `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL` shows the error return path:
   `ncclNvmlEnsureInitialized` → `ncclNvmlDeviceGetHandleByPciBusId` → `commAlloc` → failure while
   creating the communicator.
3. **Isolate the layer.** A 20-line ctypes probe calls NVML directly, with no NCCL involved:
   - DGX: `nvmlDeviceGetP2PStatus(gpu0, gpu0, READ)` → `Success`
   - Orin: the same call → `3 = NVML_ERROR_NOT_SUPPORTED`. Jetson's NVML (`nvidia-l4t-nvml`) implements
     only part of the API.
4. **Read the source.** In NCCL's `src/misc/nvmlwrap.cc`, NVML init checks the P2P status of *every device
   pair, including a GPU with itself*, and returns `ncclSystemError` on any failure. That's true in
   2.27, 2.30 and the latest 2.32. NVML itself is mandatory (NCCL fails the same way without it).

So `NCCL_P2P_DISABLE=1` and friends can't help. They take effect when NCCL picks transports, which is
*after* this check. **Decision: use gloo.** Over WiFi the link is the bottleneck anyway. (Stretch fix: a tiny
`libnvidia-ml.so.1` shim on the Orin that answers that one call with "not supported" instead of an error.)

### 3.6 Install Slurm

`scripts/install-node.sh` (run with sudo on each node) installs `slurm-wlm` and `munge`, **disables
the units at boot** (this cluster only runs when nothing else is using the boxes), and copies the
repo's configs into `/etc/slurm/`.

Get each node's real hardware description from `slurmd -C`:

```
NodeName=spark-79b7 CPUs=20 Boards=1 SocketsPerBoard=1 CoresPerSocket=20 ThreadsPerCore=1 RealMemory=124609
NodeName=orin       CPUs=12 Boards=1 SocketsPerBoard=1 CoresPerSocket=12 ThreadsPerCore=1 RealMemory=62878
```

### 3.7 `slurm.conf`, annotated

```ini
ClusterName=miramar
SlurmctldHost=spark-79b7(192.168.1.200)   # controller, with an explicit IP (see 3.2)
AuthType=auth/munge

SchedulerType=sched/backfill
SchedulerParameters=bf_interval=2          # see section 5, "InvalidAccount"
SelectType=select/cons_tres                # schedule individual cores/memory/GPUs, not whole nodes
SelectTypeParameters=CR_Core_Memory
ProctrackType=proctrack/cgroup             # track job processes with cgroups (clean kill on scancel)
TaskPlugin=task/cgroup,task/affinity       # pin tasks to their allocated CPUs
ReturnToService=2                          # a node that went DOWN comes back when slurmd re-registers
GresTypes=gpu
LaunchParameters=disable_send_gids         # see 3.9

AccountingStorageType=accounting_storage/none   # no slurmdbd (yet)

# RealMemory is well below physical: both boxes are unified-memory and share RAM with other services
NodeName=spark-79b7 NodeAddr=192.168.1.200 CPUs=20 ... RealMemory=32000 Gres=gpu:gb10:1
NodeName=orin       NodeAddr=192.168.1.202 CPUs=12 ... RealMemory=16000 Gres=gpu:orin:1

PartitionName=all Nodes=spark-79b7,orin Default=YES
PartitionName=dgx Nodes=spark-79b7
PartitionName=agx Nodes=orin
```

`slurm.conf` must be **identical on every node**. Slurm warns (and misbehaves) if the copies differ.

### 3.8 munge: the shared secret

Every node needs the *same* `/etc/munge/munge.key` (owner `munge`, mode 0400). The package generated
one on the DGX. We copied it to the Orin (for example like this) and confirmed the md5 matched:

```bash
sudo cat /etc/munge/munge.key | ssh aaron@192.168.1.202 \
  'sudo install -o munge -g munge -m 400 /dev/stdin /etc/munge/munge.key'
munge -n | ssh aaron@192.168.1.202 unmunge      # STATUS: Success (0)
```

A credential made on one node decodes on the other, so the nodes trust each other.

### 3.9 GPUs as GRES, and two surprises

`gres.conf` tells each `slurmd` which device files make up its GPUs. We turn off auto-detection
(it relies on NVML, which is only partial on the Orin) and declare them by hand:

```ini
AutoDetect=off
NodeName=spark-79b7 Name=gpu Type=gb10 File=/dev/nvidia0
NodeName=orin       Name=gpu Type=orin File=/dev/nvgpu/igpu0/ctrl
```

**Surprise 1: the Orin's GPU isn't `/dev/nvidia0`.** The Orin has `/dev/nvidia0` *and* `/dev/nvidia1`,
and `nvidia-smi` reports the minor number as N/A. Our first try declared the GPU with a count only
and no `File=`, and the node went `INVAL`:

```
orin  inval  gres/gpu count reported lower than configured (0 < 1)
slurmd.log: warning: Ignoring file-less GPU gpu:orin from final GRES list
```

Slurm 23.11 won't accept a GPU without a device file. To find the real one, start CUDA and list its
open file descriptors:

```python
import torch, os; torch.zeros(1, device="cuda")
print({os.readlink(f"/proc/self/fd/{f}") for f in os.listdir("/proc/self/fd")} )
# → /dev/nvgpu/igpu0/ctrl, /dev/nvmap, /dev/dri/renderD128
```

The Jetson `nvgpu` driver uses `/dev/nvgpu/igpu0/ctrl`, so that's the GRES file.

**Surprise 2: CUDA works over ssh but not under Slurm.** `srun` on the Orin failed with:

```
NvRmMemInitNvmap failed: error Permission denied
RuntimeError: No CUDA GPUs are available
```

`/dev/nvmap` is `root:video 0440`, so you need the `video` group. Inside the job, `id` showed:

```
groups=1000(aaron),4(adm),27(sudo),...,122(gdm),983(weston-launch),988(polkitd)      ← no video, no render
```

Those are the **DGX's** group memberships, as numbers, applied on the Orin. By default, 23.11's
`slurmctld` looks up the user's supplementary groups on the *controller* and sends the numeric gids with
the job. The DGX `aaron` isn't in `video`, and gid numbers mean different groups on each machine (988 is
`docker` on the DGX but `polkitd` on the Orin).

**Fix:** `LaunchParameters=disable_send_gids` makes each `slurmd` look up groups locally. After that,
the job on the Orin shows `44(video)` and `993(render)`, and CUDA works.

> **Lesson:** this is the uid problem again, one level down. Anything identified by *number* must mean
> the same thing on every node, or be resolved locally.

### 3.10 Start, verify, stop

```bash
scripts/cluster-up.sh      # munge on both → munge cross-check → slurmctld+slurmd → resume nodes → sinfo
scripts/cluster-down.sh    # stop everything on both nodes (frees the boxes)
```

Done when:

```
$ sinfo -N -o "%N %P %T %G"
orin       all*  idle  gpu:orin:1
spark-79b7 all*  idle  gpu:gb10:1
$ srun -N2 --gres=gpu:1 python -c "import torch,socket; print(socket.gethostname(), torch.cuda.get_device_name(0))"
orin Orin
spark-79b7 NVIDIA GB10
```

---

## 4. Exercises: using the cluster

Run everything from the repo root (`cd ~/shared/slurm-lab`). Output paths in the job files are relative to it.

### 4.1 Looking around

```bash
sinfo                      # partitions and node states
sinfo -N -l                # one line per node, with CPUs/memory/reason
scontrol show node orin    # everything Slurm knows about a node (Gres, State, Reason, ...)
scontrol show partition all
squeue                     # the queue (add -u $USER, --start, -o formats)
```

Node states you'll see: `idle`, `mix` (partly used), `alloc` (full), `drain` (no new jobs), `down`,
`inval` (config doesn't match reality). A trailing `*` (`idle*`) means the node isn't responding.

### 4.2 Exercise 01: what a job knows about itself (`jobs/01-env.sbatch`)

```bash
sbatch jobs/01-env.sbatch
```

```
== batch script runs once, on the first node: orin
SLURM_JOB_ID=8  SLURM_JOB_NODELIST=orin,spark-79b7  SLURM_NNODES=2  SLURM_NTASKS=2
expanded hostnames: orin spark-79b7
== srun launches one task per allocated task slot, on every node:
  task 0/2 on orin: SLURM_NODEID=0 SLURM_LOCALID=0 cpus=1
  task 1/2 on spark-79b7: SLURM_NODEID=1 SLURM_LOCALID=0 cpus=1
```

What to notice:
- The batch part ran **once**, on `orin`. The `srun` part ran once per task.
- Node lists are **sorted by name**, so `orin` is node 0 and hosts rank 0. For distributed training,
  derive the master address from `scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1`, never hard-code it.
- `cpus=1`: `task/cgroup` confines each task to the CPUs it asked for (1 by default). Use `--cpus-per-task`.
- Useful variables: `SLURM_PROCID` (global rank), `SLURM_LOCALID` (rank on this node),
  `SLURM_NODEID`, `SLURM_NTASKS`, `SLURM_JOB_NODELIST` (compressed; expand it with `scontrol show hostnames`).

### 4.3 Exercise 02: watch it, then kill it (`jobs/02-gpu-sleep.sbatch`)

```bash
J=$(sbatch --parsable jobs/02-gpu-sleep.sbatch)   # --parsable prints just the ID
squeue
#  JOBID       NAME    STATE   TIME    NODELIST TRES_PER_NODE
#     17  gpu-sleep  RUNNING   0:05        orin gres/gpu:1
scontrol show job $J        # full detail: JobState, NodeList, TresPerNode, RunTime, TimeLimit, ...
scancel $J
cat logs/gpu-sleep-$J.out
# ... slurmstepd-orin: error: *** JOB 17 ON orin CANCELLED AT 2026-09-23T09:37:52 ***
```

`--time=00:10:00` in the script is the job's wall-clock limit. Slurm kills it when time runs out. Shorter
limits also help the backfill scheduler slot your job in sooner.

### 4.4 Exercise 03: choosing a GPU type (`jobs/03-gres-type.sbatch`)

`--gres=gpu:1` means "any GPU". `--gres=gpu:orin:1` means "an Orin GPU", which steers the job to a node
without naming the node:

```bash
sbatch --gres=gpu:gb10:1 jobs/03-gres-type.sbatch
sbatch --gres=gpu:orin:1 jobs/03-gres-type.sbatch
sbatch --gres=gpu:orin:1 jobs/03-gres-type.sbatch    # only one Orin GPU exists…
squeue -o "%.4i %.10j %.8T %.12r %.11N %b"
#   20  gres-type  PENDING   None               gres/gpu:orin:1    ← waits for job 19
#   18  gres-type  RUNNING   None   spark-79b7  gres/gpu:gb10:1
#   19  gres-type  RUNNING   None         orin  gres/gpu:orin:1
```

Inside each job, `CUDA_VISIBLE_DEVICES=0` is set for you. Slurm renumbers the allocated GPUs from 0.
(The reason column says `None` rather than `Resources` because of the scheduler quirk in section 5.)

Other ways to target nodes: `-p agx` (partition), `-w orin` (exact node), `-x orin` (exclude a node).

### 4.5 Exercise 04: job arrays (`jobs/04-array.sbatch`)

Arrays are how you run a parameter sweep: one submission becomes many near-identical tasks.

```bash
#SBATCH --array=0-7%2        # tasks 0..7, at most 2 running at once
#SBATCH --output=logs/%x-%A_%a.out   # %A = array job id, %a = task index
LRS=(0.3 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
```

```
$ squeue
     JOBID   NAME    STATE    NODELIST
21_[2-7%2]  sweep  PENDING                 ← the waiting tasks, folded into one line
      21_0  sweep  RUNNING        orin
      21_1  sweep  RUNNING  spark-79b7
```

Results (a toy SGD fit of `y = 3x`, 50 steps):

```
lr=0.3 w=3.0000 loss=0.000000     lr=0.003  w=0.7393 loss=4.861138
lr=0.1 w=2.9999 loss=0.000000     lr=0.001  w=0.2695 loss=7.037955
lr=0.03 w=2.8356 loss=0.028550    lr=0.0003 w=0.0835 loss=8.008505
lr=0.01 w=1.8391 loss=1.316529    lr=0.0001 w=0.0281 loss=8.309365
```

The eight tasks split 4/4 across the two nodes, since each went wherever a GPU was free.
`scancel 21_5` cancels a single task, and `scancel 21` cancels the whole array.

### 4.6 Exercise 05: pipelines with dependencies (`jobs/05-pipeline.sh`)

```bash
J1=$(sbatch --parsable ... --wrap "prepare data")
J2=$(sbatch --parsable --dependency=afterok:$J1 ... --wrap "train on it")
```

`--wrap` turns a one-line command into a batch job without writing a script.

```bash
bash jobs/05-pipeline.sh           # stage2 shows Reason=Dependency, then runs: "sum = 5050"
FAIL=1 bash jobs/05-pipeline.sh    # stage1 exits 1 → stage2: Reason=DependencyNeverSatisfied
```

Dependency types: `afterok` (succeeded), `afternotok` (failed), `afterany` (finished either way),
`after` (started), `singleton` (one job of this name at a time). By default a job whose
dependency can never be met is left pending for you to inspect. `--kill-on-invalid-dep=yes` cancels it instead.

### 4.7 Exercise 06: draining a node (maintenance)

```bash
sudo scontrol update nodename=orin state=drain reason="test"
sinfo -N -o "%N %T %E"               # orin  drained  test
sbatch -p agx --wrap hostname        # PENDING: "Nodes required for job are DOWN, DRAINED or reserved…"
sudo scontrol update nodename=orin state=resume   # the pending job runs right away
```

`drain` lets running jobs finish but takes no new ones. It's how you take a node out for maintenance
without killing anybody's work. `down` is the hard version.

### 4.8 Try it yourself: interactive jobs

```bash
srun -p agx --gres=gpu:1 --pty bash     # a shell on the Orin, holding its GPU
tegrastats                              # Jetson's nvidia-smi equivalent
exit                                    # releases the allocation

salloc -N2 --gres=gpu:1                 # hold both nodes…
srun hostname                           # …and launch steps into the allocation
exit
```

---

## 5. Troubleshooting playbook

The first place to look is always the logs: `/var/log/slurm/slurmctld.log` on the DGX and
`/var/log/slurm/slurmd.log` on the node in question. `scontrol show node <n>` shows the `Reason=`.

| Symptom | Cause | Fix |
|---|---|---|
| Every job sits `PENDING (InvalidAccount)` for up to 30 s on an idle cluster | No slurmdbd. See below. | `SchedulerParameters=bf_interval=2`, or run slurmdbd |
| Node `inval`, "gres/gpu count reported lower than configured (0 < 1)" | GPU declared without `File=`, and the slurmd log says "Ignoring file-less GPU" | Declare the real device file (3.9) |
| `NvRmMemInitNvmap failed: Permission denied` / "No CUDA GPUs are available" only under Slurm | Job got the controller's numeric groups, missing `video`/`render` | `LaunchParameters=disable_send_gids` (3.9) |
| Node shows `idle*` / `down*` | slurmd not running or unreachable | Start slurmd; check ports 6817/6818; `ReturnToService=2` brings it back |
| Node stays `down` after a restart | It was marked down while away | `scontrol update nodename=X state=resume` (`cluster-up.sh` does this) |
| Reason text is stale after fixing | Reason strings stick until the next state change | Harmless. Cycle the node or the cluster. |
| `usermod: user is currently used by process` | Processes still running as that user, e.g. a linger-respawned user manager | `loginctl disable-linger` first (3.1) |
| `pkill -f pattern` over ssh kills your own ssh | The pattern matches the ssh command line itself | Kill by PID, or use a pattern that can't match itself |
| NCCL `nvmlDeviceGetP2PStatus ... Not Supported` on Jetson | Jetson NVML is partial, and NCCL treats the failure as fatal | Use gloo (3.5) |

### Deep dive: `InvalidAccount` without an accounting database

**Symptom.** On an idle cluster, jobs wait:

```
09:36:08 PENDING InvalidAccount
...
09:36:18 RUNNING  orin              ← ~10–30 s later
slurmctld.log: sched: JobId=10 has invalid account
               error: _refresh_assoc_mgr_qos_list: no new list given back keeping cached one.
```

**What's happening.** Slurm has two schedulers:

- The **main scheduler** runs on every submit. It walks the queue in priority order and starts what fits.
- The **backfill scheduler** runs every `bf_interval` seconds (default 30). It looks ahead and
  slips smaller jobs into gaps.

In 23.11, before starting a job, the main scheduler calls `assoc_mgr_validate_assoc_id()`: *is this
user's account valid?* With no association data loaded, it tries to fetch it
(`assoc_mgr_refresh_lists`). That asks the accounting plugin for the QOS list, and with no slurmdbd
the answer is NULL. The refresh "fails", validation returns an error, and the job gets `InvalidAccount`.
Backfill doesn't run that check, so it starts the job on its next pass. (Source:
`src/slurmctld/job_scheduler.c` and `src/common/assoc_mgr.c`; unchanged through 23.11.11.)

**Wrong turn.** We first suspected `PriorityType=priority/multifactor` (the default, which normally
uses accounting data) and tried `priority/basic`. There was no change, so we reverted it. Change one thing at a
time and undo what didn't work.

**Fix.** `SchedulerParameters=bf_interval=2` makes backfill run every 2 s, so jobs start in about 1–2 s.
Side effect: pending jobs show reason `None` instead of `Resources`. The proper fix is running slurmdbd,
which also gives you `sacct` job history.

---

## 6. Cheat sheet

```bash
# ── cluster ──────────────────────────────────────────────────────────────
scripts/cluster-up.sh | scripts/cluster-down.sh
sinfo  |  sinfo -N -l  |  sinfo -N -o "%N %P %T %G %E"
scontrol show node orin | scontrol show partition | scontrol show config | grep -i sched
sudo scontrol update nodename=orin state=drain reason="why"   # …state=resume
sudo scontrol reconfigure        # re-read slurm.conf (some changes need a daemon restart)

# ── submit ───────────────────────────────────────────────────────────────
sbatch job.sbatch                         sbatch --parsable job.sbatch   # just the id
sbatch --wrap "cmd"                       sbatch -N2 --gres=gpu:1 job.sbatch
srun -N2 hostname                         srun -p agx --gres=gpu:1 --pty bash
salloc -N2 --gres=gpu:1                   # then srun … inside
--gres=gpu:orin:1  -p dgx  -w orin  -x orin  --time=00:10:00  --cpus-per-task=4  --mem=8G
--array=0-7%2      --dependency=afterok:<id>  --output=logs/%x-%j.out  (%A_%a for arrays)

# ── watch / control ──────────────────────────────────────────────────────
squeue  |  squeue -u $USER --start  |  squeue -o "%.6i %.10j %.8T %.12r %.11N %b"
scontrol show job <id>                    scancel <id> | scancel -n <name> | scancel 21_5
scontrol hold <id> / release <id>         scontrol update jobid=<id> TimeLimit=00:30:00

# ── inside a job ─────────────────────────────────────────────────────────
$SLURM_JOB_ID $SLURM_JOB_NODELIST $SLURM_NNODES $SLURM_NTASKS
$SLURM_PROCID $SLURM_LOCALID $SLURM_NODEID $SLURM_ARRAY_TASK_ID $CUDA_VISIBLE_DEVICES
scontrol show hostnames "$SLURM_JOB_NODELIST"

# ── debug ────────────────────────────────────────────────────────────────
sudo tail -f /var/log/slurm/slurmctld.log      # DGX
sudo tail -f /var/log/slurm/slurmd.log         # each node
slurmd -C                                      # what this node really has
munge -n | ssh <other-node> unmunge            # auth works across nodes?
```

---

## 7. What's next

- **Phase 4: distributed training under Slurm.** `train/ddp_train.py` plus `jobs/ddp.sbatch`:
  `-N2 --ntasks-per-node=1 --gres=gpu:1`, master address = the first host in the node list (the `orin`!),
  `srun torchrun --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29500`, backend **gloo** with
  `GLOO_SOCKET_IFNAME` set per host. Public dataset only (CIFAR-10 / tiny-shakespeare) in `data/`.
  Then checkpoint to `ckpt/`, `scancel`, and resume with `--requeue`.
- **Phase 5: wrap-up.** `cluster-down.sh`, confirm the units are disabled at boot, and a blog post.
- **Stretch goals:**
  - slurmdbd + MariaDB, which gives you `sacct`/`sreport` and fixes `InvalidAccount` properly (slurmdbd doesn't support Postgres).
  - `ConstrainDevices=yes` in `cgroup.conf`: jobs can then only open the GPU device files they were
    allocated. Watch what the Orin's extra device nodes (`/dev/nvmap`, render node) do to that.
  - The NVML shim, to get NCCL working on the Orin.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **Allocation** | The set of resources (nodes/CPUs/GPUs/memory) granted to a job |
| **Backfill** | A scheduler that starts lower-priority jobs early when they fit in gaps without delaying higher-priority ones |
| **cgroup** | A Linux kernel feature for grouping and limiting processes. Slurm uses it to track job processes and fence CPUs, memory and devices. |
| **cons_tres** | "Consumable trackable resources": a select plugin that lets several jobs share a node by core, memory and GPU |
| **Drain** | A node state: finish current jobs, accept no new ones |
| **GRES** | Generic resource, e.g. GPUs, declared in `slurm.conf` (`Gres=`) and `gres.conf` (device files) |
| **gloo / NCCL** | PyTorch collective-communication backends: gloo is CPU/TCP and portable, NCCL is GPU-native and topology-aware |
| **Job step** | One `srun` invocation inside a job |
| **munge** | An authentication service where a shared key lets nodes verify each other's credentials |
| **NVML** | NVIDIA Management Library, the API behind `nvidia-smi` |
| **Partition** | A named set of nodes acting as a queue with its own limits |
| **Rank** | A process's index in a distributed job (`SLURM_PROCID`, torch `RANK`) |
| **slurmctld / slurmd / slurmdbd** | Controller / per-node compute daemon / accounting database daemon |
| **Task** | One process launched by `srun` |
