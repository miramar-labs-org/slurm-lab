"""All-reduce smoke test across nodes. Launch with torchrun (see PLAN.md Phase 1).

    python nccl_test.py [--backend nccl|gloo] [--mb 16] [--iters 20]
"""
import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--mb", type=float, default=16, help="tensor size in MiB (fp32)")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    # gloo all-reduces CPU tensors; nccl needs them on the GPU.
    device = torch.device("cuda", local_rank) if args.backend == "nccl" else torch.device("cpu")

    print(f"[rank {rank}/{world}] host={socket.gethostname()} gpu={torch.cuda.get_device_name(local_rank)} "
          f"backend={args.backend}", flush=True)

    # Correctness: each rank contributes rank+1, so the sum is world*(world+1)/2.
    t = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(t)
    expected = world * (world + 1) / 2
    assert torch.allclose(t, torch.full_like(t, expected)), t
    print(f"[rank {rank}] correctness OK: {t.tolist()}", flush=True)

    # Timing
    n = int(args.mb * 2**20 / 4)
    x = torch.ones(n, device=device)
    dist.all_reduce(x)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters

    # Ring all-reduce moves 2*(world-1)/world of the buffer per rank ("bus bandwidth", as in nccl-tests).
    busbw = args.mb * 2 * (world - 1) / world / dt
    if rank == 0:
        print(f"all_reduce {args.mb} MiB x {args.iters}: {dt * 1e3:.1f} ms/iter, busbw {busbw:.1f} MiB/s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
