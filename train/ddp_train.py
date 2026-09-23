"""Char-level GPT on tiny-shakespeare with DistributedDataParallel. Launched per node by torchrun (see jobs/ddp.sbatch).

    python ddp_train.py [--steps 300] [--batch 32] [--ckpt ckpt/ddp.pt]

The model is deliberately tiny (~0.8M params, ~3 MB of gradients per step), because gloo over WiFi moves only ~6 MiB/s.
Resumes from --ckpt if it exists, so a requeued job carries on where it left off.
"""
import argparse
import math
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


class TinyGPT(nn.Module):
    def __init__(self, vocab, block, dim=128, layers=4, heads=4):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        return self.head(self.norm(self.blocks(x, mask=mask, is_causal=True)))


def get_batch(data, batch, block, gen, device):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(device), y.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tinyshakespeare.txt")
    p.add_argument("--ckpt", default="ckpt/ddp.pt")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=32, help="per-rank batch size")
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--ckpt-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    dist.init_process_group("gloo")  # NCCL can't init on the Orin (see PLAN.md Phase 1)
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    host = socket.gethostname()

    text = open(args.data).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]

    torch.manual_seed(0)  # same init on every rank (DDP also broadcasts rank 0's weights)
    model = TinyGPT(len(chars), args.block).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    nparams = sum(p.numel() for p in model.parameters())

    start = 0
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"]
    ddp = DDP(model, device_ids=[local_rank])

    print(f"[rank {rank}/{world}] host={host} gpu={torch.cuda.get_device_name(local_rank)} "
          f"params={nparams / 1e6:.2f}M batch={args.batch} "
          f"{'resumed at step ' + str(start) if start else 'fresh start'}", flush=True)

    # Per-rank data stream; seeded by (rank, step) so a resumed run doesn't replay the same batches.
    gen = torch.Generator().manual_seed(1000 * rank + start)
    t_log = time.perf_counter()
    for step in range(start, args.steps):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))  # cosine decay
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train, args.batch, args.block, gen, device)
        loss = F.cross_entropy(ddp(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()  # gradients are all-reduced across nodes here
        opt.step()

        if (step + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t_log) / args.log_every
            t_log = time.perf_counter()
            print(f"[rank {rank} {host}] step {step + 1}/{args.steps} loss {loss.item():.3f} "
                  f"{dt * 1e3:.0f} ms/step", flush=True)

        if (step + 1) % args.ckpt_every == 0 or step + 1 == args.steps:
            if rank == 0:
                tmp = args.ckpt + ".tmp"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1}, tmp)
                os.replace(tmp, args.ckpt)  # atomic: a kill mid-save never leaves a torn checkpoint
                print(f"[rank 0] checkpoint step {step + 1} -> {args.ckpt}", flush=True)
            dist.barrier()

    if rank == 0:
        model.eval()
        with torch.no_grad():
            vx, vy = get_batch(val, 64, args.block, torch.Generator().manual_seed(0), device)
            vloss = F.cross_entropy(model(vx).flatten(0, 1), vy.flatten()).item()
            idx = torch.tensor([[stoi["\n"]]], device=device)
            for _ in range(200):
                probs = F.softmax(model(idx[:, -args.block:])[:, -1], dim=-1)
                idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        print(f"[rank 0] val loss {vloss:.3f}\n--- sample ---{''.join(chars[i] for i in idx[0].tolist())}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
