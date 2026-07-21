"""Train AdamW or Kenian on the vision and language benchmarks.

Runs are resumable and write metrics, configuration, and a final summary to
``results/<task>/<run_name>/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import CIFAR100Loader, WikiTextLoader
from kenian import Kenian
from models import GPT, ViT, count_params
from torch.nn.attention import sdpa_kernel, SDPBackend

RESULTS = Path(__file__).resolve().parents[1] / "results"


def set_seed(seed: int) -> None:
    """Seed all random generators used by a training run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_at(
    step: int,
    peak: float,
    warmup: int,
    total: int,
    floor_fraction: float = 0.1,
) -> float:
    """Return a linear-warmup, cosine-decay learning rate."""
    if step < warmup:
        return peak * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return peak * (floor_fraction + (1.0 - floor_fraction) * cosine)


def build_model(task: str) -> torch.nn.Module:
    return ViT() if task == "vision" else GPT()


def make_optimizer(
    name: str,
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
    backend: str,
    probe_interval: int,
    correction_cap: float,
    beta3: float,
) -> torch.optim.Optimizer:
    params = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    return Kenian(
        params,
        lr=learning_rate,
        beta3=beta3,
        eps=1e-8,
        weight_decay=weight_decay,
        correction_cap=correction_cap,
        probe_interval=probe_interval,
        backend=backend,
    )


def loss_fn(task, model, inputs, targets):
    if task == "vision":
        logits = model(inputs)
        return torch.nn.functional.cross_entropy(logits, targets), logits
    logits, loss = model(inputs, targets)
    return loss, logits


@torch.no_grad()
def evaluate(task, model, loader, max_windows=None):
    model.eval()
    if task == "vision":
        tot_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader.iter_eval(batch=500):
            logits = model(xb)
            tot_loss += torch.nn.functional.cross_entropy(
                logits, yb, reduction="sum"
            ).item()
            correct += (logits.argmax(1) == yb).sum().item()
            n += yb.numel()
        model.train()
        return {"val_loss": tot_loss / n, "val_acc": correct / n}
    tot_loss, n = 0.0, 0
    for xb, yb in loader.iter_eval(max_windows=max_windows):
        _, loss = model(xb, yb)
        k = yb.numel()
        tot_loss += loss.item() * k
        n += k
    model.train()
    vl = tot_loss / n
    return {"val_loss": vl, "val_ppl": math.exp(min(vl, 20))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["vision", "language"], required=True)
    parser.add_argument("--optimizer", choices=["adamw", "kenian"], required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--probe-interval", type=int, default=10)
    parser.add_argument("--probe-batch", type=int, default=None)
    parser.add_argument("--correction-cap", type=float, default=0.1)
    parser.add_argument("--beta3", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tag", default="")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--fast-validation-every", type=int, default=500)
    parser.add_argument("--full-validation-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    device = "cuda"
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    batch = args.batch or (128 if args.task == "vision" else 24)
    wd = args.wd if args.wd is not None else (0.05 if args.task == "vision" else 0.1)
    probe_batch = args.probe_batch or (32 if args.task == "vision" else 8)

    run = args.tag or f"{args.optimizer}_lr{args.lr:g}_seed{args.seed}"
    output_dir = Path(args.outdir) if args.outdir else RESULTS / args.task / run
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    checkpoint_path = output_dir / "ckpt.pt"

    model = build_model(args.task).to(device)
    nparam = count_params(model)
    opt = make_optimizer(
        args.optimizer,
        model,
        args.lr,
        wd,
        args.backend,
        args.probe_interval,
        args.correction_cap,
        args.beta3,
    )
    is_kenian = isinstance(opt, Kenian)

    if args.task == "vision":
        train_loader = CIFAR100Loader(
            batch=batch, device=device, seed=args.seed, train=True
        )
        val_loader = CIFAR100Loader(batch=batch, device=device, seed=0, train=False)
    else:
        train_loader = WikiTextLoader(
            ctx=512, batch=batch, device=device, seed=args.seed, split="train"
        )
        val_loader = WikiTextLoader(
            ctx=512, batch=batch, device=device, seed=0, split="validation"
        )

    start_step = 0
    if checkpoint_path.exists():
        ck = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"[resume] {run} from step {start_step}", flush=True)

    meta = dict(
        task=args.task,
        optimizer=args.optimizer,
        lr=args.lr,
        seed=args.seed,
        steps=args.steps,
        batch=batch,
        wd=wd,
        nparam=nparam,
        backend=args.backend,
        probe_interval=args.probe_interval,
        correction_cap=args.correction_cap,
        beta3=args.beta3,
        clip=args.clip,
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Truncate on a fresh run (start_step==0) so re-running a tag never mixes old + new
    # records; append only when genuinely resuming from a checkpoint.
    logf = log_path.open("a" if start_step > 0 else "w", encoding="utf-8")

    def log(d):
        logf.write(json.dumps(d) + "\n")
        logf.flush()

    model.train()
    # Keep the primary loss numerically stable in mixed precision. The third-order
    # probe stays in full precision and is never scaled.
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    t_start = time.time()
    run_t0 = time.time()
    for step in range(start_step, args.steps):
        lr = learning_rate_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        xb, yb = train_loader.sample()
        opt.zero_grad(set_to_none=True)
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext()
        )
        with amp_ctx:
            loss, _ = loss_fn(args.task, model, xb, yb)
        if is_kenian:
            if opt.needs_probe():
                # The probe uses the MATH attention backend because it supports
                # double backward. A smaller batch keeps its graph manageable.
                # OOM-resilient: on a memory spike, shrink the probe sub-batch and retry;
                # if it still fails, fall back to a plain step (no probe) so a 40k run
                # never dies on one spike. Reductions are logged.
                done = False
                while not done:
                    try:
                        with (
                            sdpa_kernel([SDPBackend.MATH]),
                            torch.autocast("cuda", enabled=False),
                        ):
                            pl, _ = loss_fn(
                                args.task, model, xb[:probe_batch], yb[:probe_batch]
                            )
                        opt.backward_and_prepare(loss, probe_loss=pl, scaler=scaler)
                        done = True
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        opt.zero_grad(set_to_none=True)
                        if probe_batch > 2:
                            probe_batch = max(2, probe_batch // 2)
                            print(
                                f"[oom] step {step + 1}: probe_batch -> {probe_batch}",
                                flush=True,
                            )
                            with amp_ctx:
                                loss, _ = loss_fn(args.task, model, xb, yb)
                        else:
                            print(
                                f"[oom] step {step + 1}: probe skipped this step",
                                flush=True,
                            )
                            with amp_ctx:
                                loss, _ = loss_fn(args.task, model, xb, yb)
                            scaler.scale(loss).backward()
                            done = True
            else:
                opt.backward_and_prepare(loss, scaler=scaler)
        else:
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
        scaler.step(opt)
        scaler.update()

        if (step + 1) % args.log_every == 0:
            torch.cuda.synchronize()
            dt = (time.time() - t_start) / args.log_every * 1000
            t_start = time.time()
            rec = {
                "step": step + 1,
                "train_loss": loss.item(),
                "grad_norm": gnorm,
                "lr": lr,
                "step_ms": dt,
                "wall_s": time.time() - run_t0,
            }
            if is_kenian:
                rec["kenian"] = {k: round(v, 6) for k, v in opt.diagnostics.items()}
            log(rec)

        if (step + 1) % args.fast_validation_every == 0:
            mw = None if args.task == "vision" else 96
            ev = evaluate(args.task, model, val_loader, max_windows=mw)
            ev.update(
                {"step": step + 1, "val_kind": "fast", "wall_s": time.time() - run_t0}
            )
            log(ev)
            msg = f"[{run}] step {step + 1}/{args.steps} loss {loss.item():.4f} val {ev['val_loss']:.4f}"
            print(msg, flush=True)

        if (step + 1) % args.full_validation_every == 0 or (step + 1) == args.steps:
            ev = evaluate(args.task, model, val_loader, max_windows=None)
            ev.update(
                {"step": step + 1, "val_kind": "full", "wall_s": time.time() - run_t0}
            )
            log(ev)

        if (step + 1) % args.checkpoint_every == 0 or (step + 1) == args.steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step + 1,
                },
                checkpoint_path,
            )

    final = evaluate(args.task, model, val_loader, max_windows=None)
    final.update(
        {
            "step": args.steps,
            "val_kind": "final",
            "total_wall_s": time.time() - run_t0,
            "nparam": nparam,
        }
    )
    with (output_dir / "final.json").open("w", encoding="utf-8") as f:
        json.dump({**meta, **final}, f, indent=2)
    log(final)
    logf.close()
    print(
        f"[done] {run}: final val_loss {final['val_loss']:.4f} "
        f"({'acc %.4f' % final['val_acc'] if 'val_acc' in final else 'ppl %.2f' % final['val_ppl']}) "
        f"in {final['total_wall_s'] / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
