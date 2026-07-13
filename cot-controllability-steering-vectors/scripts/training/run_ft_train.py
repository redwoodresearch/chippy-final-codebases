"""Train ONE LoRA configuration on an SFT set via Tinker.

Engine = Tinker LoRA (rank 32, adapts MLP/MoE-experts + attention + unembed by default), fed EXACT
pre-rendered Harmony tokens (no 'Current date' line; rendering alignment proven by
verify_render_alignment.py) with a COMPLETION-ONLY loss mask. Saves sampler weights, downloads the
adapter to a stable dir, and records the loss curve + config to results/ft_train_<tag>.json.

The downstream durable artifact is the CPU-merged MXFP4 model on the Modal volume (run_ft_merge.py),
loaded into the HF harness for eval. Tinker training is not file-cached (it's a one-time FT run); we
make it idempotent (skip if results + adapter already exist, unless --force).

Usage:
  python run_ft_train.py --which compliant --tag c32   # rank 32, lr 2e-4, 3 epochs (defaults)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TINKER_API_KEY", open(os.path.expanduser("~/.tinker_key")).read().strip())
os.environ.setdefault("HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())

import tinker  # noqa: E402
from tinker_cookbook import weights  # noqa: E402

import ft_data  # noqa: E402
from cost_tracker import CostTracker  # noqa: E402

BASE_MODEL = "openai/gpt-oss-20b"
RESULTS = Path("results")


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def lr_at(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    """Linear warmup then linear decay to ~0."""
    warmup = max(1, int(warmup_frac * total))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base_lr * max(0.0, 1.0 - prog)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", default="compliant", choices=list(ft_data.FILES))
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)  # the c32/ctrl32 artifacts were trained at 3
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--plain-frac", type=float, default=0.0,
                   help="fraction of plain (no-instruction) rows to mix in (degeneracy fix)")
    p.add_argument("--plain-n", type=int, default=0,
                   help="absolute number of plain rows to mix in (overrides --plain-frac; lets the "
                        "compliant + control arms share the IDENTICAL plain set)")
    p.add_argument("--match-control", action="store_true",
                   help="(compliant only) restrict to the prompts shared with the control set so the "
                        "compliant vs control comparison differs ONLY in target compliance")
    p.add_argument("--no-unembed", action="store_true",
                   help="do NOT LoRA the unembed (target modules = mlp+attn only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="c32")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    adapter_dir = f"/tmp/ft_adapter_{args.tag}/raw"
    out_json = RESULTS / f"ft_train_{args.tag}.json"
    if out_json.exists() and Path(adapter_dir).exists() and not args.force:
        rec = json.loads(out_json.read_text())
        print(f"[skip] {out_json} + adapter dir exist (use --force to retrain). "
              f"sampler={rec.get('sampler_tinker_path')}")
        return

    tracker = CostTracker(Path("total_cost.jsonl"),
                          run_description=f"tinker FT {args.which} tag={args.tag}")
    t0 = time.monotonic()
    rows = ft_data.load_rows_mixed(args.which, plain_frac=args.plain_frac, plain_n=args.plain_n,
                                   seed=args.seed, match_control=args.match_control)
    dhash = ft_data.data_hash(rows)
    print(f"Building datums for {len(rows)} '{args.which}' examples (data_hash={dhash}) ...")
    datums, kept, stats = ft_data.build_datums(rows, args.max_length)
    print(f"  {stats}")

    # tinker fixes lora_alpha=rank; the tunable target-module levers are train_mlp/attn/unembed.
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=args.rank, seed=args.seed,
                                        train_unembed=not args.no_unembed)
    targets = "mlp(MoE)+attn" + ("" if args.no_unembed else "+unembed")
    print(f"Created LoRA client: rank={args.rank} alpha={args.rank} ({targets}), lr={args.lr}, "
          f"epochs={args.epochs}, batch={args.batch_size}, plain_frac={args.plain_frac}")

    # steps
    n = len(datums)
    steps_per_epoch = math.ceil(n / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    rng = random.Random(args.seed)
    losses, lrs, step_idx = [], [], 0
    for epoch in range(args.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for b in range(0, n, args.batch_size):
            batch = [datums[i] for i in order[b:b + args.batch_size]]
            cur_lr = lr_at(step_idx, total_steps, args.lr, args.warmup_frac)
            fwd = tc.forward_backward(batch, loss_fn="cross_entropy")
            opt = tc.optim_step(tinker.AdamParams(learning_rate=cur_lr))
            fb = fwd.result(); opt.result()
            loss_sum = (getattr(fb, "metrics", {}) or {}).get("loss:sum")
            # reduction="mean" => loss:sum == sum of per-example mean-NLL; divide for mean-per-example
            mean_loss = float(loss_sum) / len(batch) if loss_sum is not None else None
            losses.append(mean_loss)
            lrs.append(cur_lr)
            if step_idx % 5 == 0 or step_idx == total_steps - 1:
                print(f"  epoch {epoch} step {step_idx}/{total_steps} lr={cur_lr:.2e} "
                      f"mean_loss={mean_loss:.4f}")
            step_idx += 1

    # save sampler weights + export adapter
    save_res = tc.save_weights_for_sampler(name=f"ft-{args.tag}").result()
    sampler_path = save_res.path
    print(f"Saved sampler weights: {sampler_path}")
    shutil.rmtree(f"/tmp/ft_adapter_{args.tag}", ignore_errors=True)
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    dl_dir = weights.download(tinker_path=sampler_path, output_dir=adapter_dir)
    adapter_files = sorted(os.listdir(dl_dir))
    # content hash of the adapter (durable merge cache key)
    import hashlib
    ah = hashlib.sha256()
    for fn in adapter_files:
        fp = Path(dl_dir) / fn
        if fp.is_file():
            ah.update(fn.encode())
            ah.update(fp.read_bytes())
    adapter_hash = ah.hexdigest()[:16]

    elapsed = time.monotonic() - t0
    # rough Tinker cost estimate: token-steps trained (sum of completion tokens * epochs), ~$0.5/1M
    token_steps = stats["total_completion_tokens"] * args.epochs
    est_cost = token_steps / 1e6 * 0.50
    tracker.add_cost(est_cost)

    rec = {
        "which": args.which, "tag": args.tag, "base_model": BASE_MODEL,
        "rank": args.rank, "alpha": args.rank, "plain_frac": args.plain_frac, "plain_n": args.plain_n,
        "target_modules": targets, "no_unembed": args.no_unembed,
        "match_control": args.match_control,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "warmup_frac": args.warmup_frac, "seed": args.seed,
        "lora_targets": targets,
        "reduction": "mean", "loss_mask": "completion_only",
        "n_examples": stats["n_examples"], "n_skipped_too_long": stats["n_skipped_too_long"],
        "data_hash": dhash, "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
        "losses": losses, "lrs": lrs, "final_loss": losses[-1] if losses else None,
        "initial_loss": losses[0] if losses else None,
        "sampler_tinker_path": sampler_path, "adapter_dir": dl_dir,
        "adapter_files": adapter_files, "adapter_hash": adapter_hash,
        "total_completion_tokens": stats["total_completion_tokens"],
        "approx_tinker_cost_est": est_cost, "wall_clock_s": elapsed,
        "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(rec, indent=2))
    print(f"\n===== FT TRAIN ({args.which}, tag={args.tag}) =====")
    print(f"examples={stats['n_examples']} steps={total_steps} "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"adapter_hash={adapter_hash} dir={dl_dir} files={adapter_files}")
    print(f"sampler_path={sampler_path}")
    print(f"wall={elapsed:.0f}s  est_tinker_cost=${est_cost:.4f}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
