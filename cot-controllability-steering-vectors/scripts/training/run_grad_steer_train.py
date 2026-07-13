"""Train a small ADDITIVE steering bias by gradient descent, weights FROZEN.

Objective = completion-only NLL of the complying TRAIN targets (the SAME loss family as the LoRA FT,
but the only trainable parameter is a fixed additive bias added to resid_post at one (or a small set
of) layers). Trained on TRAIN instructions ONLY -> held-out (esp. formatting/bullet) is a genuine
generalization test. Persists the trained vector(s) to data/grad_steer_<tag>.npz + provenance, and
the loss curve + config + SIZE (param count, #layers) to results/grad_steer_train_<tag>.json.

The matched "trained-on-control" arm trains the SAME way on the NON-complying raw-trace control
targets (--which control --match-control): it should NOT raise compliance (the steering analogue of
the FT raw-trace control).

Usage:
  python run_grad_steer_train.py --tag gL8  --layers 8       --which compliant --match-control --steps 600
  python run_grad_steer_train.py --tag gML  --layers 6 8 10  --which compliant --match-control --steps 600
  python run_grad_steer_train.py --tag gL8ctrl --layers 8    --which control   --match-control --steps 600
  python run_grad_steer_train.py --assert-cached --tag gL8 ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np

import gpt_oss_infer as G
import grad_steer_lib as GS

RESULTS = Path("results")


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--which", default="compliant", choices=["compliant", "control"])
    p.add_argument("--match-control", action="store_true",
                   help="restrict to prompts shared with the control set (clean compliant-vs-control)")
    p.add_argument("--layers", type=int, nargs="+", default=[8])
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1280)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--assert-cached", action="store_true")
    args = p.parse_args()

    rows = GS.load_train_rows(args.which, args.match_control)
    seqs, starts, metas, dropped = GS.build_sequences(rows, args.max_length)
    dh = GS.data_hash(seqs, starts, args.max_length)
    print(f"[train] tag={args.tag} which={args.which} match_control={args.match_control}: "
          f"{len(seqs)} seqs ({dropped} dropped > {args.max_length} tok); layers={args.layers} "
          f"data_hash={dh}")

    config = {"layers": args.layers, "n_steps": args.steps, "lr": args.lr,
              "batch_size": args.batch_size, "seed": args.seed, "warmup_frac": args.warmup_frac,
              "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
              "max_length": args.max_length, "grad_ckpt": True, "kl_coef": args.kl_coef,
              "log_every": 25}

    cache, tracker = G._cache_and_cost()
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        res = G.train_steering_vectors(obj, seqs, starts, config, dh, cache, tracker,
                                       assert_cached=args.assert_cached)

    vectors = res["vectors"]
    GS.save_vectors(args.tag, args.layers, vectors, {
        "tag": args.tag, "which": args.which, "match_control": args.match_control,
        "layers": args.layers, "config": config, "data_hash": dh,
        "n_train_seqs": len(seqs), "n_dropped": dropped,
        "param_count": GS.param_count(vectors), "n_layers": len(args.layers),
        "vector_norms": res["vector_norms"],
        "final_loss": res["losses"][-1] if res["losses"] else None,
        "initial_loss": res["losses"][0] if res["losses"] else None,
        "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    rec = {"tag": args.tag, "which": args.which, "layers": args.layers, "config": config,
           "data_hash": dh, "n_train_seqs": len(seqs), "n_dropped": dropped,
           "param_count": GS.param_count(vectors), "n_layers": len(args.layers),
           "vector_norms": res["vector_norms"], "losses": res["losses"],
           "kl_losses": res["kl_losses"], "grad_norms": res["grad_norms"],
           "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat()}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"grad_steer_train_{args.tag}.json").write_text(json.dumps(rec, indent=2))

    ls = res["losses"]
    sm = lambda a: float(np.mean(a)) if a else float("nan")
    print(f"\n===== trained vector tag={args.tag} =====")
    print(f"  params={GS.param_count(vectors)} ({len(args.layers)} layers x {len(vectors[0])}); "
          f"||v||={[round(x,1) for x in res['vector_norms']]}")
    print(f"  loss: first25 mean {sm(ls[:25]):.4f} -> last25 mean {sm(ls[-25:]):.4f}")
    print(f"  saved data/grad_steer_{args.tag}.npz + results/grad_steer_train_{args.tag}.json")
    print(f"Modal cost this run: ${tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
