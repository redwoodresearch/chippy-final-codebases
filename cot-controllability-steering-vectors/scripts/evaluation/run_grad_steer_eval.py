"""Evaluate gradient-trained steering vectors on VAL (search) or HELD-OUT (report).

Each arm is either ``base`` (no steering) or a persisted trained-vector tag (e.g. ``gL8``) applied
via the existing steering path. On VAL we print a cheap raw-compliance ranking (no judges) to pick
the config; on held-out we generate all arms (base + the chosen trained vector + its trained-on-
control twin + extra variants) with the SAME batching/mnt cap for an apples-to-apples controlled
eval (judge with run_steer_judges.py, analyze with analyze_grad_steer_eval.py).

Usage:
  # VAL search over several trained tags:
  python run_grad_steer_eval.py --preset val --tags gL6 gL8 gL10 gML --mnt-cap 512 --train-check
  # held-out report on the chosen vector + control twin:
  python run_grad_steer_eval.py --preset heldout --tag-run hg1 --tags gL8 gL8ctrl gML --mnt-cap 1024
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import modal
import numpy as np

import gpt_oss_infer as G
import grad_steer_lib as GS
import steer_eval_lib as E
import instructions as I
from run_ft_eval import (subsample_tasks, VAL_SIZES, VAL_INSTRS, HELDOUT_INSTRS,
                         HELDOUT_SIZES_BIG)
from run_steer_eval import STEER_HELDOUT_SIZES
from sft_edit import is_degenerate

RESULTS = Path("results")
TRAIN_CHECK = ["all_caps", "no_the", "questions", "brief_50w"]


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="val", choices=["val", "heldout", "deliverable"])
    p.add_argument("--tags", nargs="*", default=[], help="trained-vector tags = steering arms")
    p.add_argument("--tag-run", default="", help="output run tag (default = preset)")
    p.add_argument("--mnt-cap", type=int, default=512)
    p.add_argument("--train-check", action="store_true", help="(val) add within-distribution TRAIN instrs")
    p.add_argument("--sizes", default="")
    p.add_argument("--no-base", action="store_true")
    p.add_argument("--no-none", action="store_true", help="(heldout/deliverable) skip the none condition")
    p.add_argument("--rand-seeds", type=int, nargs="*", default=[],
                   help="add random matched-norm null arms at --rand-layer / --rand-norm for each seed")
    p.add_argument("--rand-layer", type=int, default=10)
    p.add_argument("--rand-norm", type=float, default=0.0,
                   help="norm for the random nulls (0 = match gL10's trained ||v||)")
    p.add_argument("--assert-cached", action="store_true")
    args = p.parse_args()

    run_tag = args.tag_run or args.preset
    if args.preset == "val":
        sizes = json.loads(args.sizes) if args.sizes else VAL_SIZES
        tasks = subsample_tasks("val", sizes, "val")
        conds = list(VAL_INSTRS)
        cond_tasks = {c: tasks for c in conds}
        if args.train_check:
            for c in TRAIN_CHECK:
                cond_tasks[c] = tasks
            conds = conds + TRAIN_CHECK
        conds = conds + ["none"]
        cond_tasks["none"] = tasks
    else:
        default_sizes = HELDOUT_SIZES_BIG if args.preset == "deliverable" else STEER_HELDOUT_SIZES
        sizes = json.loads(args.sizes) if args.sizes else default_sizes
        tasks = subsample_tasks("heldout", sizes, "heldout")
        conds = list(HELDOUT_INSTRS) + ([] if args.no_none else ["none"])
        cond_tasks = {c: tasks for c in conds}
    items = E.build_items(conds, cond_tasks)
    print(f"[{args.preset}] {len(tasks)} tasks x {len(conds)} conds = {len(items)} items")

    arms = []  # (name, steering, cfg)
    if not args.no_base:
        arms.append(("base", None, {"kind": "base"}))
    for tag in args.tags:
        steering, total_norm, layers, vectors = GS.make_steering(tag)
        arms.append((tag, steering, {"kind": "trained", "tag": tag, "layers": layers,
                                     "norm": total_norm, "param_count": GS.param_count(vectors)}))
        print(f"  arm {tag}: layers={layers} ||v||={total_norm:.1f} params={GS.param_count(vectors)}")
    if args.rand_seeds:
        rnorm = args.rand_norm or GS.make_steering("gL10")[1]
        for s in args.rand_seeds:
            steering, snorm = E.make_random_steering(args.rand_layer, rnorm, s)
            name = f"randL{args.rand_layer}s{s}"
            arms.append((name, steering, {"kind": "random", "tag": name, "layers": [args.rand_layer],
                                          "norm": snorm, "seed": s, "param_count": G.HIDDEN_SIZE}))
            print(f"  arm {name}: random L{args.rand_layer} ||v||={snorm:.1f} seed={s}")

    meta_run = {"run_tag": run_tag, "preset": args.preset, "tags": args.tags, "sizes": sizes,
                "mnt_cap": args.mnt_cap, "no_none": args.no_none,
                "rand_seeds": args.rand_seeds, "rand_layer": args.rand_layer,
                "rand_norm": (args.rand_norm or (GS.make_steering("gL10")[1] if args.rand_seeds else 0.0)),
                "git_hash": git_hash(),
                "timestamp": datetime.now(timezone.utc).isoformat()}

    cache, tracker = G._cache_and_cost()
    out_path = RESULTS / f"grad_steer_eval_{args.preset}_{run_tag}.jsonl"
    f = open(out_path, "w")
    n = 0
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        for name, steering, cfg in arms:
            rec = (cfg["kind"] in ("base", "trained")) and not args.mnt_cap
            gbk, mbk = E.gen_arm(obj, items, "base", steering, cache, tracker, args.assert_cached,
                                 recover=rec, verbose=True, mnt_cap=args.mnt_cap)
            for cond, t, toks, src in items:
                g = gbk[(cond, t["task_id"])]
                extra = {"steer_kind": cfg.get("kind"), "steer_tag": cfg.get("tag"),
                         "steer_layers": cfg.get("layers"), "steer_norm": cfg.get("norm"),
                         "param_count": cfg.get("param_count"), "metadata": meta_run}
                row = E.row_for(name, cond, t, src, g, mbk[(cond, t["task_id"])], extra)
                f.write(json.dumps(row) + "\n")
                n += 1
            print(f"  arm {name}: done ({cfg.get('kind')})")
    f.close()
    print(f"\nWrote {n} rows to {out_path}")
    if args.preset == "val":
        _cheap_rank(out_path, [a[0] for a in arms])
    print(f"Modal cost this run: ${tracker.run_cost:.4f}")


def _cheap_rank(path, arm_names):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["arm"]][r["condition"]].append(r)
    val_conds = list(VAL_INSTRS)

    def arm_metrics(arm, conds):
        rcs, mals, accs, degs = [], [], [], []
        for cond in conds:
            for r in by[arm][cond]:
                instr = I.INSTRUCTIONS.get(cond)
                if instr and instr.scorer is not None:
                    rcs.append(False if r["malformed"] else bool(instr.scorer(r["analysis"])))
                mals.append(r["malformed"])
                accs.append(r["accuracy"])
                degs.append(is_degenerate(r.get("analysis", "")))
        m = lambda x: (100 * sum(bool(v) for v in x) / len(x)) if x else float("nan")
        return m(rcs), m(mals), m(accs), m(degs)

    base_rc = arm_metrics("base", val_conds)[0] if "base" in by else 0.0
    print("\n=== CHEAP VAL ranking (no judges): raw-compliance / malformed / acc / degen ===")
    print(f"  base VAL raw-compliance: {base_rc:.0f}%")
    print(f"  {'arm':10s} {'rawVAL':>7s} {'Δraw':>6s} {'malf':>6s} {'acc':>6s} {'degen':>6s} "
          f"{'noneMalf':>9s} {'noneDeg':>8s}")
    res = []
    for name in arm_names:
        if name == "base":
            continue
        rc, mal, acc, deg = arm_metrics(name, val_conds)
        n_mal, _, _, n_deg = arm_metrics(name, ["none"])[1], 0, 0, arm_metrics(name, ["none"])[3]
        res.append((name, rc, rc - base_rc, mal, acc, deg, n_mal, n_deg))
    for name, rc, drc, mal, acc, deg, n_mal, n_deg in sorted(res, key=lambda x: (-x[2], x[3])):
        print(f"  {name:10s} {rc:6.0f}% {drc:+5.0f} {mal:5.0f}% {acc:5.0f}% {deg:5.0f}% "
              f"{n_mal:8.0f}% {n_deg:7.0f}%")


if __name__ == "__main__":
    main()
