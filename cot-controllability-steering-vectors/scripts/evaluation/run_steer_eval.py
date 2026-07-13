"""Controlled HELD-OUT steered evaluation.

Applies the VAL-selected single-general TRAIN direction at the chosen (layer, c, sign) to all 9
HELD-OUT instructions (incl. the bullet probe) x a source-stratified held-out-task sample (nested in
the FT deliverable's n=100 sample so the base + FT `cdel` cached gens are reused), against the REQUIRED
controls — all generated with the SAME batching/gen params:
  * base       — no steering (cache hits from the FT deliverable's held-out base gens).
  * real       — the selected direction/(layer,c,sign).
  * signrev    — the negated selected vector (same layer/magnitude).
  * rand0..N   — >=5 random matched-norm directions (same layer, same ||vector||) = a null distribution.
(The FT `cdel` benchmark is pulled from results/ft_eval_cdel_heldout_full_judged.jsonl in analysis.)

Writes results/steer_eval_<tag>.jsonl. Judge with run_steer_judges.py, analyze with analyze_steer_eval.py.

Usage:
  python run_steer_eval.py --tag heldout --layer 12 --c 8 --sign 1 --n-random 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal

import gpt_oss_infer as G
import steering_lib as S
import steer_eval_lib as E
from run_ft_eval import subsample_tasks, HELDOUT_INSTRS

RESULTS = Path("results")
# Held-out task sample (~39/instr); NESTED in HELDOUT_SIZES_BIG (n=100) via the shared "heldout" key,
# so the FT-deliverable base + cdel gens for these tasks are already cached.
STEER_HELDOUT_SIZES = {"arc_challenge": 8, "gsm8k": 8, "openbookqa": 8, "mmlu_pro": 7,
                       "math": 5, "reasonif": 3}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="heldout")
    p.add_argument("--direction", default="pooled")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--c", type=float, required=True)
    p.add_argument("--sign", type=int, default=1)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--mnt-cap", type=int, default=0,
                   help="cap max_new_tokens for ALL controlled arms (apples-to-apples + faster; "
                        "recovery disabled when capped). 0 = full MNT_BY_SOURCE + recovery.")
    p.add_argument("--sizes", default="", help="json override of per-source sizes")
    p.add_argument("--no-base", action="store_true")
    p.add_argument("--assert-cached", action="store_true")
    args = p.parse_args()

    groups, resid_norm, meta = S.load_directions()
    sizes = json.loads(args.sizes) if args.sizes else STEER_HELDOUT_SIZES
    tasks = subsample_tasks("heldout", sizes, "heldout")
    conds = list(HELDOUT_INSTRS) + ["none"]
    cond_tasks = {c: tasks for c in conds}
    items = E.build_items(conds, cond_tasks)
    print(f"{len(items)} (cond,task) held-out items ({len(tasks)} tasks x {len(conds)} conds)")

    real_steer, real_norm = E.make_steering(groups, resid_norm, args.direction, args.layer,
                                             args.c, args.sign)
    signrev_steer, _ = E.make_steering(groups, resid_norm, args.direction, args.layer,
                                       args.c, -args.sign)
    arms = []
    if not args.no_base:
        arms.append(("base", None, {"kind": "base"}))
    arms.append(("real", real_steer, {"kind": "real", "layer": args.layer, "c": args.c,
                                       "sign": args.sign, "norm": real_norm, "direction": args.direction}))
    arms.append(("signrev", signrev_steer, {"kind": "signrev", "layer": args.layer, "c": args.c,
                                            "sign": -args.sign, "norm": real_norm}))
    for s in range(args.n_random):
        rsteer, rnorm = E.make_random_steering(args.layer, real_norm, seed=1000 + s)
        arms.append((f"rand{s}", rsteer, {"kind": "random", "layer": args.layer, "seed": 1000 + s,
                                          "norm": rnorm}))
    print(f"{len(arms)} arms: {[a[0] for a in arms]}; real ||vector||={real_norm:.1f}")

    meta_run = {"tag": args.tag, "direction": args.direction, "layer": args.layer, "c": args.c,
                "sign": args.sign, "n_random": args.n_random, "real_norm": real_norm,
                "sizes": sizes, "git_hash": git_hash(),
                "timestamp": datetime.now(timezone.utc).isoformat()}

    cache, tracker = G._cache_and_cost()
    out_path = RESULTS / f"steer_eval_{args.tag}.jsonl"
    f = open(out_path, "w")
    n = 0
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        for name, steering, cfg in arms:
            # When mnt-capped, ALL arms use the same cap + no recovery (clean apples-to-apples,
            # faster). Uncapped: recover (8192) only base/real/signrev (random can't have legit
            # long-compliant traces).
            if args.mnt_cap:
                rec = False
            else:
                rec = cfg.get("kind") in ("base", "real", "signrev")
            gbk, mbk = E.gen_arm(obj, items, "base", steering, cache, tracker, args.assert_cached,
                                 recover=rec, verbose=True, mnt_cap=args.mnt_cap)
            for cond, t, toks, src in items:
                g = gbk[(cond, t["task_id"])]
                extra = {"steer_kind": cfg.get("kind"), "steer_layer": cfg.get("layer"),
                         "steer_c": cfg.get("c"), "steer_sign": cfg.get("sign"),
                         "steer_norm": cfg.get("norm"), "steer_seed": cfg.get("seed"),
                         "direction": args.direction, "metadata": meta_run}
                row = E.row_for(name, cond, t, src, g, mbk[(cond, t["task_id"])], extra)
                f.write(json.dumps(row) + "\n")
                n += 1
            print(f"  arm {name}: done ({cfg.get('kind')})")
    f.close()
    print(f"\nWrote {n} rows to {out_path}")
    print(f"Modal cost this run: ${tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
