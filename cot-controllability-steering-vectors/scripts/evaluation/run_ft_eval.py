"""Preliminary held-out CoT-compliance eval: FT model vs MATCHED base.

Evaluates the fine-tuned gpt-oss-20b (merged MXFP4 on the volume, loaded into the HF harness) on
HELD-OUT instructions x HELD-OUT tasks, plus a few TRAIN instructions (the "did it learn?" check)
and the no-instruction condition (accuracy / default-behaviour). The BASE model is generated on the
EXACT same (task x instruction) subset so the FT-vs-base delta is apples-to-apples (both later scored
with the SAME Opus-meta judge pipeline by run_ft_eval_judges.py).

Same generation params as eval: medium effort, greedy (temp 0), seed 0, per-source max_new_tokens +
a truncation-recovery pass at 8192. Held-out tasks are split=="heldout" (disjoint from the train
tasks the FT model saw) -> a genuine generalization test, NOT memorization.

Writes results/ft_eval_<tag>.jsonl (one row per (arm, task, condition); arm in {base, ft}), with
full analysis/final text + flags, for downstream judging + analysis. GPU pass only (no LLM judges).

Usage:
  python run_ft_eval.py --tag c32 --merged-path /cache/merged_ft_c32
  python run_ft_eval.py --tag c32 --assert-cached
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modal  # noqa: E402

import harmony_utils as H  # noqa: E402
import gpt_oss_infer as G  # noqa: E402
import answer_scoring as A  # noqa: E402
import instructions as I  # noqa: E402
from run_base_accuracy import MNT_BY_SOURCE, BS_BY_SOURCE, length_rank  # noqa: E402
from run_instruction_baseline import none_user_content, instr_user_content  # noqa: E402

RESULTS = Path("results")
REASONING_EFFORT = "medium"
TEMPERATURE = 0.0
SEED = 0
RECOVERY_MNT = 8192

# Source-stratified held-out task sample (shared across all conditions). ~18 tasks/instruction;
# long/expensive sources (math, reasonif) get fewer since base compliance is ~0 there.
HELDOUT_SIZES = {"arc_challenge": 4, "gsm8k": 4, "openbookqa": 4, "mmlu_pro": 3, "math": 2,
                 "reasonif": 1}

# Held-out instructions (the generalization test). The ENTIRE formatting category is cross-category
# (no instruction trained); the others are within-category novel instances of TRAINED categories.
HELDOUT_INSTRS = ["initial_caps", "no_word_so", "include_exactly_twice",
                  "bullet", "numbered", "section_headers", "xml_steps",
                  "terse_25w", "child_explanation"]
CROSS_CATEGORY = {"bullet", "numbered", "section_headers", "xml_steps"}  # formatting (never trained)

# A few TRAIN instructions (one per trained category) to confirm the FT actually LEARNED to comply
# (train-instruction compliance should rise a lot). Evaluated on the same held-out TASKS.
TRAIN_CHECK_INSTRS = ["all_caps", "no_the", "include_therefore", "brief_50w", "questions"]

# VAL instructions (Priority-1 sweep selection proxy). All within-category, all programmatic; the FT
# never trains on these so they probe generalization. Selection NEVER touches the held-out set.
VAL_INSTRS = ["no_answer_word", "include_quote_marker", "under_70w", "no_first_person"]

# Generous source-stratified held-out sample for the DELIVERABLE (~100 tasks/instruction). Nested
# superset of the smoke's 18 (same per-source shuffle key) so base gens for the smoke tasks re-hit.
HELDOUT_SIZES_BIG = {"arc_challenge": 20, "gsm8k": 20, "openbookqa": 20, "mmlu_pro": 18,
                     "math": 14, "reasonif": 8}
# Held-out sample for the train-check (did-it-learn) instructions — fewer needed (large effect).
TRAIN_CHECK_SIZES = {"arc_challenge": 8, "gsm8k": 8, "openbookqa": 8, "mmlu_pro": 8,
                     "math": 5, "reasonif": 3}
# Source-stratified VAL sample for the sweep (~40 tasks/instruction; sufficient for ranking configs
# on raw-uplift + degenerate-rate + accuracy, and a nested subset of larger samples so cached gens
# are reused if the size is later changed).
VAL_SIZES = {"arc_challenge": 8, "gsm8k": 8, "openbookqa": 8, "mmlu_pro": 8,
             "math": 5, "reasonif": 3}

# Eval presets: (task_split, instrs, train_check instrs, none-condition, per-condition sizes).
# 'smoke' = the exact original c32 preliminary eval (kept byte-reproducible).
EVAL_PRESETS = {
    "smoke": {"task_split": "heldout", "instrs": HELDOUT_INSTRS, "train_check": TRAIN_CHECK_INSTRS,
              "none": True, "sizes": HELDOUT_SIZES, "key": "heldout"},
    "val": {"task_split": "val", "instrs": VAL_INSTRS, "train_check": [],
            "none": True, "sizes": VAL_SIZES, "key": "val"},
    "heldout_full": {"task_split": "heldout", "instrs": HELDOUT_INSTRS,
                     "train_check": TRAIN_CHECK_INSTRS, "none": True,
                     "sizes": HELDOUT_SIZES_BIG, "train_check_sizes": TRAIN_CHECK_SIZES,
                     "none_sizes": HELDOUT_SIZES_BIG, "key": "heldout"},
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def subsample_tasks(split: str, sizes: dict, key: str) -> list[dict]:
    """Source-stratified deterministic task sample from `split`. The per-source shuffle key is
    `ft_eval_{key}:{src}` so different presets that share a key produce NESTED samples (the smoke's
    18 held-out tasks are a subset of the deliverable's 100)."""
    import random
    tasks = [json.loads(l) for l in open("data/tasks_all.jsonl") if json.loads(l)["split"] == split]
    by = defaultdict(list)
    for t in tasks:
        by[t["source"]].append(t)
    out = []
    for src, n in sizes.items():
        items = sorted(by[src], key=lambda t: t["task_id"])
        random.Random(f"ft_eval_{key}:{src}").shuffle(items)
        out.extend(items[:n])
    return out


def subsample_heldout(sizes: dict) -> list[dict]:  # back-compat shim (smoke)
    return subsample_tasks("heldout", sizes, "heldout")


def bucket_of(cond: str) -> str:
    if cond == "none":
        return "none"
    instr = I.INSTRUCTIONS[cond]
    if instr.split == "heldout":
        return "cross_category" if cond in CROSS_CATEGORY else "within_category"
    if instr.split == "val":
        return "val_within"
    return "train_check"


def gen_arm(obj, items, weights_id, cache, tracker, assert_cached, batch_override=0, recover=True):
    """Generate for all items on one model arm (per-source batching + truncation recovery).
    items: list of (cond, task, prompt_tokens, source). Returns {(cond, task_id): gen, ...}."""
    gen_by_key, mnt_by_key = {}, {}
    by_src = defaultdict(list)
    for it in items:
        by_src[it[3]].append(it)
    src_order = sorted(by_src, key=lambda s: MNT_BY_SOURCE.get(s, 4096))
    for src in src_order:
        src_items = sorted(by_src[src], key=lambda it: (length_rank(it[1]), it[0]))
        mnt = MNT_BY_SOURCE.get(src, 4096)
        bs = batch_override or BS_BY_SOURCE.get(src, 24)
        prompts = [it[2] for it in src_items]
        gp = {"max_new_tokens": mnt, "temperature": TEMPERATURE, "seed": SEED}
        print(f"  [{weights_id}][gen] {src:14s} items={len(src_items):4d} mnt={mnt} bs={bs}")
        gens = G.generate_many(obj, prompts, gp, cache, tracker, batch_size=bs,
                               assert_cached=assert_cached, weights_id=weights_id)
        for it, g in zip(src_items, gens):
            gen_by_key[(it[0], it[1]["task_id"])] = g
            mnt_by_key[(it[0], it[1]["task_id"])] = mnt
    # truncation recovery
    trunc = [k for k, g in gen_by_key.items()
             if H.parse_channels(g["token_ids"]).truncated and mnt_by_key[k] < RECOVERY_MNT]
    if trunc and recover:
        tasks_by_id = {it[1]["task_id"]: it[1] for it in items}
        rec_items = sorted(trunc, key=lambda k: length_rank(tasks_by_id[k[1]]))
        prompts = []
        for cond, tid in rec_items:
            t = tasks_by_id[tid]
            uc = none_user_content(t) if cond == "none" else instr_user_content(t, I.INSTRUCTIONS[cond])
            prompts.append(H.render_prompt_tokens(uc, reasoning_effort=REASONING_EFFORT))
        gp = {"max_new_tokens": RECOVERY_MNT, "temperature": TEMPERATURE, "seed": SEED}
        print(f"  [{weights_id}][recover] {len(rec_items)} truncated -> mnt={RECOVERY_MNT}")
        gens = G.generate_many(obj, prompts, gp, cache, tracker, batch_size=8,
                               assert_cached=assert_cached, weights_id=weights_id)
        for k, g in zip(rec_items, gens):
            gen_by_key[k] = g
            mnt_by_key[k] = RECOVERY_MNT
    return gen_by_key, mnt_by_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="c32")
    p.add_argument("--preset", default="smoke", choices=list(EVAL_PRESETS))
    p.add_argument("--merged-path", default="")
    p.add_argument("--assert-cached", action="store_true")
    p.add_argument("--no-base", action="store_true", help="skip the base arm (already generated)")
    p.add_argument("--no-recover", action="store_true",
                   help="skip the 8192-token truncation-recovery pass (faster; for the sweep)")
    p.add_argument("--batch-size", type=int, default=0)
    args = p.parse_args()
    merged_path = args.merged_path or json.loads(
        (RESULTS / f"ft_merge_{args.tag}.json").read_text())["merged_path"]
    # Derive the cache weights_id from the SERVED weights (e.g. merged_ft_cdel_bf16 -> ft_cdel_bf16)
    # so the bf16 and mxfp4 serving paths never collide in the generation cache.
    ft_weights_id = "ft_" + os.path.basename(merged_path).replace("merged_ft_", "")
    preset = EVAL_PRESETS[args.preset]
    split, key = preset["task_split"], preset["key"]

    # Build (cond, task, prompt_tokens, source) items. Held-out instructions share ONE task sample
    # (so the task-level cluster bootstrap is valid); train-check + none may use their own samples.
    main_tasks = subsample_tasks(split, preset["sizes"], key)
    tc_tasks = subsample_tasks(split, preset.get("train_check_sizes", preset["sizes"]), key) \
        if preset["train_check"] else []
    none_tasks = subsample_tasks(split, preset.get("none_sizes", preset["sizes"]), key) \
        if preset["none"] else []
    print(f"[{args.preset}] {len(main_tasks)} {split} tasks (instr) / {len(tc_tasks)} (train-check) / "
          f"{len(none_tasks)} (none)")
    cond_tasks = {}
    for cond in preset["instrs"]:
        cond_tasks[cond] = main_tasks
    for cond in preset["train_check"]:
        cond_tasks[cond] = tc_tasks
    if preset["none"]:
        cond_tasks["none"] = none_tasks
    conds = (["none"] if preset["none"] else []) + list(preset["instrs"]) + list(preset["train_check"])

    items = []
    for cond in conds:
        for t in cond_tasks[cond]:
            uc = none_user_content(t) if cond == "none" else instr_user_content(t, I.INSTRUCTIONS[cond])
            toks = H.render_prompt_tokens(uc, reasoning_effort=REASONING_EFFORT)
            items.append((cond, t, toks, t["source"]))
    print(f"{len(conds)} conditions = {len(items)} (task,cond) pairs PER ARM")

    cache, tracker = G._cache_and_cost()
    with modal.enable_output(), G.app.run():
        if args.no_base:
            base_gen, base_mnt = {}, {}
        else:
            base_obj = G.GptOss()
            print("== BASE arm ==")
            base_gen, base_mnt = gen_arm(base_obj, items, "base", cache, tracker, args.assert_cached,
                                         args.batch_size, recover=not args.no_recover)
        ft_obj = G.GptOss(model_path=merged_path)
        print("== FT arm ==")
        ft_gen, ft_mnt = gen_arm(ft_obj, items, ft_weights_id, cache, tracker, args.assert_cached,
                                 args.batch_size, recover=not args.no_recover)

    meta = {
        "ft_tag": args.tag, "preset": args.preset, "merged_path": merged_path,
        "ft_weights_id": ft_weights_id,
        "model": G.MODEL_NAME, "reasoning_effort": REASONING_EFFORT, "temperature": TEMPERATURE,
        "seed": SEED, "max_new_tokens_by_source": MNT_BY_SOURCE, "recovery_mnt": RECOVERY_MNT,
        "sizes": preset["sizes"], "task_split": split,
        "git_hash": git_hash(), "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    suffix = "" if args.preset == "smoke" else f"_{args.preset}"
    out_path = RESULTS / f"ft_eval_{args.tag}{suffix}.jsonl"
    arms = [("ft", ft_gen, ft_mnt)]
    if not args.no_base:
        arms = [("base", base_gen, base_mnt)] + arms
    n = 0
    with open(out_path, "w") as f:
        for arm, gen_by_key, mnt_by_key in arms:
            for cond, t, toks, src in items:
                g = gen_by_key[(cond, t["task_id"])]
                parsed = H.parse_channels(g["token_ids"])
                instr = I.INSTRUCTIONS.get(cond)
                raw_compliant = leaked_final = None
                if instr is not None and instr.scorer is not None:
                    raw_compliant = (bool(instr.scorer(parsed.analysis)) if not parsed.malformed else False)
                    leaked_final = bool(instr.scorer(parsed.final)) if parsed.final.strip() else False
                acc = A.score_accuracy(parsed.final, t["answer"], t["answer_type"], t.get("n_options"))
                vac_floor = instr.vacuous_min_words if instr is not None else 25
                row = {
                    "arm": arm, "task_id": t["task_id"], "source": src, "split": t["split"],
                    "answer_type": t["answer_type"], "condition": cond,
                    "category": (instr.category if instr else "none"),
                    "instr_split": (instr.split if instr else "none"),
                    "bucket": bucket_of(cond),
                    "judge_kind": (instr.judge_kind if instr else None),
                    "scorer_kind": ("programmatic" if (instr and instr.scorer) else
                                    ("judge" if (instr and instr.judge_kind) else "none")),
                    "raw_compliant": raw_compliant, "leaked_final": leaked_final,
                    "malformed": parsed.malformed, "truncated": parsed.truncated,
                    "has_analysis": parsed.has_analysis, "has_final": parsed.has_final,
                    "has_commentary": parsed.has_commentary,
                    "accuracy": acc, "n_gen_tokens": g["n_tokens"],
                    "max_new_tokens": mnt_by_key[(cond, t["task_id"])],
                    "analysis_words": I.word_count(parsed.analysis),
                    "final_words": I.word_count(parsed.final),
                    "vacuous_floor": vac_floor,
                    "is_vacuous_analysis": I.is_vacuous(parsed.analysis, vac_floor),
                    "uppercase_fraction": I.uppercase_fraction(parsed.analysis),
                    "gold": t["answer"], "n_options": t.get("n_options"),
                    "question": t["question"],
                    "analysis": parsed.analysis, "commentary": parsed.commentary,
                    "final": parsed.final, "num_analysis_segments": parsed.num_analysis_segments,
                    "metadata": meta,
                }
                f.write(json.dumps(row) + "\n")
                n += 1
    print(f"\nWrote {n} rows to {out_path}")
    _quick_summary(out_path)
    print(f"\nModal cost this run: ${tracker.run_cost:.4f}")


def _quick_summary(path):
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(lambda: defaultdict(list))
    order = []
    for r in rows:
        if r["condition"] not in order:
            order.append(r["condition"])
        if r["raw_compliant"] is not None:
            by[r["condition"]][r["arm"]].append(r["raw_compliant"])
    print("\n=== quick programmatic raw-compliance (base vs ft) ===")
    print(f"  {'condition':22s} {'bucket':16s} {'base':>6s} {'ft':>6s}")
    for cond in order:
        if cond not in by:
            continue
        b = by[cond]["base"]; ft = by[cond]["ft"]
        bp = (100*sum(b)/len(b)) if b else float("nan")
        fp = (100*sum(ft)/len(ft)) if ft else float("nan")
        print(f"  {cond:22s} {bucket_of(cond):16s} {bp:5.0f}% {fp:5.0f}%")


if __name__ == "__main__":
    main()
