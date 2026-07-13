"""Stage A of the edited-reasoning SFT build: assemble + QUALIFY the source traces.

For each base-solvable TRAIN task we recover the model's **natural base ``analysis``** (generated with
NO reasoning instruction, answer-format only, medium/greedy/seed-0) from the cached base
generations (re-rendering the exact ``run_base_accuracy`` prompt -> cache hit, $0). We then QUALIFY
each source (desiderata v2 §1): keep it only if the source ``analysis`` is non-empty/well-formed,
**genuine** (``judge_genuine``), and **itself concludes the gold answer** (the concludes-gold gate) —
``base_correct`` alone is not enough (a lucky-correct final on flawed reasoning).

Output: ``data/source_traces_train{tag}.jsonl`` (the qualified source pool the editing stage draws
from; also the raw-trace control's non-complying targets).

Usage:
  python build_source_traces.py --sample-per-source 4 --tag dev     # quick slice
  python build_source_traces.py --full --tag full                   # whole base-solvable train pool
  python build_source_traces.py --full --tag full --assert-cached   # verify $0 re-run (gens+judges)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modal  # noqa: E402

import answer_scoring as A  # noqa: E402
import gpt_oss_infer as G  # noqa: E402
import harmony_utils as H  # noqa: E402
import judges as J  # noqa: E402
import sft_edit as E  # noqa: E402
import sft_judges as SJ  # noqa: E402

RESULTS = Path("results")
DATA = Path("data")
MNT_BY_SOURCE = {"openbookqa": 2048, "arc_challenge": 2048, "gsm8k": 2048,
                 "mmlu_pro": 4096, "math": 4096, "reasonif": 4096}
BS_BY_SOURCE = {"openbookqa": 48, "arc_challenge": 48, "gsm8k": 48,
                "mmlu_pro": 24, "math": 12, "reasonif": 12}


def none_user_content(task: dict) -> str:
    """The no-instruction prompt -- byte-for-byte run_base_accuracy.build_prompt (cache hit)."""
    ai = A.build_answer_instruction(task["answer_type"])
    return f"{task['question'].strip()}\n\n{ai}"


def load_mnt_by_task() -> dict:
    """Map task_id -> the max_new_tokens that produced its STORED base trace (per-source cap, or 8192
    for truncation-recovered items), from results/base_accuracy_full.jsonl, so we hit the exact
    cached generation whose `final` was scored correct."""
    path = RESULTS / "base_accuracy_full.jsonl"
    out = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            out[r["task_id"]] = r.get("max_new_tokens")
    return out


def subsample(tasks, n_per_source: int):
    by = defaultdict(list)
    for t in tasks:
        by[t["source"]].append(t)
    out = []
    for src, items in by.items():
        items = sorted(items, key=lambda t: t["task_id"])
        random.Random(f"srcsample:{src}").shuffle(items)
        out.extend(items[:n_per_source])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-cached", action="store_true")
    ap.add_argument("--sample-per-source", type=int, default=0)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--tag", type=str, default="dev")
    ap.add_argument("--judge-concurrency", type=int, default=50)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(DATA / "tasks_all_scored.jsonl")]
    train = [t for t in tasks if t["split"] == "train" and t.get("base_correct_aug")
             and not t.get("base_truncated")]
    if args.sample_per_source > 0 and not args.full:
        train = subsample(train, args.sample_per_source)
    print(f"Qualifying {len(train)} base-solvable, non-truncated TRAIN source traces (tag={args.tag})")

    mnt_by_task = load_mnt_by_task()
    cache, tracker = G._cache_and_cost()

    # ---- recover natural base analysis (cached generations) ----
    by_src = defaultdict(list)
    for t in train:
        by_src[t["source"]].append(t)
    analysis_by_id, parsed_ok = {}, {}
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        for src in sorted(by_src, key=lambda s: MNT_BY_SOURCE.get(s, 4096)):
            src_tasks = by_src[src]
            prompts = [H.render_prompt_tokens(none_user_content(t), reasoning_effort="medium")
                       for t in src_tasks]
            # group by the exact stored mnt (per-source cap or 8192 recovery) for cache hits
            mnts = [mnt_by_task.get(t["task_id"], MNT_BY_SOURCE.get(src, 4096)) for t in src_tasks]
            for mnt in sorted(set(mnts)):
                idxs = [i for i, m in enumerate(mnts) if m == mnt]
                gp = {"max_new_tokens": mnt, "temperature": 0.0, "seed": 0}
                gens = G.generate_many(obj, [prompts[i] for i in idxs], gp, cache, tracker,
                                       batch_size=BS_BY_SOURCE.get(src, 24),
                                       assert_cached=args.assert_cached)
                for k, i in enumerate(idxs):
                    t = src_tasks[i]
                    parsed = H.parse_channels(gens[k]["token_ids"])
                    analysis_by_id[t["task_id"]] = parsed.analysis
                    parsed_ok[t["task_id"]] = (not parsed.malformed and not parsed.truncated
                                               and parsed.has_analysis and parsed.analysis.strip() != "")
            print(f"[recovered] {src:14s} n={len(src_tasks)}")

    # ---- qualify (genuine + concludes-gold) via async judges ----
    async def qualify():
        sem = asyncio.Semaphore(args.judge_concurrency)

        async def one(t):
            tid = t["task_id"]
            analysis = analysis_by_id.get(tid, "")
            if not parsed_ok.get(tid):
                return tid, {"qualified": False, "source_genuine": None,
                             "source_concludes_gold": None, "source_concludes_method": None,
                             "source_degenerate": None, "reason": "bad_parse_or_empty"}
            # degeneracy gate (no LLM): reject looping/repetitive base traces that would teach the FT
            # model to loop (programmatic edits have no faithfulness backstop). See sft_edit.is_degenerate.
            if E.is_degenerate(analysis):
                return tid, {"qualified": False, "source_genuine": None,
                             "source_concludes_gold": None, "source_concludes_method": None,
                             "source_degenerate": True, "reason": "degenerate_loop"}
            async with sem:
                genuine_task = J.judge_genuine(t["question"], analysis,
                                               assert_cached=args.assert_cached)
                cg = await SJ.analysis_concludes_gold(
                    t["question"], analysis, str(t["answer"]), t["answer_type"],
                    t.get("n_options"), force_judge=False, assert_cached=args.assert_cached)
                genuine = await genuine_task
            qualified = bool(genuine) and bool(cg["passed"])
            return tid, {"qualified": qualified, "source_genuine": genuine,
                         "source_concludes_gold": cg["passed"],
                         "source_concludes_method": cg["method"], "source_degenerate": False,
                         "reason": None if qualified else
                         ("not_genuine" if not genuine else "source_not_conclude_gold")}

        return dict(await asyncio.gather(*[one(t) for t in train]))

    quals = asyncio.run(qualify())

    rows = []
    for t in train:
        tid = t["task_id"]
        q = quals[tid]
        analysis = analysis_by_id.get(tid, "")
        rows.append({
            "task_id": tid, "source": t["source"], "split": t["split"],
            "answer_type": t["answer_type"], "n_options": t.get("n_options"),
            "gold_answer": t["answer"], "question": t["question"],
            "source_analysis": analysis, "source_analysis_words": len(analysis.split()),
            **q,
        })

    out_path = DATA / f"source_traces_train_{args.tag}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_qual = sum(r["qualified"] for r in rows)
    print(f"\nWrote {len(rows)} source rows ({n_qual} qualified, {100*n_qual/max(1,len(rows)):.1f}%) "
          f"to {out_path}")
    by = defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["source"]][0] += 1
        by[r["source"]][1] += int(r["qualified"])
    for s in sorted(by):
        print(f"  {s:14s} {by[s][1]:4d}/{by[s][0]:4d} qualified")
    rsn = defaultdict(int)
    for r in rows:
        if not r["qualified"]:
            rsn[r["reason"]] += 1
    print("  reject reasons:", dict(rsn))
    print(f"\nCost this run: ${tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
