"""Optional: a small "plain" / default-behaviour SFT set for the training runs to MIX IN.

Each plain example: prompt = task question + answer-format instruction with **NO reasoning
instruction**; target `analysis` = the natural base trace (unchanged); target `final` = gold. Mixing
a small fraction of these preserves default reasoning when no instruction is given (FT only on
complying-format traces can make a model over-apply formats). Same schema as the SFT set + `plain=True`.

These are pure renders of already-qualified source traces (no LLM calls); the recipe MIX is a
training-run decision — this just makes it trivial.

Usage: python build_plain.py --tag full [--per-source 50]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import answer_scoring as A
import harmony_utils as H
from build_sft import gold_final

DATA = Path("data")


def none_user_content(task: dict) -> str:
    ai = A.build_answer_instruction(task["answer_type"])
    return f"{task['question'].strip()}\n\n{ai}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--per-source", type=int, default=50)
    args = ap.parse_args()

    sources = [json.loads(l) for l in open(DATA / f"source_traces_train_{args.tag}.jsonl")
               if json.loads(l).get("qualified")]
    by_src = defaultdict(list)
    for s in sources:
        by_src[s["source"]].append(s)
    rows = []
    for src, items in by_src.items():
        items = sorted(items, key=lambda t: t["task_id"])
        random.Random(f"plain:{src}").shuffle(items)
        for s in items[:args.per_source]:
            uc = none_user_content(s)
            final = gold_final(s["answer_type"], s["gold_answer"])
            prompt_toks, comp_toks = H.render_training_example(uc, s["source_analysis"], final)
            p = H.parse_channels(comp_toks)
            assert not p.malformed and not p.truncated and p.analysis == s["source_analysis"]
            assert A.score_accuracy(final, s["gold_answer"], s["answer_type"], s.get("n_options")) is True
            rows.append({
                "example_id": f"plain__{s['task_id']}", "task_id": s["task_id"],
                "source": s["source"], "split": "train", "answer_type": s["answer_type"],
                "n_options": s.get("n_options"), "gold_answer": s["gold_answer"],
                "instruction_id": "none", "category": "none", "instruction_split": "train",
                "prompt_user_content": uc, "target_analysis": s["source_analysis"],
                "target_final": final, "source_analysis": s["source_analysis"],
                "edit_method": "plain_none", "edit_meta": {},
                "prompt_token_sha256": hashlib.sha256(json.dumps(prompt_toks).encode()).hexdigest()[:16],
                "n_prompt_tokens": len(prompt_toks), "n_completion_tokens": len(comp_toks),
                "gates": {"genuine_by_source_qual": True, "concludes_gold_by_source_qual": True,
                          "final_correct": True, "malformed": False, "truncated": False,
                          "analysis_words": len(s["source_analysis"].split())},
                "control": False, "plain": True,
            })
    out_path = DATA / f"sft_plain_{args.tag}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    bysrc = defaultdict(int)
    for r in rows:
        bysrc[r["source"]] += 1
    print(f"Wrote {len(rows)} plain (no-instruction) examples to {out_path}; per source: {dict(bysrc)}")


if __name__ == "__main__":
    main()
