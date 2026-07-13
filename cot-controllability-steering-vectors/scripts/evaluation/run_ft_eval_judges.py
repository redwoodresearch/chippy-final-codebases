"""Run the LLM judges over the FT held-out eval.

CRITICAL judge-model policy (from the SFT-build validation): the Haiku meta judge OVER-FLAGS complying/
styled CoTs as 'meta', which would UNDERSTATE the FT effective_control uplift. So we score:
  * meta            -> OPUS (claude-opus-4-7), for BOTH base and FT arms (apples-to-apples).
  * judged_compliant (style: questions / child_explanation) -> OPUS.
  * genuine         -> HAIKU (validated, 5x cheaper; re-checked on FT outputs separately).

Reads results/ft_eval_<tag>.jsonl, writes results/ft_eval_<tag>_judged.jsonl. All judge calls are
cached + cost-tracked (llm.py). Analysis text head+tail-truncated to bound cost (meta-discussion in
long traces often appears at the very END).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instructions as I  # noqa: E402
import judges as J  # noqa: E402
import llm  # noqa: E402

RESULTS = Path("results")
META_MODEL = "claude-opus-4-7"
COMPLIANCE_MODEL = "claude-opus-4-7"
GENUINE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_HEAD, JUDGE_TAIL = 2200, 1500


def trunc(text: str) -> str:
    t = text or ""
    if len(t) <= JUDGE_HEAD + JUDGE_TAIL:
        return t
    return t[:JUDGE_HEAD] + "\n …[middle truncated for judge]… \n" + t[-JUDGE_TAIL:]


async def judge_row(row: dict, assert_cached: bool, sem: asyncio.Semaphore, terse_set=frozenset()):
    cond = row["condition"]
    instr = I.INSTRUCTIONS.get(cond)
    analysis = trunc(row.get("analysis", ""))
    genuine_model = COMPLIANCE_MODEL if cond in terse_set else GENUINE_MODEL
    out = {}
    async with sem:
        if instr is not None and instr.judge_kind:
            out["judged_compliant"] = (False if row.get("malformed") else
                                       await J.judge_compliance(instr.judge_kind, analysis,
                                                                model=COMPLIANCE_MODEL,
                                                                assert_cached=assert_cached))
        else:
            out["judged_compliant"] = None
        if instr is not None:
            out["meta"] = (None if row.get("malformed") else
                           await J.judge_meta(instr.prompt_text, analysis,
                                              required_phrases=instr.required_phrases,
                                              model=META_MODEL, assert_cached=assert_cached))
        else:
            out["meta"] = None
        out["genuine"] = await J.judge_genuine(row.get("question", ""), analysis,
                                               model=genuine_model, assert_cached=assert_cached)
        out["genuine_model_used"] = genuine_model
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="c32")
    p.add_argument("--preset", default="smoke")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--genuine-opus-terse", action="store_true",
                   help="use Opus (not Haiku) genuine for terse-style instructions (terse_25w, "
                        "under_70w) so very-short real reasoning isn't under-flagged")
    p.add_argument("--assert-cached", action="store_true")
    args = p.parse_args()

    suffix = "" if args.preset == "smoke" else f"_{args.preset}"
    in_path = RESULTS / f"ft_eval_{args.tag}{suffix}.jsonl"
    rows = [json.loads(l) for l in open(in_path)]
    print(f"Loaded {len(rows)} rows from {in_path}")

    # Opus genuine for very-terse-style instructions (Haiku false-negatives on <=25-word real
    # reasoning -> would understate the uplift). terse_25w is the within-bucket
    # driver; under_70w is the val-length analogue.
    terse_set = {"terse_25w", "under_70w"} if args.genuine_opus_terse else set()

    async def run():
        sem = asyncio.Semaphore(args.concurrency)
        return await asyncio.gather(*[judge_row(r, args.assert_cached, sem, terse_set) for r in rows])

    judged = asyncio.run(run())
    for r, j in zip(rows, judged):
        r.update(j)
        r["meta_judge_model"] = META_MODEL
        r["compliance_judge_model"] = COMPLIANCE_MODEL
        r["genuine_judge_model"] = GENUINE_MODEL

    out_path = RESULTS / f"ft_eval_{args.tag}{suffix}_judged.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {out_path}")
    print(f"LLM judge cost this run: ${llm._tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
