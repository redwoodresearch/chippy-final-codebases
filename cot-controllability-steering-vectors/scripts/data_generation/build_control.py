"""Stage C of the edited-reasoning SFT build: the matched RAW-TRACE control set.

Same prompts as the accepted compliant examples, but the target ``analysis`` is the ORIGINAL natural
base trace that does NOT comply with the instruction (paired with the same correct gold ``final``).
This isolates *generic SFT-on-reasoning-with-instructions-in-context* from *learning to actually
comply* in the FT reproduction.

Design (desiderata v2 §5): for each accepted compliant ``(task, instruction)`` we emit a control row
with ``target_analysis = source_analysis``, BUT ONLY if the source is **non-compliant** for that
instruction (so the contrast is clean). High-base-compliance instructions (esp. ``include_therefore``)
therefore have lower control coverage — reported, not hidden. The source traces are genuine + correct
+ concludes-gold by construction (they passed the Stage-A qualification gate).

Output: ``data/sft_raw_trace_control_{tag}.jsonl`` (same schema as the compliant set, ``control=True``).

Usage: python build_control.py --tag full   [--assert-cached]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import answer_scoring as A
import harmony_utils as H
import instructions as I
import judges as J
from build_sft import COMPLIANCE_MODEL_BY_INSTR, gold_final

DATA = Path("data")


async def source_is_compliant(instr: I.Instruction, source_analysis: str, *,
                              assert_cached: bool = False) -> bool:
    """Whether the natural source trace already satisfies the instruction. Programmatic scorer for
    casing/suppression/inclusion/length; for the style/Spanish JUDGED instructions we MEASURE
    compliance with the same Opus judge the gate uses (base CoT is ~0% compliant, but we verify per
    row rather than assume) so every control target is provably non-compliant."""
    if instr.scorer is not None:
        return bool(instr.scorer(source_analysis))
    model = COMPLIANCE_MODEL_BY_INSTR.get(instr.id, J.JUDGE_MODEL_DEFAULT)
    v = await J.judge_compliance(instr.judge_kind, source_analysis, model=model,
                                 assert_cached=assert_cached)
    # Treat an unparseable judge verdict (None) as "assume compliant -> SKIP from control" so an
    # unverifiable row can never silently land in the control as a (possibly) compliant target.
    return v is not False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()

    compliant = [json.loads(l) for l in open(DATA / f"sft_edited_reasoning_{args.tag}.jsonl")]

    async def all_source_compliance():
        sem = asyncio.Semaphore(50)

        async def one(ex):
            instr = I.INSTRUCTIONS[ex["instruction_id"]]
            async with sem:
                return await source_is_compliant(instr, ex["source_analysis"],
                                                 assert_cached=args.assert_cached)
        return await asyncio.gather(*[one(ex) for ex in compliant])

    src_compliant = asyncio.run(all_source_compliance())

    rows, skipped_src_compliant = [], defaultdict(int)
    for ex, sc in zip(compliant, src_compliant):
        instr = I.INSTRUCTIONS[ex["instruction_id"]]
        src_analysis = ex["source_analysis"]
        if sc:
            skipped_src_compliant[ex["instruction_id"]] += 1
            continue  # source already complies -> not a clean control target
        final = gold_final(ex["answer_type"], ex["gold_answer"])
        # well-formed Harmony round-trip on the control target
        prompt_toks, comp_toks = H.render_training_example(ex["prompt_user_content"], src_analysis,
                                                           final)
        parsed = H.parse_channels(comp_toks)
        rt_ok = (not parsed.malformed and not parsed.truncated
                 and parsed.analysis == src_analysis and parsed.final == final)
        final_correct = (A.score_accuracy(final, ex["gold_answer"], ex["answer_type"],
                                          ex.get("n_options")) is True)
        compliant_now = sc  # measured above; False here (we skipped the source-compliant ones)
        rows.append({
            "example_id": f"control__{ex['instruction_id']}__{ex['task_id']}",
            "task_id": ex["task_id"], "source": ex["source"], "split": "train",
            "answer_type": ex["answer_type"], "n_options": ex.get("n_options"),
            "gold_answer": ex["gold_answer"],
            "instruction_id": ex["instruction_id"], "category": ex["category"],
            "instruction_split": "train",
            "prompt_user_content": ex["prompt_user_content"],
            "target_analysis": src_analysis, "target_final": final,
            "source_analysis": src_analysis,
            "edit_method": "raw_trace",
            "edit_meta": {"paired_compliant_example_id": ex["example_id"]},
            "prompt_token_sha256": hashlib.sha256(json.dumps(prompt_toks).encode()).hexdigest()[:16],
            "n_prompt_tokens": len(prompt_toks), "n_completion_tokens": len(comp_toks),
            "gates": {"compliant": compliant_now, "final_correct": final_correct,
                      "malformed": parsed.malformed, "truncated": parsed.truncated,
                      "round_trip_ok": rt_ok,
                      "analysis_words": len(src_analysis.split()),
                      # genuine + concludes-gold hold by Stage-A qualification (recorded there)
                      "genuine_by_source_qual": True, "concludes_gold_by_source_qual": True},
            "control": True,
        })

    # assert pairing parity vs compliant (control should match prompts; sha must equal)
    by_id = {ex["example_id"]: ex for ex in compliant}
    for r in rows:
        comp = by_id[r["edit_meta"]["paired_compliant_example_id"]]
        assert r["prompt_token_sha256"] == comp["prompt_token_sha256"], "control/compliant prompt mismatch"
        assert r["gates"]["compliant"] is False, "control target must be non-compliant"

    out_path = DATA / f"sft_raw_trace_control_{args.tag}.jsonl"
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} raw-trace control rows to {out_path} "
          f"(from {len(compliant)} compliant examples)")
    by_instr = defaultdict(int)
    for r in rows:
        by_instr[r["instruction_id"]] += 1
    print("\n=== control coverage per instruction (and source-already-compliant skips) ===")
    for iid in sorted(set(ex["instruction_id"] for ex in compliant)):
        n_comp = sum(1 for ex in compliant if ex["instruction_id"] == iid)
        print(f"  {iid:20s} control={by_instr[iid]:4d}/{n_comp:4d} compliant  "
              f"(skipped source-already-compliant: {skipped_src_compliant[iid]})")


if __name__ == "__main__":
    main()
