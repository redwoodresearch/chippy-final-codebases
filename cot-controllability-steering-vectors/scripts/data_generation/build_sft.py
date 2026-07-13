"""Stage B of the edited-reasoning SFT build: build the COMPLIANT SFT targets.

For each (qualified TRAIN source trace x TRAIN instruction): edit the source ``analysis`` into an
instruction-complying target (programmatic or LLM-assisted), set ``final`` = the gold answer in the
stable answer-format, and pass it through every gate (compliant / genuine / not-meta / faithful [LLM
edits] / analysis-concludes-gold / well-formed Harmony round-trip). Accept the first editor attempt
that passes all gates; balance the accepted set source-stratified per instruction.

Output: ``data/sft_edited_reasoning_{tag}.jsonl`` (accepted) + ``results/sft_build_{tag}_log.jsonl``
(every attempt's gate results, for yield/rejection analysis + the raw-trace control build).

Usage:
  python build_sft.py --sources-tag dev  --per-source 6  --oversample 2.0 --tag dev
  python build_sft.py --sources-tag full --per-source 55 --oversample 1.4 --tag full
  python build_sft.py --sources-tag full --tag full --assert-cached   # $0 re-run check
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import answer_scoring as A  # noqa: E402
import harmony_utils as H  # noqa: E402
import instructions as I  # noqa: E402
import judges as J  # noqa: E402
import llm  # noqa: E402
import sft_edit as E  # noqa: E402
import sft_judges as SJ  # noqa: E402
from instruction_meta import (FORCE_CONCLUDES_JUDGE, LLM_EDIT_INSTRUCTIONS,  # noqa: E402
                              PROGRAMMATIC_EDITS, TRAIN_INSTRUCTIONS)
from instructions import word_count  # noqa: E402

DATA = Path("data")
RESULTS = Path("results")
MAX_ATTEMPTS = 3                       # LLM-editor retries (sample_idx 0..MAX_ATTEMPTS-1)
# Candidate source-length cap (words). The SAME cap for ALL instructions so that source length (a
# proxy for task complexity) is NOT confounded with instruction type — otherwise an "edit fits the
# editor output" cap of 600 for LLM edits but 1500 for programmatic would entangle instruction-type
# with complexity and the FT model could learn "stylistic constraint => short reasoning" (flagged by
# multiple LLM reviewers). 600 also fits the editor's 2048-token output and avoids pathologically
# long SFT targets. The very-long-trace tail of math/reasonif is uniformly excluded (documented).
MAX_SOURCE_WORDS = 600


def max_source_words(instruction_id: str) -> int:
    return MAX_SOURCE_WORDS
# Meta gate: use Opus across the board. The validated Haiku meta judge (hand-label precision ~0.93)
# has an ELEVATED false-positive rate on the new COMPLYING/edited distributions — it reads the
# adopted FORM as "following the instruction": ~40-95% FP on questions/second-person, and ~4-16% on
# casing / include_therefore / include_marker_note (verified: 36/36 sampled Haiku meta-rejects were
# non-meta per Opus AND by eye). Opus is reliable in both directions (canary-validated). Using Opus
# here removes a selection bias (Haiku would systematically drop e.g. strong-"Therefore" conclusions)
# and yields meaningful rejects. STRONG handoff recommendation: the downstream FT eval should also use
# Opus meta (or a tightened Haiku prompt), else Haiku would UNDERSTATE effective_control uplift.
DEFAULT_META_MODEL = "claude-opus-4-7"
META_MODEL_BY_INSTR: dict = {}  # (all instructions now use DEFAULT_META_MODEL = Opus)
# The STYLE/LANGUAGE compliance is itself LLM-judged. Haiku's `judge_compliance` is unreliable on
# the new edited distributions — it FALSE-NEGATIVES on valid Spanish (too strict on math/inline-
# English) and is shaky on questions/second-person form — while Opus agrees with by-eye labels. So
# gate style-compliance with Opus. (Programmatic-scorer instructions are unaffected.)
COMPLIANCE_MODEL_BY_INSTR = {"questions": "claude-opus-4-7", "second_person": "claude-opus-4-7",
                             "reason_in_spanish": "claude-opus-4-7"}
# The analysis-concludes-gold BLIND judge is the most critical gate (it protects against silent
# conclusion drift in LLM edits). Haiku sometimes SOLVES the problem itself instead of reporting what
# a question-form trace concludes (canary: a trace concluding "5" was reported as the correct "10");
# Opus reliably reports the trace's own conclusion. So gate concludes-gold with Opus.
CONCLUDES_GATE_MODEL = "claude-opus-4-7"
# Source-faithfulness is the PRIMARY defense against the hardest failure mode: an edit that keeps the
# gold conclusion but silently changes the decisive REASONING (concludes-gold can't catch this). Use
# Opus (not Haiku) so a strong judge audits the strong editor's own LLM edits (avoid weak-judging-
# strong rubber-stamping, flagged by reviewers).
FAITHFUL_GATE_MODEL = "claude-opus-4-7"


def gold_final(answer_type: str, gold: str) -> str:
    """The target `final` = the gold answer in the stable answer-format (correct by construction)."""
    if answer_type == "mcq_letter":
        return str(gold).strip().upper()
    return "\\boxed{" + str(gold).strip() + "}"


def make_edits_programmatic(instruction_id: str, source_analysis: str) -> str:
    return E.PROGRAMMATIC_EDIT_FN[instruction_id](source_analysis)


async def gate_target(instr: I.Instruction, src: dict, target_analysis: str,
                      *, is_llm_edit: bool, assert_cached: bool) -> dict:
    """Run all gates on a candidate target_analysis. Returns a dict of gate results + `accepted`."""
    g = {"compliant": None, "genuine": None, "meta": None, "faithful": None,
         "concludes_gold": None, "concludes_method": None, "concludes_judge_answer": None,
         "final_correct": None, "malformed": None, "truncated": None, "vacuous": None,
         "analysis_words": word_count(target_analysis), "accepted": False, "reject_reason": None}

    if not target_analysis or not target_analysis.strip():
        g["reject_reason"] = "empty_edit"
        return g

    # vacuous floor (length)
    g["vacuous"] = word_count(target_analysis) < instr.vacuous_min_words
    if g["vacuous"]:
        g["reject_reason"] = "vacuous_floor"
        return g

    # degeneracy (defensive): sources are already non-degenerate (Stage A) and programmatic/LLM edits
    # don't loop, but reject any target that is degenerate so a looping target can never slip through.
    if E.is_degenerate(target_analysis):
        g["reject_reason"] = "degenerate"
        return g

    # compliance (programmatic scorer or style judge)
    if instr.scorer is not None:
        g["compliant"] = bool(instr.scorer(target_analysis))
    else:
        comp_model = COMPLIANCE_MODEL_BY_INSTR.get(instr.id, J.JUDGE_MODEL_DEFAULT)
        g["compliant"] = await J.judge_compliance(instr.judge_kind, target_analysis,
                                                  model=comp_model, assert_cached=assert_cached)
    if not g["compliant"]:
        g["reject_reason"] = "not_compliant"
        return g

    # genuine
    g["genuine"] = await J.judge_genuine(src["question"], target_analysis, assert_cached=assert_cached)
    if not g["genuine"]:
        g["reject_reason"] = "not_genuine"
        return g

    # meta (Opus for all instructions; see DEFAULT_META_MODEL rationale)
    meta_model = META_MODEL_BY_INSTR.get(instr.id, DEFAULT_META_MODEL)
    g["meta"] = await J.judge_meta(instr.prompt_text, target_analysis,
                                   required_phrases=instr.required_phrases, model=meta_model,
                                   assert_cached=assert_cached)
    if g["meta"]:
        g["reject_reason"] = "meta"
        return g

    # faithfulness (LLM edits only)
    if is_llm_edit:
        g["faithful"] = await SJ.judge_faithful(src["question"], src["source_analysis"],
                                                target_analysis, model=FAITHFUL_GATE_MODEL,
                                                assert_cached=assert_cached)
        if not g["faithful"]:
            g["reject_reason"] = "not_faithful"
            return g

    # analysis-concludes-gold (blind judge forced for LLM edits)
    cg = await SJ.analysis_concludes_gold(
        src["question"], target_analysis, str(src["gold_answer"]), src["answer_type"],
        src.get("n_options"), force_judge=(instr.id in FORCE_CONCLUDES_JUDGE),
        model=CONCLUDES_GATE_MODEL, assert_cached=assert_cached)
    g["concludes_gold"] = cg["passed"]
    g["concludes_method"] = cg["method"]
    g["concludes_judge_answer"] = cg["judge_answer"]
    if not cg["passed"]:
        g["reject_reason"] = "analysis_not_conclude_gold"
        return g

    # final correctness + well-formed Harmony round-trip
    final = gold_final(src["answer_type"], src["gold_answer"])
    g["final_correct"] = (A.score_accuracy(final, src["gold_answer"], src["answer_type"],
                                           src.get("n_options")) is True)
    uc = I.build_user_content(src["question"], A.build_answer_instruction(src["answer_type"]),
                              instr, mode="cot")
    prompt_toks, comp_toks = H.render_training_example(uc, target_analysis, final)
    parsed = H.parse_channels(comp_toks)
    g["malformed"] = parsed.malformed
    g["truncated"] = parsed.truncated
    rt_ok = (not parsed.malformed and not parsed.truncated and parsed.analysis == target_analysis
             and parsed.final == final)
    if not g["final_correct"]:
        g["reject_reason"] = "final_incorrect"
        return g
    if not rt_ok:
        g["reject_reason"] = "round_trip_failed"
        return g

    g["accepted"] = True
    g["_final"] = final
    g["_uc"] = uc
    g["_prompt_sha"] = hashlib.sha256(json.dumps(prompt_toks).encode()).hexdigest()[:16]
    g["_n_prompt_tokens"] = len(prompt_toks)
    g["_n_completion_tokens"] = len(comp_toks)
    return g


async def build_for_instruction(instr: I.Instruction, candidates: list[dict], per_source: int,
                                *, assert_cached: bool, sem: asyncio.Semaphore) -> tuple[list, list]:
    """Build accepted examples for one instruction. Returns (accepted_examples, attempt_log)."""
    is_llm = instr.id in LLM_EDIT_INSTRUCTIONS
    attempt_log = []

    async def one_candidate(src: dict) -> dict:
        async with sem:
            if not is_llm:
                target = make_edits_programmatic(instr.id, src["source_analysis"])
                g = await gate_target(instr, src, target, is_llm_edit=False,
                                      assert_cached=assert_cached)
                return {"src": src, "target": target, "gate": g, "attempts": 1, "edit_sample": 0}
            # LLM edit with retries
            last_g, last_target = None, None
            for k in range(MAX_ATTEMPTS):
                target = await E.llm_edit(instr.id, src["question"], src["source_analysis"],
                                          sample_idx=k, assert_cached=assert_cached)
                if target is None:
                    last_g = {"accepted": False, "reject_reason": "cannot_edit_or_unsafe",
                              "analysis_words": 0}
                    last_target = None
                    continue
                g = await gate_target(instr, src, target, is_llm_edit=True,
                                      assert_cached=assert_cached)
                last_g, last_target = g, target
                if g["accepted"]:
                    return {"src": src, "target": target, "gate": g, "attempts": k + 1,
                            "edit_sample": k}
            return {"src": src, "target": last_target, "gate": last_g, "attempts": MAX_ATTEMPTS,
                    "edit_sample": None}

    results = await asyncio.gather(*[one_candidate(s) for s in candidates])

    # log every attempt outcome
    for r in results:
        attempt_log.append({
            "instruction_id": instr.id, "task_id": r["src"]["task_id"],
            "source": r["src"]["source"], "attempts": r["attempts"],
            "edit_sample": r["edit_sample"], **{k: v for k, v in r["gate"].items()
                                                if not k.startswith("_")},
        })

    # select balanced accepted set (per source cap), preserving deterministic candidate order
    accepted_by_src = defaultdict(list)
    examples = []
    for r in results:
        g = r["gate"]
        if not g["accepted"]:
            continue
        src = r["src"]
        if len(accepted_by_src[src["source"]]) >= per_source:
            continue
        accepted_by_src[src["source"]].append(src["task_id"])
        examples.append({
            "example_id": f"{instr.id}__{src['task_id']}",
            "task_id": src["task_id"], "source": src["source"], "split": "train",
            "answer_type": src["answer_type"], "n_options": src.get("n_options"),
            "gold_answer": src["gold_answer"],
            "instruction_id": instr.id, "category": instr.category, "instruction_split": "train",
            "prompt_user_content": g["_uc"],
            "target_analysis": r["target"], "target_final": g["_final"],
            "source_analysis": src["source_analysis"],
            "edit_method": (f"llm_edit:{E.EDITOR_MODEL}" if is_llm
                            else f"programmatic:{instr.id}"),
            "edit_meta": {"attempts": r["attempts"], "edit_sample": r["edit_sample"],
                          "editor_model": (E.EDITOR_MODEL if is_llm else None),
                          "meta_judge_model": META_MODEL_BY_INSTR.get(instr.id, DEFAULT_META_MODEL),
                          "compliance_judge_model": (
                              COMPLIANCE_MODEL_BY_INSTR.get(instr.id, J.JUDGE_MODEL_DEFAULT)
                              if instr.judge_kind else None),
                          "faithful_judge_model": (FAITHFUL_GATE_MODEL if is_llm else None),
                          "concludes_gate_model": CONCLUDES_GATE_MODEL},
            "prompt_token_sha256": g["_prompt_sha"],
            "n_prompt_tokens": g["_n_prompt_tokens"],
            "n_completion_tokens": g["_n_completion_tokens"],
            "gates": {k: g[k] for k in ("compliant", "genuine", "meta", "faithful",
                                        "concludes_gold", "concludes_method", "final_correct",
                                        "malformed", "truncated", "analysis_words")},
            "control": False,
        })
    return examples, attempt_log


def select_candidates(sources: list[dict], instruction_id: str, per_source: int,
                      oversample: float) -> list[dict]:
    """Deterministically pick a source-stratified candidate set for one instruction (per-instruction
    seed => different-but-overlapping subsets across instructions; oversampled for rejections)."""
    n_cand = int(round(per_source * oversample))
    max_words = max_source_words(instruction_id)
    by_src = defaultdict(list)
    for s in sources:
        if s.get("qualified") and s.get("source_analysis_words", 0) <= max_words:
            by_src[s["source"]].append(s)
    out = []
    for src, items in by_src.items():
        items = sorted(items, key=lambda t: t["task_id"])
        random.Random(f"cand:{instruction_id}:{src}").shuffle(items)
        out.extend(items[:n_cand])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-cached", action="store_true")
    ap.add_argument("--sources-tag", type=str, default="dev")
    ap.add_argument("--per-source", type=int, default=6, help="accepted target per source per instr")
    ap.add_argument("--oversample", type=float, default=2.0, help="candidate multiplier")
    ap.add_argument("--instructions", type=str, default="all")
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--tag", type=str, default="dev")
    args = ap.parse_args()

    sources = [json.loads(l) for l in open(DATA / f"source_traces_train_{args.sources_tag}.jsonl")]
    instr_ids = (TRAIN_INSTRUCTIONS if args.instructions == "all"
                 else [s.strip() for s in args.instructions.split(",") if s.strip()])
    for iid in instr_ids:
        assert I.INSTRUCTIONS[iid].split == "train", f"{iid} not a train instruction"

    sem = asyncio.Semaphore(args.concurrency)

    async def run_all():
        all_examples, all_log = [], []
        for iid in instr_ids:
            instr = I.INSTRUCTIONS[iid]
            cands = select_candidates(sources, iid, args.per_source, args.oversample)
            exs, log = await build_for_instruction(instr, cands, args.per_source,
                                                   assert_cached=args.assert_cached, sem=sem)
            n_acc = sum(1 for r in log if r["accepted"])
            print(f"[{iid:20s}] cand={len(cands):4d} accepted_attempts={n_acc:4d} "
                  f"kept(balanced)={len(exs):4d}")
            all_examples.extend(exs)
            all_log.extend(log)
        return all_examples, all_log

    examples, attempt_log = asyncio.run(run_all())

    if args.assert_cached:
        # Pure cache-verification run: do NOT overwrite the canonical dataset (the per-source size
        # may differ from the build that produced it). Just confirm $0 and the count.
        print(f"\n[--assert-cached] verified $0; built {len(examples)} examples "
              f"(NOT written — canonical file untouched).")
        print(f"LLM cost this run: ${llm._tracker.run_cost:.4f}")
        return

    out_path = DATA / f"sft_edited_reasoning_{args.tag}.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    log_path = RESULTS / f"sft_build_{args.tag}_log.jsonl"
    with open(log_path, "w") as f:
        for r in attempt_log:
            f.write(json.dumps(r) + "\n")

    # ---- summary ----
    print(f"\nWrote {len(examples)} accepted examples to {out_path}")
    print(f"Wrote {len(attempt_log)} attempt records to {log_path}")
    by_instr = defaultdict(lambda: defaultdict(int))
    for ex in examples:
        by_instr[ex["instruction_id"]][ex["source"]] += 1
    print("\n=== accepted per instruction x source ===")
    srcs = ["arc_challenge", "gsm8k", "math", "mmlu_pro", "openbookqa", "reasonif"]
    print(f"  {'instruction':20s} " + " ".join(f"{s[:6]:>6s}" for s in srcs) + "  total")
    for iid in instr_ids:
        row = by_instr[iid]
        print(f"  {iid:20s} " + " ".join(f"{row.get(s,0):6d}" for s in srcs) +
              f"  {sum(row.values()):5d}")
    # acceptance rates + reject reasons per instruction
    print("\n=== candidate acceptance + reject reasons (per instruction) ===")
    by_i = defaultdict(list)
    for r in attempt_log:
        by_i[r["instruction_id"]].append(r)
    for iid in instr_ids:
        sub = by_i[iid]
        acc = sum(1 for r in sub if r["accepted"])
        reasons = defaultdict(int)
        for r in sub:
            if not r["accepted"]:
                reasons[r["reject_reason"]] += 1
        print(f"  {iid:20s} acc={acc}/{len(sub)} ({100*acc/max(1,len(sub)):.0f}%)  "
              f"rejects={dict(reasons)}")
    print(f"\nLLM cost this run: ${llm._tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
