"""Shared scaffolding for the BROADENED attention-gating
verification, the token-level sub-span causal attribution, and the mechanism-guided
prompting eval.

Extends `mech_lib` (which it imports as M): same teacher-forced design (a shared base
no-instruction CoT prefix per task; the only difference across conditions is the in-context
instruction), the same candidate-token set, the same gL10/gL10spec arms, the same full-attention
layer list. Adds:
  * a SUBSTANTIALLY LARGER source-stratified held-out task sample (SIZES15; key "heldout" so it is a
    NESTED subset of the n=100/instr Deliverable-#2 sample → gL10/cdel reference numbers are on the
    same task pool);
  * the held-out FORMATTING + casing instructions (bullet/numbered/section_headers/xml_steps/
    initial_caps) + `none` control;
  * a token-level SUB-SPAN decomposition of each in-context format instruction into
    {format_specifier, cot_target, directive, rest} (+ the sink + nearby context as comparison spans),
    located by tokenizing the instruction text and matching substrings, validated by decoding.

The cardinal rule: an attention map is a HYPOTHESIS; every load-bearing claim is confirmed
by a causal edit (sub-span mask / ablation) against the validated behavioral readout (the induced
form-logit shift AND a greedy-generation form spot-check).
"""
from __future__ import annotations

import json
from pathlib import Path

import harmony_utils as H
import instructions as I
import mech_lib as M
from run_instruction_baseline import none_user_content, instr_user_content
from run_ft_eval import subsample_tasks

DATA = Path("data")
LAYER = M.LAYER
FULL_ATTN_LAYERS = M.FULL_ATTN_LAYERS
LATE = [13, 15, 17, 19, 21, 23]   # late full-attention layers (the recruited writers L17/19/21 + 23)

# Broadened, source-stratified held-out task sample. key="heldout" → NESTED subset of the
# Deliverable-#2 n=100/instr sample (subsample_tasks shuffles per source with key "ft_eval_heldout").
# 8/source × 6 = 48 tasks for the teacher-forced attribution (vs the original 12).
SIZES15 = {"arc_challenge": 8, "gsm8k": 8, "openbookqa": 8, "mmlu_pro": 8, "math": 8, "reasonif": 8}

# The held-out FORMATTING + casing instructions (the carriers + flat formatting) + `none` control.
CONDS15 = ["none", "bullet", "numbered", "section_headers", "xml_steps", "initial_caps"]
# the asked single-token form marker per condition (None = no clean single-token marker → casing,
# read qualitatively via the casing-preference / uppercase readout).
FORM_OF_COND15 = {"none": None, "bullet": "hyphen", "numbered": "num_1",
                  "section_headers": "Given", "xml_steps": "lt", "initial_caps": None}
# extra structural-marker tokens (beyond mech_lib.FORM_STRINGS) used by section_headers / xml_steps
EXTRA_FORM_STRINGS = {"Given": "Given", "lt": "<", "Work": "Work", "Check": "Check"}


def candidate_tokens15():
    """mech_lib candidate set + the section_headers/xml_steps structural markers."""
    cand_ids, cand_idx, label2id, groups = M.candidate_tokens()
    extra = M.leading_ids(EXTRA_FORM_STRINGS)
    label2id = {**label2id, **extra}
    cand_ids = sorted(set(label2id.values()))
    cand_idx = {t: i for i, t in enumerate(cand_ids)}
    groups = {**groups, "form": groups["form"] + list(extra.keys())}
    return cand_ids, cand_idx, label2id, groups

N_PREFIX = M.N_PREFIX     # 48 teacher-forced base-CoT content tokens
BASE_MNT = M.BASE_MNT     # 96 base tokens to source the prefix


def tasks15():
    return subsample_tasks("heldout", SIZES15, "heldout")


# ---------------------------------------------------------------------------
# Full instruction-span locator (robust; replaces the earlier suffix-diff which can truncate the span
# when format tokens like "2. , 3." coincide with the answer-instruction suffix).
# ---------------------------------------------------------------------------
def instr_span_full(task, cond):
    """Return (prompt_ids, (start, end)) where [start, end) covers the WHOLE prompt_text of the
    in-context instruction. Locate the START via the longest common prefix with the no-instruction
    prompt; walk FORWARD token-by-token until the decoded text covers prompt_text (avoids the
    suffix-diff truncation). For `none` returns (prompt_ids, None)."""
    if cond == "none":
        p = H.render_prompt_tokens(none_user_content(task), reasoning_effort="medium")
        return p, None
    instr = I.INSTRUCTIONS[cond]
    enc = H.get_encoding()
    p_with = H.render_prompt_tokens(instr_user_content(task, instr), reasoning_effort="medium")
    p_none = H.render_prompt_tokens(none_user_content(task), reasoning_effort="medium")
    n = min(len(p_with), len(p_none))
    pre = 0
    while pre < n and p_with[pre] == p_none[pre]:
        pre += 1
    target = instr.prompt_text.strip()
    txt = ""
    end = pre
    for j in range(pre, len(p_with)):
        txt += enc.decode_utf8([p_with[j]])
        end = j + 1
        if target in txt:
            break
    return p_with, (pre, end)


def token_char_map(ids, start, end):
    """For tokens ids[start:end], return (decoded_text, list of (tok_idx, c0, c1)) where [c0,c1) is
    the character span of token tok_idx within decoded_text (per-token decode concatenation == full
    decode, so this is exact for ASCII instruction text)."""
    enc = H.get_encoding()
    spans = []
    txt = ""
    for j in range(start, end):
        try:
            s = enc.decode_utf8([ids[j]])
        except Exception:
            s = "\ufffd"   # byte-fragment token (non-ASCII multibyte split) -> placeholder; never
            #                matches the ASCII format-specifier substrings, keeps the map self-consistent
        c0 = len(txt)
        txt += s
        spans.append((j, c0, len(txt)))
    return txt, spans


# ---------------------------------------------------------------------------
# Per-instruction sub-span substring definitions. Each token is assigned to the HIGHEST-priority
# group whose substring covers (any of) its characters: spec > cot_target > directive > rest. This
# yields a clean PARTITION of the instruction tokens. The "format specifier" = the literal form
# tokens (marker chars + form-name word); cot_target = references to the model's reasoning; directive
# = the imperative verb(s). Validated by decoding (see make_subspan_inspection.py).
# ---------------------------------------------------------------------------
_COT = ["chain of thought", "step-by-step reasoning", "your reasoning", "reasoning step",
        "each step"]
SUBSPAN_DEFS = {
    "bullet": {
        "spec": ["bulleted list", "bulleted", "'- '", "a hyphen and a space", "hyphen"],
        "cot": _COT,
        "dir": ["write", "Every line", "must start with"],
    },
    "numbered": {
        "spec": ["numbered list", "numbered", "a number followed by a period",
                 "(1. , 2. , 3. , ...)", "1.", "2.", "3.", "number followed by a period"],
        "cot": _COT,
        "dir": ["write", "Every line", "must start with"],
    },
    "section_headers": {
        "spec": ["three labeled sections", "labeled sections", "'Given:'", "'Work:'", "'Check:'",
                 "the label", "Given:", "Work:", "Check:"],
        "cot": _COT,
        "dir": ["structure", "starting on its own line", "each starting"],
    },
    "xml_steps": {
        "spec": ["XML tags", "<step> ... </step>", "<step>", "</step>", "one pair of tags",
                 "pair of tags", "tags"],
        "cot": _COT,
        "dir": ["wrap"],
    },
    "initial_caps": {
        "spec": ["capitalize the first letter of every word", "capitalize", "first letter",
                 "every word"],
        "cot": _COT,
        "dir": ["capitalize"],
    },
}
GROUP_PRIORITY = ["spec", "cot", "dir"]  # rest = complement


def _match_char_set(text, substrings):
    """Return a set of char indices in `text` covered by any of `substrings` (case-sensitive,
    all occurrences)."""
    cset = set()
    for sub in substrings:
        idx = text.find(sub)
        while idx != -1:
            for c in range(idx, idx + len(sub)):
                cset.add(c)
            idx = text.find(sub, idx + 1)
    return cset


def subspans(task, cond):
    """Return (prompt_ids, (start, end), groups) where groups = {"spec","cot_target","directive",
    "rest"} each a sorted list of ABSOLUTE token indices (a partition of [start,end)). cond must be
    a formatting/casing instruction in SUBSPAN_DEFS."""
    p_ids, span = instr_span_full(task, cond)
    if span is None:
        return p_ids, None, None
    start, end = span
    text, tokspans = token_char_map(p_ids, start, end)
    defs = SUBSPAN_DEFS[cond]
    char_by_group = {g: _match_char_set(text, defs[g]) for g in GROUP_PRIORITY}
    groups = {"spec": [], "cot_target": [], "directive": [], "rest": []}
    name_of = {"spec": "spec", "cot": "cot_target", "dir": "directive"}
    for (tok_idx, c0, c1) in tokspans:
        chars = set(range(c0, c1))
        assigned = None
        for g in GROUP_PRIORITY:
            if chars & char_by_group[g]:
                assigned = name_of[g]
                break
        groups[assigned or "rest"].append(tok_idx)
    return p_ids, (start, end), groups


# ---------------------------------------------------------------------------
# Build the teacher-forced contexts on the broadened sample, with FULL instruction span + sub-spans.
# ---------------------------------------------------------------------------
def build_contexts15(prefix_by_task: dict, conds=None, tasks=None):
    """Like mech_lib.build_contexts but on the broadened sample + with the FULL instruction span and
    the sub-span ranges. Returns (tasks, seqs, positions, seq_meta, span_ranges) where span_ranges[i]
    has: instruction [a,b] (contiguous full span), prompt [0,prompt_len], and the sub-spans
    spec/cot_target/directive/rest as {"ids":[...]} (possibly disjoint)."""
    conds = conds or CONDS15
    tasks = tasks if tasks is not None else tasks15()
    hdr = M.analysis_header_ids()
    seqs, positions, seq_meta, span_ranges = [], [], [], {}
    for cond in conds:
        for t in tasks:
            if t["task_id"] not in prefix_by_task:
                continue
            if cond == "none":
                p_ids, _ = instr_span_full(t, "none")
                ispan, groups = None, None
            else:
                p_ids, ispan, groups = subspans(t, cond)
            prefix = list(prefix_by_task[t["task_id"]])
            seq = list(p_ids) + hdr + prefix
            base = len(p_ids) + len(hdr)
            first_pos = base - 1
            if first_pos < 1 or first_pos >= len(seq) - 1:
                continue
            i = len(seqs)
            seqs.append(seq)
            positions.append([first_pos])
            seq_meta.append({"cond": cond, "task_id": t["task_id"], "first_pos": first_pos,
                             "prompt_len": len(p_ids), "source": t["source"]})
            sr = {"prompt": (0, len(p_ids))}
            if ispan is not None:
                sr["instruction"] = (int(ispan[0]), int(ispan[1]))
                for g in ("spec", "cot_target", "directive", "rest"):
                    sr[g] = {"ids": [int(x) for x in groups[g]]}
            span_ranges[i] = sr
    return tasks, seqs, positions, seq_meta, span_ranges


def find_substring_tokens(prompt_ids, substrings):
    """Return sorted token indices in `prompt_ids` whose decoded characters fall inside any occurrence
    of any of `substrings` (case-sensitive). Used to locate the FORMAT-SPECIFIER tokens ANYWHERE in a
    rendered prompt variant (the original instruction's specifier + a restated end-reminder copy)."""
    text, tokspans = token_char_map(prompt_ids, 0, len(prompt_ids))
    cset = _match_char_set(text, substrings)
    return sorted(j for (j, c0, c1) in tokspans if set(range(c0, c1)) & cset)


# Key specifier substrings for held-out instructions NOT in SUBSPAN_DEFS (so the specifier can be
# located in BOTH the plain instruction and the restated reminder, which have different surrounds).
SPEC_LOCATE = {
    "no_word_so": ['the word "so"', '"so"'],
    "terse_25w": ["at most 25 words", "25 words", "extremely terse"],
    "include_exactly_twice": ['the word "hence"', '"hence"', "exactly twice", "twice"],
    "child_explanation": ["young child", "very simple words", "simple words"],
}


def spec_substrings(cond):
    """Specifier substrings for locating the format specifier in any prompt variant (the SUBSPAN_DEFS
    spec list, the prompt-variant reminder phrase, and per-instruction KEY fragments)."""
    import prompt_variants as PV
    subs = list(SUBSPAN_DEFS.get(cond, {}).get("spec", []))
    subs += SPEC_LOCATE.get(cond, [])
    if cond in PV.REMIND:
        subs.append(PV.REMIND[cond])
    return subs


def candidate_tokens():
    return candidate_tokens15()


def make_arms():
    return M.make_arms()
