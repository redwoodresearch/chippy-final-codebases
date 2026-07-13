"""Shared scaffolding for mechanistic attribution of gL10.

Builds the teacher-forced contexts (the SAME design as run_induced_shift: a shared
base no-instruction CoT prefix per task, so the ONLY difference across conditions is the prompt
instruction) and the candidate-token set (form markers + meta/persona tokens) that all the downstream
analyses attribute. Anchors all mechanistic work to the VALIDATED behavioral readout
(the induced conditional form-logit shift + meta-suppression), not a projection.

Key gpt-oss facts used downstream (confirmed on the Modal image, transformers 4.57.1):
  * 24 layers; EVEN layers = sliding(128) attention, ODD layers (1,3,…,23) = FULL attention — only
    the 12 full-attention layers (+ the per-head attention sink) can attend to the far-back
    instruction span at the (late) reasoning position.
  * 64 heads, 8 KV heads (GQA grp 8), head_dim 64; 32 experts, top-4; rms_norm_eps 1e-5.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import grad_steer_lib as GSL
import harmony_utils as H
import instructions as I
from run_instruction_baseline import none_user_content, instr_user_content
from run_ft_eval import subsample_tasks

DATA = Path("data")
LAYER = 10
FULL_ATTN_LAYERS = [L for L in range(24) if L % 2 == 1]   # odd layers = full attention
SLIDING_LAYERS = [L for L in range(24) if L % 2 == 0]

# The form/meta conditions to attribute. bullet/numbered have a clean single-token form marker;
# initial_caps is casing (handled qualitatively via Title-Case content tokens); none = the
# conditionality control (form markers must stay flat).
CONDS = ["none", "bullet", "numbered", "initial_caps", "terse_25w"]
# 2 tasks per source (12 tasks) — deterministic, source-stratified, a superset-nested sample of the
# earlier 1/source set (same shuffle key) so cached base generations re-hit.
SIZES = {"arc_challenge": 2, "gsm8k": 2, "openbookqa": 2, "mmlu_pro": 2, "math": 2, "reasonif": 2}
N_PREFIX = 48          # teacher-force this many base-CoT content tokens
BASE_MNT = 96          # generate this many base tokens to source the prefix

# Form / structure tokens (decoded label -> the leading token id of its encoding).
FORM_STRINGS = {
    "hyphen": "-", "hyphen_sp": "- ", "newline": "\n", "dbl_newline": "\n\n",
    "num_1": "1", "num_1.": "1.", "bullet_dot": "• ", "gt": ">", "star": "*", "hash": "#",
}
# Meta / persona tokens gL10 suppresses everywhere (the second readout to attribute).
META_STRINGS = {
    "user": "user", "_user": " user", "User": "User", "_User": " User",
    "task": "task", "_task": " task", "Assistant": "Assistant", "_assistant": " assistant",
    "We": " We", "The": " The",
}
# A few neutral content/continuation tokens for context.
CONTENT_STRINGS = {"period": ".", "space": " ", "So": " So", "Therefore": " Therefore",
                   "First": " First", "Let": " Let"}

# the asked form-marker token (label) per condition (None = no single-token form marker)
FORM_OF_COND = {"bullet": "hyphen", "numbered": "num_1", "none": None,
                "initial_caps": None, "terse_25w": None}


def leading_ids(strings: dict) -> dict:
    enc = H.get_encoding()
    out = {}
    for label, s in strings.items():
        ids = enc.encode(s, allowed_special=set())
        if ids:
            out[label] = ids[0]
    return out


def candidate_tokens():
    """Return (cand_ids sorted, cand_idx {id->row}, label2id {label->id}, groups{group->[labels]})."""
    form = leading_ids(FORM_STRINGS)
    meta = leading_ids(META_STRINGS)
    content = leading_ids(CONTENT_STRINGS)
    label2id = {**form, **meta, **content}
    cand_ids = sorted(set(label2id.values()))
    cand_idx = {t: i for i, t in enumerate(cand_ids)}
    groups = {"form": list(form.keys()), "meta": list(meta.keys()), "content": list(content.keys())}
    return cand_ids, cand_idx, label2id, groups


def tok_repr(tid: int) -> str:
    enc = H.get_encoding()
    try:
        s = enc.decode_utf8([int(tid)])
        return repr(s) if s != "" else f"⟨{tid}:∅⟩"
    except Exception:
        return f"⟨{tid}:bytefrag⟩"


def find_content_start(completion_ids):
    """Index in the completion of the first ANALYSIS-content token (after the analysis header)."""
    try:
        m = completion_ids.index(H.TOK_MESSAGE)
        return m + 1
    except ValueError:
        return None


def analysis_header_ids():
    return [H.TOK_CHANNEL] + H.get_encoding().encode("analysis", allowed_special=set()) + [H.TOK_MESSAGE]


def instruction_span(task, cond):
    """Return (prompt_ids, (instr_start, instr_end)) — the token span of the in-context format
    instruction text within the rendered prompt. We locate the instruction by rendering the prompt
    WITH and WITHOUT the instruction and diffing the user-content region. For `none` there is no
    instruction span (returns (prompt_ids, None))."""
    if cond == "none":
        p = H.render_prompt_tokens(none_user_content(task), reasoning_effort="medium")
        return p, None
    instr = I.INSTRUCTIONS[cond]
    uc_with = instr_user_content(task, instr)
    p_with = H.render_prompt_tokens(uc_with, reasoning_effort="medium")
    p_none = H.render_prompt_tokens(none_user_content(task), reasoning_effort="medium")
    # find the longest common prefix and suffix between the two token lists; the instruction tokens
    # are the part of p_with NOT in that shared prefix/suffix.
    a, b = p_with, p_none
    n = min(len(a), len(b))
    pre = 0
    while pre < n and a[pre] == b[pre]:
        pre += 1
    suf = 0
    while suf < n - pre and a[len(a) - 1 - suf] == b[len(b) - 1 - suf]:
        suf += 1
    start, end = pre, len(a) - suf
    if end <= start:  # degenerate; fall back to None
        return p_with, None
    return p_with, (start, end)


def build_contexts(prefix_by_task: dict, conds=None):
    """Build the teacher-forced sequences. ``prefix_by_task`` maps task_id -> the shared base
    none-CoT content-token prefix (sourced once via generate). Returns
    (tasks, seqs, positions, seq_meta, span_ranges):
      * seqs[i]      = render(prompt for cond) + analysis header + shared base CoT prefix
      * positions[i] = [first_reasoning_pos]   (logits[pos] predict the first reasoning token —
                       the clean conditional test; gL10 raises the asked marker there)
      * seq_meta[i]  = {cond, task_id, first_pos}
      * span_ranges  = {i: {"instruction": [s,e], "sink": [0,1], "prompt": [0, prompt_len]}}
    """
    conds = conds or CONDS
    tasks = subsample_tasks("heldout", SIZES, "heldout")
    hdr = analysis_header_ids()
    seqs, positions, seq_meta, span_ranges = [], [], [], {}
    for cond in conds:
        for t in tasks:
            if t["task_id"] not in prefix_by_task:
                continue
            p_ids, ispan = instruction_span(t, cond)
            prefix = list(prefix_by_task[t["task_id"]])
            seq = list(p_ids) + hdr + prefix
            base = len(p_ids) + len(hdr)        # index of first reasoning content token
            first_pos = base - 1                 # logits[first_pos] predict the first content token
            if first_pos < 1 or first_pos >= len(seq) - 1:
                continue
            i = len(seqs)
            seqs.append(seq)
            positions.append([first_pos])
            seq_meta.append({"cond": cond, "task_id": t["task_id"], "first_pos": first_pos,
                             "prompt_len": len(p_ids)})
            sr = {"prompt": (0, len(p_ids))}
            if ispan is not None:
                sr["instruction"] = ispan
            span_ranges[i] = sr
    return tasks, seqs, positions, seq_meta, span_ranges


def make_arms():
    """The steering arms to attribute: gL10 (primary) + the gL10-control-specific direction (a
    cleaner target = gL10 minus its component shared with the inert generic-SFT twin gL10ctrl,
    rescaled to ‖gL10‖)."""
    steering, vnorm, _, _ = GSL.make_steering("gL10")
    g10v = np.asarray(GSL.load_vectors("gL10")[1][0], np.float64)
    gctrl = np.asarray(GSL.load_vectors("gL10ctrl")[1][0], np.float64)
    uctrl = gctrl / np.linalg.norm(gctrl)
    g_spec = g10v - (g10v @ uctrl) * uctrl
    g_spec = g_spec / np.linalg.norm(g_spec) * np.linalg.norm(g10v)
    spec_steering = [{"layer": LAYER, "vector": g_spec.tolist()}]
    return {"gL10": steering, "gL10spec": spec_steering}, vnorm
