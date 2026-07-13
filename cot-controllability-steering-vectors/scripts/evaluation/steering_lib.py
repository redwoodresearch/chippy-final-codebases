"""Shared helpers for single-layer steering.

Contains:
  * ``analysis_span`` — build the teacher-forced token sequence (prompt + assistant completion) and
    the index span of the ANALYSIS-channel content tokens (what we mean-pool resid_post over to
    derive a steering direction). Reuses the canonical Harmony renderer/parser so the captured
    positions are exactly the analysis text tokens.
  * ``load_matched_pairs`` — load the matched complying-vs-non-complying SFT pairs (shared TRAIN
    task x instruction prompts; compliant target from the edited set, non-complying target from the
    raw-trace control set), restricted to TRAIN instructions.
  * direction (de)serialization to ``data/steering_directions.npz`` + a provenance JSON.

Direction convention: ``v_L = mean_complying_resid_L - mean_noncomplying_resid_L`` (per layer,
raw diff-of-means over analysis-token mean-pooled residuals). At apply time the vector is
unit-normalized and scaled to ``c * mean||resid||`` at that layer (see run_steer_*).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import harmony_utils as H
import instructions as I

REASONING_EFFORT = "medium"
DATA = Path("data")
COMPLIANT_PATH = DATA / "sft_edited_reasoning_full.jsonl"
CONTROL_PATH = DATA / "sft_raw_trace_control_full.jsonl"

# Capture these layers' resid_post (covers early/mid/late depth; the residual norm grows ~200x
# across depth so we sample a spread for the layer sweep). gpt-oss-20b has 24 layers (0..23).
CAPTURE_LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23]

TRAIN_INSTRS = [i.id for i in I.INSTRUCTIONS.values() if i.split == "train"]


def analysis_span(prompt_user_content: str, target_analysis: str, target_final: str):
    """Return (sequence_token_ids, (span_start, span_end)) for teacher-forcing.

    sequence = render_prompt_tokens(user) + render_assistant_completion(analysis, final) truncated
    right after the analysis content (the final tokens don't affect the causal analysis residuals).
    span = the [start, end) indices of the analysis CONTENT tokens within ``sequence``.
    Raises ValueError if the Harmony structure can't be located (caller should skip the example).
    """
    prompt = H.render_prompt_tokens(prompt_user_content, reasoning_effort=REASONING_EFFORT)
    completion = H.render_assistant_completion(target_analysis, target_final)
    # completion = <|channel|> analysis <|message|> {analysis} <|end|> <|start|> assistant ...
    try:
        msg_idx = completion.index(H.TOK_MESSAGE)  # first <|message|> = end of "analysis" header
    except ValueError as e:
        raise ValueError("no <|message|> in completion") from e
    try:
        end_idx = completion.index(H.TOK_END, msg_idx + 1)  # first <|end|> after analysis content
    except ValueError as e:
        raise ValueError("no <|end|> after analysis content") from e
    if end_idx <= msg_idx + 1:
        raise ValueError("empty analysis span")
    seq = prompt + completion[:end_idx]  # keep up to (not incl.) <|end|> -> last tok = last analysis tok
    span_start = len(prompt) + msg_idx + 1
    span_end = len(prompt) + end_idx
    return seq, (span_start, span_end)


def load_matched_pairs(train_only: bool = True):
    """Load matched (compliant, non-complying) SFT pairs keyed by (task_id, instruction_id).

    Returns a list of dicts: {task_id, instruction_id, category, prompt_user_content,
    comp_analysis, comp_final, noncomp_analysis, noncomp_final}. Only pairs present in BOTH the
    compliant set and the matched raw-trace control set (so prompt is identical and the only
    difference is target compliance). Restricted to TRAIN instructions when train_only.
    """
    comp = {}
    for line in open(COMPLIANT_PATH):
        d = json.loads(line)
        if train_only and d["instruction_split"] != "train":
            continue
        comp[(d["task_id"], d["instruction_id"])] = d
    ctrl = {}
    for line in open(CONTROL_PATH):
        d = json.loads(line)
        if train_only and d["instruction_split"] != "train":
            continue
        ctrl[(d["task_id"], d["instruction_id"])] = d
    pairs = []
    for k in sorted(comp.keys() & ctrl.keys()):
        c, n = comp[k], ctrl[k]
        pairs.append({
            "task_id": k[0], "instruction_id": k[1], "category": c["category"],
            "source": c["source"],
            "prompt_user_content": c["prompt_user_content"],
            "comp_analysis": c["target_analysis"], "comp_final": c["target_final"],
            "noncomp_analysis": n["target_analysis"], "noncomp_final": n["target_final"],
        })
    return pairs


# ---------------------------------------------------------------------------
# direction persistence
# ---------------------------------------------------------------------------
DIRECTIONS_NPZ = DATA / "steering_directions.npz"
DIRECTIONS_META = DATA / "steering_directions_meta.json"


def save_directions(groups: dict, resid_norm: dict, meta: dict):
    """groups[name][L] = vec(2880) for each direction group (e.g. 'pooled', 'cat_casing',
    'instr_all_caps'); resid_norm[L] = float (per-layer typical ||resid|| over analysis tokens)."""
    arrays = {}
    for name, d in groups.items():
        for L, v in d.items():
            arrays[f"{name}@@{L}"] = np.asarray(v, dtype=np.float32)
    arrays["resid_norm_layers"] = np.asarray(sorted(resid_norm.keys()), dtype=np.int32)
    arrays["resid_norm_vals"] = np.asarray([resid_norm[L] for L in sorted(resid_norm.keys())],
                                           dtype=np.float32)
    os.makedirs(DATA, exist_ok=True)
    np.savez(DIRECTIONS_NPZ, **arrays)
    json.dump(meta, open(DIRECTIONS_META, "w"), indent=2)


def load_directions():
    """Returns (groups{name:{L:vec}}, resid_norm{L:float}, meta). The pooled single-general
    direction is groups['pooled']; per-category is groups['cat_<category>']."""
    z = np.load(DIRECTIONS_NPZ)
    groups = {}
    for key in z.files:
        if key in ("resid_norm_layers", "resid_norm_vals"):
            continue
        name, L = key.rsplit("@@", 1)
        groups.setdefault(name, {})[int(L)] = z[key]
    resid_norm = {int(L): float(v) for L, v in
                  zip(z["resid_norm_layers"], z["resid_norm_vals"])}
    meta = json.load(open(DIRECTIONS_META)) if DIRECTIONS_META.exists() else {}
    return groups, resid_norm, meta


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def steering_vector(direction_vec, resid_norm_L: float, c: float, sign: int = 1):
    """Build the steering vector to add: sign * c * ||resid||_L * unit(direction)."""
    return (sign * c * resid_norm_L * unit(direction_vec)).astype(np.float32)
