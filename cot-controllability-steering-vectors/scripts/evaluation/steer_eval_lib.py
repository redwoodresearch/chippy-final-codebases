"""Shared steered-eval machinery (VAL sweep + held-out eval).

Reuses the FT-eval conventions (per-source max_new_tokens + 8192 truncation recovery,
medium effort / greedy / seed 0, source-stratified task sampling) but adds a ``steering`` argument so
every arm (base, real direction, sign-reversed, random-null) is generated with the SAME batching/gen
params and differs only by the injected vector. Pins batch composition (deterministic length-sorted
within-source batching, identical across arms) so the arms are apples-to-apples despite steered greedy
not being bit-reproducible (the difference between arms is the steering, not the batch composition).
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

import harmony_utils as H
import gpt_oss_infer as G
import answer_scoring as A
import instructions as I
import steering_lib as S
from run_base_accuracy import MNT_BY_SOURCE, BS_BY_SOURCE, length_rank
from run_instruction_baseline import none_user_content, instr_user_content
from run_ft_eval import (REASONING_EFFORT, TEMPERATURE, SEED, RECOVERY_MNT, subsample_tasks,
                         CROSS_CATEGORY)

REASONING_EFFORT = REASONING_EFFORT  # re-export


def bucket_of(cond):
    if cond == "none":
        return "none"
    instr = I.INSTRUCTIONS[cond]
    if instr.split == "heldout":
        return "cross_category" if cond in CROSS_CATEGORY else "within_category"
    if instr.split == "val":
        return "val_within"
    if instr.split == "train":
        return "train_check"
    return "other"


def build_items(conds, cond_tasks, effort=None):
    """conds: ordered list of conditions; cond_tasks: {cond: [task,...]}.
    Returns list of (cond, task, prompt_tokens, source). ``effort`` overrides the default
    reasoning effort (medium) -- used by the reasoning-effort robustness sweep."""
    eff = effort or REASONING_EFFORT
    items = []
    for cond in conds:
        for t in cond_tasks[cond]:
            uc = none_user_content(t) if cond == "none" else instr_user_content(t, I.INSTRUCTIONS[cond])
            toks = H.render_prompt_tokens(uc, reasoning_effort=eff)
            items.append((cond, t, toks, t["source"]))
    return items


def gen_arm(obj, items, weights_id, steering, cache, tracker, assert_cached, *,
            batch_override=0, recover=True, verbose=True, mnt_cap=0, effort=None):
    """Generate one arm (optionally steered) over all items, with per-source batching + truncation
    recovery (recovery is ALSO steered for parity). Returns {(cond, task_id): gen, ...}."""
    gen_by_key, mnt_by_key = {}, {}
    by_src = defaultdict(list)
    for it in items:
        by_src[it[3]].append(it)
    for src in sorted(by_src, key=lambda s: MNT_BY_SOURCE.get(s, 4096)):
        src_items = sorted(by_src[src], key=lambda it: (length_rank(it[1]), it[0]))
        mnt = MNT_BY_SOURCE.get(src, 4096)
        if mnt_cap:
            mnt = min(mnt, mnt_cap)
        bs = batch_override or BS_BY_SOURCE.get(src, 24)
        prompts = [it[2] for it in src_items]
        gp = {"max_new_tokens": mnt, "temperature": TEMPERATURE, "seed": SEED}
        if verbose:
            print(f"  [{weights_id}][gen] {src:14s} items={len(src_items):4d} mnt={mnt} bs={bs}")
        gens = G.generate_many(obj, prompts, gp, cache, tracker, batch_size=bs, steering=steering,
                               assert_cached=assert_cached, weights_id=weights_id)
        for it, g in zip(src_items, gens):
            gen_by_key[(it[0], it[1]["task_id"])] = g
            mnt_by_key[(it[0], it[1]["task_id"])] = mnt
    trunc = [k for k, g in gen_by_key.items()
             if H.parse_channels(g["token_ids"]).truncated and mnt_by_key[k] < RECOVERY_MNT]
    if trunc and recover:
        tasks_by_id = {it[1]["task_id"]: it[1] for it in items}
        rec_items = sorted(trunc, key=lambda k: length_rank(tasks_by_id[k[1]]))
        prompts = []
        for cond, tid in rec_items:
            t = tasks_by_id[tid]
            uc = none_user_content(t) if cond == "none" else instr_user_content(t, I.INSTRUCTIONS[cond])
            prompts.append(H.render_prompt_tokens(uc, reasoning_effort=(effort or REASONING_EFFORT)))
        gp = {"max_new_tokens": RECOVERY_MNT, "temperature": TEMPERATURE, "seed": SEED}
        if verbose:
            print(f"  [{weights_id}][recover] {len(rec_items)} truncated -> mnt={RECOVERY_MNT}")
        gens = G.generate_many(obj, prompts, gp, cache, tracker, batch_size=8, steering=steering,
                               assert_cached=assert_cached, weights_id=weights_id)
        for k, g in zip(rec_items, gens):
            gen_by_key[k] = g
            mnt_by_key[k] = RECOVERY_MNT
    return gen_by_key, mnt_by_key


def row_for(arm, cond, t, src, g, mnt, extra):
    """Build a result row (same schema as run_ft_eval rows + steering metadata)."""
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
        "accuracy": acc, "n_gen_tokens": g["n_tokens"], "max_new_tokens": mnt,
        "analysis_words": I.word_count(parsed.analysis), "final_words": I.word_count(parsed.final),
        "vacuous_floor": vac_floor, "is_vacuous_analysis": I.is_vacuous(parsed.analysis, vac_floor),
        "uppercase_fraction": I.uppercase_fraction(parsed.analysis),
        "gold": t["answer"], "n_options": t.get("n_options"), "question": t["question"],
        "analysis": parsed.analysis, "commentary": parsed.commentary, "final": parsed.final,
        "num_analysis_segments": parsed.num_analysis_segments,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# steering-vector construction (from derived directions)
# ---------------------------------------------------------------------------
def make_steering(groups, resid_norm, direction, layer, c, sign):
    """Build steering=[{layer, vector}] for the named direction group at (layer, c, sign).
    vector = sign * c * resid_norm[layer] * unit(groups[direction][layer]). Returns (steering, norm)."""
    vec = S.steering_vector(groups[direction][layer], resid_norm[layer], c, sign)
    return [{"layer": int(layer), "vector": vec.tolist()}], float(np.linalg.norm(vec))


def make_random_steering(layer, norm, seed):
    """A random Gaussian direction at ``layer`` scaled to ``norm`` (matched-norm null)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(G.HIDDEN_SIZE).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9) * norm
    return [{"layer": int(layer), "vector": v.tolist()}], float(np.linalg.norm(v))


def make_multi_steering(groups, resid_norm, direction, layers, c, sign):
    """Multi-layer diff-of-means steering: apply ``direction`` at EACH layer in ``layers``, each
    scaled to its OWN per-layer residual norm (vector_L = sign*c*||resid||_L*unit(v_L)). Returns
    (steering=[{layer,vector}...], total_norm)."""
    steering = []
    sq = 0.0
    for L in layers:
        vec = S.steering_vector(groups[direction][L], resid_norm[L], c, sign)
        steering.append({"layer": int(L), "vector": vec.tolist()})
        sq += float(np.linalg.norm(vec)) ** 2
    return steering, float(np.sqrt(sq))


def make_random_multi_steering(layers, per_layer_norms, seed):
    """Random matched-norm null for a MULTI-layer arm: an independent Gaussian direction at each
    layer scaled to that arm's per-layer ||vector||. ``per_layer_norms`` aligns with ``layers``."""
    rng = np.random.default_rng(seed)
    steering, sq = [], 0.0
    for L, nrm in zip(layers, per_layer_norms):
        v = rng.standard_normal(G.HIDDEN_SIZE).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9) * nrm
        steering.append({"layer": int(L), "vector": v.tolist()})
        sq += float(np.linalg.norm(v)) ** 2
    return steering, float(np.sqrt(sq))
