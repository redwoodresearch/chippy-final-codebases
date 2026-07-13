"""Derive candidate single-layer steering directions from the matched complying-vs-non-
complying activation contrast on MATCHED TRAIN prompts.

For each shared TRAIN (task x instruction) pair we have a COMPLYING target analysis (edited set) and
a NON-COMPLYING target analysis (the natural base trace, raw-trace control set) under the SAME prompt
(the instruction is present in both). We teacher-force ``prompt + assistant(analysis, final)`` through
the BASE model and capture resid_post mean-pooled over the analysis-channel content tokens, per layer.

  v_L (pooled)        = mean_complying_resid_L - mean_noncomplying_resid_L   (over all 12 TRAIN instr)
  v_L (per category)  = same, restricted to that category's TRAIN instructions
  v_L (per instr)     = same, restricted to one instruction (diagnostic)
  resid_norm_L        = mean per-token ||resid|| over all captured analysis tokens (magnitude scale)

The pooled direction is the headline (a single general 'control-CoT' direction tested on held-out
instruction TYPES). Persisted to data/steering_directions.npz + a provenance JSON. Cheap (forward
passes only, per-sequence cached).

Usage:
  python run_derive_directions.py --per-instr 0           # use ALL shared pairs
  python run_derive_directions.py --per-instr 150         # cap per instruction (faster)
  python run_derive_directions.py --assert-cached
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import modal

import gpt_oss_infer as G
import steering_lib as S
import instructions as I


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-instr", type=int, default=0,
                   help="cap pairs per instruction (0 = all shared pairs)")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--assert-cached", action="store_true")
    args = p.parse_args()

    pairs = S.load_matched_pairs(train_only=True)
    if args.per_instr > 0:
        by = defaultdict(list)
        for pr in pairs:
            by[pr["instruction_id"]].append(pr)
        kept = []
        for instr, lst in by.items():
            lst = sorted(lst, key=lambda d: d["task_id"])  # deterministic
            kept.extend(lst[:args.per_instr])
        pairs = kept
    print(f"[derive] {len(pairs)} matched TRAIN pairs "
          f"({len(set(p['instruction_id'] for p in pairs))} instructions)")

    # Build the teacher-forced sequences + analysis spans for complying & non-complying targets.
    seqs, spans, meta_rows = [], [], []
    n_skip = 0
    for pr in pairs:
        for which, (a, f) in [("comp", (pr["comp_analysis"], pr["comp_final"])),
                              ("noncomp", (pr["noncomp_analysis"], pr["noncomp_final"]))]:
            try:
                seq, sp = S.analysis_span(pr["prompt_user_content"], a, f)
            except ValueError:
                n_skip += 1
                continue
            seqs.append(seq)
            spans.append(list(sp))
            meta_rows.append({"which": which, "instruction_id": pr["instruction_id"],
                              "category": pr["category"], "task_id": pr["task_id"]})
    print(f"[derive] {len(seqs)} sequences to capture ({n_skip} skipped); "
          f"layers={S.CAPTURE_LAYERS}")

    cache, tracker = G._cache_and_cost()
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        caps = G.capture_span_residuals(obj, seqs, spans, S.CAPTURE_LAYERS, cache, tracker,
                                        batch_size=args.batch_size, assert_cached=args.assert_cached)

    # Aggregate: per (group, layer) sum of mean-analysis residuals + counts. Groups: pooled, each
    # category, each instruction. resid_norm: mean per-token norm over ALL captured analysis tokens.
    H = len(caps[0][str(S.CAPTURE_LAYERS[0])]["mean"])
    sums = {}        # group -> {L: np.array(H)}  for complying
    sums_nc = {}     # group -> {L: np.array(H)}  for non-complying
    counts = defaultdict(int)      # (group, "comp"/"noncomp") -> n
    norm_sum = defaultdict(float)  # L -> sum of mean_token_norm
    norm_n = defaultdict(int)

    def add(group, which, L, vec):
        tgt = sums if which == "comp" else sums_nc
        tgt.setdefault(group, {})
        if L not in tgt[group]:
            tgt[group][L] = np.zeros(H, dtype=np.float64)
        tgt[group][L] += vec

    for mr, cap in zip(meta_rows, caps):
        groups = ["pooled", f"cat_{mr['category']}", f"instr_{mr['instruction_id']}"]
        # ablation groups (reviewer #6): pooled excluding the Spanish / the whole style category, to
        # test whether the no_word_so/suppression win + Spanish artifact are a language-shift side
        # effect of the style-dominated pooled blend (still single-layer, pooled-from-TRAIN).
        if mr["instruction_id"] != "reason_in_spanish":
            groups.append("pooled_nospanish")
        if mr["category"] != "style":
            groups.append("pooled_nostyle")
        which = mr["which"]
        for g in groups:
            counts[(g, which)] += 1
        for L in S.CAPTURE_LAYERS:
            vec = np.asarray(cap[str(L)]["mean"], dtype=np.float64)
            for g in groups:
                add(g, which, L, vec)
            norm_sum[L] += cap[str(L)]["mean_token_norm"]
            norm_n[L] += 1

    resid_norm = {L: norm_sum[L] / norm_n[L] for L in S.CAPTURE_LAYERS}

    # diff-of-means per group/layer
    out_groups = {}
    diag = {}  # group -> {L: {diff_norm, ratio, cos_to_pooled}}
    all_group_names = sorted(set(sums.keys()) & set(sums_nc.keys()))
    for g in all_group_names:
        out_groups[g] = {}
        diag[g] = {}
        nc_comp = counts[(g, "comp")]
        nc_nc = counts[(g, "noncomp")]
        for L in S.CAPTURE_LAYERS:
            mc = sums[g][L] / nc_comp
            mn = sums_nc[g][L] / nc_nc
            diff = (mc - mn).astype(np.float32)
            out_groups[g][L] = diff
            dn = float(np.linalg.norm(diff))
            diag[g][L] = {"diff_norm": dn, "ratio_to_resid": dn / resid_norm[L],
                          "n_comp": nc_comp, "n_noncomp": nc_nc}

    # cosine of per-category / per-instr directions to the pooled direction (do they agree?)
    for g in all_group_names:
        for L in S.CAPTURE_LAYERS:
            pv = out_groups["pooled"][L]
            gv = out_groups[g][L]
            cos = float(pv @ gv / (np.linalg.norm(pv) * np.linalg.norm(gv) + 1e-9))
            diag[g][L]["cos_to_pooled"] = cos

    meta = {
        "phase": "single_layer_steering", "git_hash": git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contrast": "complying(edited) - non_complying(natural raw-trace) on MATCHED TRAIN prompts",
        "pooling": "mean over analysis-channel content tokens of teacher-forced target",
        "train_instructions": S.TRAIN_INSTRS,
        "capture_layers": S.CAPTURE_LAYERS,
        "n_pairs": len(pairs), "n_sequences": len(seqs), "n_skipped": n_skip,
        "per_instr_cap": args.per_instr, "hidden_size": H,
        "resid_norm_by_layer": resid_norm,
        "counts": {f"{g}|{w}": counts[(g, w)] for (g, w) in counts},
        "diagnostics": {g: {str(L): diag[g][L] for L in S.CAPTURE_LAYERS} for g in all_group_names},
    }
    S.save_directions(out_groups, resid_norm, meta)
    print(f"\n[derive] saved {len(out_groups)} direction groups x {len(S.CAPTURE_LAYERS)} layers "
          f"-> {S.DIRECTIONS_NPZ}")

    # quick console diagnostic (pooled): diff norm vs resid norm by layer
    print("\n=== pooled direction: diff-of-means norm vs residual norm by layer ===")
    print(f"{'layer':>6} {'resid_norm':>11} {'diff_norm':>10} {'ratio':>8}")
    for L in S.CAPTURE_LAYERS:
        d = diag["pooled"][L]
        print(f"{L:>6} {resid_norm[L]:>11.1f} {d['diff_norm']:>10.2f} {d['ratio_to_resid']:>8.4f}")
    print(f"\nModal cost this run: ${tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
