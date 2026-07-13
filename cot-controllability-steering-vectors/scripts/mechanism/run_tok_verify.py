"""BROADEN the original onset attention-gating verdict.

Re-runs the QK-pattern-vs-OV-value patch + the surgical attention-to-instruction knockout
(`mask_instr`) on a SUBSTANTIALLY LARGER source-stratified held-out task sample (SIZES15 = 8/source =
48 tasks vs the original 12) × the held-out FORMATTING + casing instructions (bullet/numbered/
section_headers/xml_steps/initial_caps) + `none` (conditionality control). Arms: gL10 (primary),
gL10spec (control-specific direction), gL10_s1 (seed sibling) — each with {pattern_full, value_full,
mask_instr_full}; + a base+mask control (zero the BASE model's instruction-attention → isolates
gL10's induced INCREMENT). Per-seq logits on the candidate tokens are saved so analyze_tok_verify.py
can report per-instruction + per-task cluster-bootstrap CIs.

SLOW (Modal H100); cached. `python run_tok_verify.py [--assert-cached]`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

import gpt_oss_infer as G
import grad_steer_lib as GS
import mech_lib as M
import tok_lib as T

RESULTS = Path("results")
FULL = T.FULL_ATTN_LAYERS


def gl10_seed_arm(tag):
    steering, _, _, _ = GS.make_steering(tag)
    return steering


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-cached", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (1/source, gL10 only)")
    args = ap.parse_args()
    cand_ids, *_ = T.candidate_tokens()
    arms, _ = T.make_arms()                      # gL10 + gL10spec
    arms["gL10_s1"] = gl10_seed_arm("gL10_s1")   # seed sibling
    if args.smoke:
        T.SIZES15 = {s: 1 for s in T.SIZES15}
        T.CONDS15 = ["none", "bullet", "numbered"]
        arms = {"gL10": arms["gL10"]}

    cache, tracker = G._cache_and_cost()
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        tasks0 = T.tasks15()
        none_prompts = [M.H.render_prompt_tokens(M.none_user_content(t), reasoning_effort="medium")
                        for t in tasks0]
        gp = {"max_new_tokens": T.BASE_MNT, "temperature": 0.0, "seed": 0}
        ngens = G.generate_many(obj, none_prompts, gp, cache, tracker, batch_size=24,
                                assert_cached=args.assert_cached)
        prefix = {}
        for t, g in zip(tasks0, ngens):
            comp = g["token_ids"]
            cs = M.find_content_start(comp)
            if cs is not None and cs < len(comp):
                prefix[t["task_id"]] = comp[cs:cs + T.N_PREFIX]
        _, seqs, positions, seq_meta, span_ranges = T.build_contexts15(prefix)
        instr_spans = {str(i): list(span_ranges[i]["instruction"]) for i in range(len(seqs))
                       if "instruction" in span_ranges.get(i, {})}
        print(f"[contexts] {len(seqs)} seqs ({len(prefix)} tasks x {len(T.CONDS15)} conds)")

        specs = {
            "pattern_full": {"mode": "qkov", "which": "pattern", "layers": FULL, "heads": "all"},
            "value_full": {"mode": "qkov", "which": "value", "layers": FULL, "heads": "all"},
            "mask_instr_full": {"mode": "mask_instr", "layers": FULL, "heads": "all",
                                "span_ranges": instr_spans},
        }
        out = {"seq_meta": seq_meta, "cand_ids": cand_ids, "specs": {}, "results": {},
               "sizes": T.SIZES15, "conds": T.CONDS15}
        for arm_name, steering in arms.items():
            for sname, spec in specs.items():
                res = G.patch_attn(obj, seqs, positions, steering, cand_ids, spec, cache, tracker,
                                   micro_batch=2, assert_cached=args.assert_cached)
                key = f"{arm_name}__{sname}"
                out["specs"][key] = {k: v for k, v in spec.items() if k != "span_ranges"}
                out["results"][key] = res["results"]
                print(f"[{key}] done (passthrough_err={res.get('passthrough_err')})")
        # base+mask control (steering=[] → zero the BASE model's instruction-attention)
        bm = G.patch_attn(obj, seqs, positions, [], cand_ids,
                          {"mode": "mask_instr", "layers": FULL, "heads": "all",
                           "span_ranges": instr_spans}, cache, tracker, micro_batch=2,
                          assert_cached=args.assert_cached)
        out["results"]["base__mask_instr_full"] = bm["results"]
        print("[base__mask_instr_full] done")

    json.dump(out, open(RESULTS / "tok_verify_raw.json", "w"))
    print("wrote results/tok_verify_raw.json")
    print(f"\nModal cost this run: ${tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
