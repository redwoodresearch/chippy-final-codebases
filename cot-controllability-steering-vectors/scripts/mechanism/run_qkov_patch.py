"""QK-pattern vs OV-value patching (the central modulation-vs-
gated-additive test) + sub-block ablation (necessity) + a SURGICAL instruction-attention
knockout (mask_instr) + a gL10spec robustness arm. SLOW (Modal); cached.

  python run_qkov_patch.py [--assert-cached]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

import gpt_oss_infer as G
import mech_lib as M

RESULTS = Path("results")
FULL = M.FULL_ATTN_LAYERS
LATE = [13, 15, 17, 19, 21, 23]
ALL24 = list(range(24))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()
    cand_ids, *_ = M.candidate_tokens()
    arms, _ = M.make_arms()

    cache, tracker = G._cache_and_cost()
    with modal.enable_output(), G.app.run():
        obj = G.GptOss()
        tasks0 = M.subsample_tasks("heldout", M.SIZES, "heldout")
        none_prompts = [M.H.render_prompt_tokens(M.none_user_content(t), reasoning_effort="medium")
                        for t in tasks0]
        gp = {"max_new_tokens": M.BASE_MNT, "temperature": 0.0, "seed": 0}
        ngens = G.generate_many(obj, none_prompts, gp, cache, tracker, batch_size=12,
                                assert_cached=args.assert_cached)
        prefix = {}
        for t, g in zip(tasks0, ngens):
            comp = g["token_ids"]
            cs = M.find_content_start(comp)
            if cs is not None and cs < len(comp):
                prefix[t["task_id"]] = comp[cs:cs + M.N_PREFIX]
        _, seqs, positions, seq_meta, span_ranges = M.build_contexts(prefix)
        # per-seq instruction span (for the surgical mask)
        instr_spans = {str(i): list(span_ranges[i]["instruction"]) for i in range(len(seqs))
                       if "instruction" in span_ranges.get(i, {})}

        specs = {  # gL10 arm
            "pattern_full": {"mode": "qkov", "which": "pattern", "layers": FULL, "heads": "all"},
            "value_full": {"mode": "qkov", "which": "value", "layers": FULL, "heads": "all"},
            "both_full": {"mode": "qkov", "which": "both", "layers": FULL, "heads": "all"},
            "pattern_late": {"mode": "qkov", "which": "pattern", "layers": LATE, "heads": "all"},
            "value_late": {"mode": "qkov", "which": "value", "layers": LATE, "heads": "all"},
            "pattern_all": {"mode": "qkov", "which": "pattern", "layers": ALL24, "heads": "all"},
            "value_all": {"mode": "qkov", "which": "value", "layers": ALL24, "heads": "all"},
            "both_all": {"mode": "qkov", "which": "both", "layers": ALL24, "heads": "all"},
            "ablate_attn_late": {"mode": "ablate", "targets": [[L, "attn"] for L in [17, 19, 21, 23]]},
            "ablate_mlp_late": {"mode": "ablate", "targets": [[L, "mlp"] for L in [17, 18, 19, 22]]},
            "ablate_both_late": {"mode": "ablate",
                                 "targets": [[L, "attn"] for L in [17, 19, 21, 23]]
                                            + [[L, "mlp"] for L in [17, 18, 19, 22]]},
            # surgical: zero recruited heads' attention onto the instruction span (steered run)
            "mask_instr_full": {"mode": "mask_instr", "layers": FULL, "heads": "all",
                                "span_ranges": instr_spans},
            "mask_instr_late": {"mode": "mask_instr", "layers": LATE, "heads": "all",
                                "span_ranges": instr_spans},
        }
        # gL10spec robustness arm (the control-specific direction): just pattern vs value
        spec_specs = {
            "spec_pattern_full": {"mode": "qkov", "which": "pattern", "layers": FULL, "heads": "all"},
            "spec_value_full": {"mode": "qkov", "which": "value", "layers": FULL, "heads": "all"},
        }

        out = {"seq_meta": seq_meta, "cand_ids": cand_ids, "specs": {}, "results": {}}
        for name, spec in specs.items():
            res = G.patch_attn(obj, seqs, positions, arms["gL10"], cand_ids, spec, cache, tracker,
                               micro_batch=2, assert_cached=args.assert_cached)
            out["specs"][name] = spec
            out["results"][name] = res["results"]
            print(f"[{name}] done")  # NB: bit-exact passthrough is checked separately in
            #     scripts_mech_interp/patch_passthrough_check.py (which='checkonly')
        for name, spec in spec_specs.items():
            res = G.patch_attn(obj, seqs, positions, arms["gL10spec"], cand_ids, spec, cache, tracker,
                               micro_batch=2, assert_cached=args.assert_cached)
            out["specs"][name] = spec
            out["results"][name] = res["results"]
        # base+mask control: zero the BASE model's instruction-attention (steering=[]) — isolates that
        # the knockout removes gL10's INCREMENT, not a generic suppression of the base `-` logit.
        bm = G.patch_attn(obj, seqs, positions, [], cand_ids,
                          {"mode": "mask_instr", "layers": FULL, "heads": "all",
                           "span_ranges": instr_spans}, cache, tracker, micro_batch=2,
                          assert_cached=args.assert_cached)
        out["results"]["mask_instr_base"] = bm["results"]

    json.dump(out, open(RESULTS / "mech_qkov_raw.json", "w"))
    print("wrote results/mech_qkov_raw.json")
    print(f"\nModal cost this run: ${tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
