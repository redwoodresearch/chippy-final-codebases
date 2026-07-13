"""Token-level causal attribution: WHICH instruction tokens
do the recruited late heads attend to, and which are NECESSARY for the induced form-logit shift?

  (a) capture_attn with the SUB-SPAN ranges (format_specifier / cot_target / directive / rest, + the
      whole instruction + sink + self_prefix + first_tok) → per-head attention mass on each sub-span,
      base vs gL10-steered, all 24 layers, first reasoning position.
  (b) patch_attn mask_instr on EACH sub-span SEPARATELY (and the complement-of-spec) on the full-
      attention heads → how much of the induced form-logit shift each removes (causal necessity).
  (c) attn_row for a couple recruited heads on a few seqs → eyeball the real attention map.

SLOW (Modal H100); cached. `python run_tok_subspan.py [--assert-cached]`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import modal

import gpt_oss_infer as G
import mech_lib as M
import tok_lib as T

RESULTS = Path("results")
FULL = T.FULL_ATTN_LAYERS
SUBSPANS = ["spec", "cot_target", "directive", "rest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-cached", action="store_true")
    args = ap.parse_args()
    cand_ids, *_ = T.candidate_tokens()
    arms, _ = T.make_arms()
    steering = arms["gL10"]

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

        # ---- (a) per-sub-span attention capture (all 24 layers) ----
        cap_sr = {}
        for i in range(len(seqs)):
            sr = dict(span_ranges[i])
            pl = seq_meta[i]["prompt_len"]; rp = positions[i][0]
            sr["self_prefix"] = (pl, rp + 1)
            sr["first_tok"] = (0, 1)
            cap_sr[i] = sr
        capres = G.capture_attn(obj, seqs, positions, steering, cap_sr, cache, tracker,
                                capture_layers=list(range(24)), micro_batch=2,
                                assert_cached=args.assert_cached)
        _save_attn(capres, seq_meta, cap_sr)

        # ---- (b) per-sub-span causal mask (full-attention heads) ----
        out = {"seq_meta": seq_meta, "cand_ids": cand_ids, "results": {}, "subspans": SUBSPANS}
        # per-sub-span ids per seq (only form conds have sub-spans)
        def sr_for(group):
            return {str(i): {"ids": list(span_ranges[i][group]["ids"])}
                    for i in range(len(seqs)) if group in span_ranges.get(i, {})}
        # complement of spec = cot_target ∪ directive ∪ rest (everything except the format specifier)
        nonspec = {}
        for i in range(len(seqs)):
            if "spec" not in span_ranges.get(i, {}):
                continue
            ids = []
            for g in ["cot_target", "directive", "rest"]:
                ids += list(span_ranges[i][g]["ids"])
            nonspec[str(i)] = {"ids": sorted(ids)}
        mask_specs = {f"mask_{g}": sr_for(g) for g in SUBSPANS}
        mask_specs["mask_nonspec"] = nonspec
        for name, sr in mask_specs.items():
            spec = {"mode": "mask_instr", "layers": FULL, "heads": "all", "span_ranges": sr}
            res = G.patch_attn(obj, seqs, positions, steering, cand_ids, spec, cache, tracker,
                               micro_batch=2, assert_cached=args.assert_cached)
            out["results"][name] = res["results"]
            print(f"[{name}] done")
        # whole-instruction mask for reference (denominator-comparable)
        whole = {str(i): list(span_ranges[i]["instruction"]) for i in range(len(seqs))
                 if "instruction" in span_ranges.get(i, {})}
        res = G.patch_attn(obj, seqs, positions, steering, cand_ids,
                           {"mode": "mask_instr", "layers": FULL, "heads": "all",
                            "span_ranges": whole}, cache, tracker, micro_batch=2,
                           assert_cached=args.assert_cached)
        out["results"]["mask_whole"] = res["results"]
        print("[mask_whole] done")
        json.dump(out, open(RESULTS / "tok_subspan_mask_raw.json", "w"))

        # ---- (c) eyeball attention rows for two recruited late heads on a few bullet seqs ----
        bullet_idx = [i for i in range(len(seqs)) if seq_meta[i]["cond"] == "bullet"][:4]
        if bullet_idx:
            rows = G.attn_row(obj, [seqs[i] for i in bullet_idx], [positions[i] for i in bullet_idx],
                              steering, [[21, 12], [19, 12], [17, 0]], cache, tracker,
                              assert_cached=args.assert_cached)
            json.dump({"rows": rows, "seq_meta": [seq_meta[i] for i in bullet_idx],
                       "span_ranges": {str(k): {g: (span_ranges[bullet_idx[k]][g]
                                                    if g == "instruction" else
                                                    span_ranges[bullet_idx[k]][g])
                                                for g in ["instruction", "spec", "cot_target",
                                                          "directive", "rest"]}
                                       for k in range(len(bullet_idx))}},
                      open(RESULTS / "tok_attn_rows.json", "w"))
            print(f"[attn_row] {len(rows)} rows")

    print("wrote results/tok_subspan_attn.npz + tok_subspan_mask_raw.json + tok_attn_rows.json")
    print(f"\nModal cost this run: ${tracker.run_cost:.4f}")


def _save_attn(res, seq_meta, span_ranges):
    rows = res["results"]
    span_names = ["instruction", "spec", "cot_target", "directive", "rest", "prompt", "self_prefix",
                  "first_tok", "sink"]
    n, nL, nH = len(rows), 24, 64
    arrs = {f"{arm}_{nm}": np.full((n, nL, nH), np.nan, np.float32)
            for arm in ["base", "steer"] for nm in span_names}
    has_instr = np.zeros(n, bool)
    for r, row in enumerate(rows):
        i = row["seq_idx"]
        has_instr[r] = "instruction" in span_ranges.get(i, {})
        for arm in ["base", "steer"]:
            if arm not in row:
                continue
            for L in range(nL):
                e = row[arm].get(str(L))
                if e is None:
                    continue
                for nm in span_names:
                    if nm in e:
                        arrs[f"{arm}_{nm}"][r, L] = e[nm]
    meta = {"conds": [m["cond"] for m in seq_meta], "tasks": [m["task_id"] for m in seq_meta],
            "sources": [m["source"] for m in seq_meta], "has_instr": has_instr.tolist(),
            "span_names": span_names, "full_attention_layers": res["full_attention_layers"]}
    np.savez(RESULTS / "tok_subspan_attn.npz", **arrs)
    json.dump(meta, open(RESULTS / "tok_subspan_attn_meta.json", "w"), indent=2)
    print(f"wrote results/tok_subspan_attn.npz (n={n})")


if __name__ == "__main__":
    main()
