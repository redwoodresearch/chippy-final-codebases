"""Analysis: controlled HELD-OUT steered eval vs base, the random matched-norm null,
sign-reversed, and the FT `cdel` benchmark.

Reads results/steer_eval_<tag>_judged.jsonl (arms: base, real, signrev, rand0..N) and the FT
deliverable judged file (for the FT benchmark + an independent base on the same tasks). Restricts to
the steering held-out task subset (shared across all 9 held-out instructions). Produces, per
instruction + category-macro (cross/within) + aggregate:
  effective_control / raw_compliance / accuracy / meta / genuine / malformed / truncated / degenerate,
the real-vs-base uplift with a TASK-LEVEL CLUSTER bootstrap CI + per-instruction McNemar, and the
RANDOM NULL distribution (mean/spread/max over the >=5 seeds) with an explicit judgment of whether the
real uplift is outside the null and whether it is SIGN-specific (real vs signrev). FAST (no API).

Usage: python analyze_steer_eval.py --tag heldout --ft-tag cdel
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import instructions as I
from analyze_ft_eval import effective, raw_compliant, accuracy
from sft_edit import is_degenerate
from run_ft_eval import HELDOUT_INSTRS, CROSS_CATEGORY
from analyze_ft_deliverable import (cluster_bootstrap_macro, cluster_bootstrap_macro_levels,
                                    mcnemar, paired_metric_vec)

RESULTS = Path("results")


def bucket_of(cond):
    instr = I.INSTRUCTIONS[cond]
    return "cross_category" if cond in CROSS_CATEGORY else "within_category"


def rate(vals):
    vals = [v for v in vals if v is not None]
    return (sum(bool(v) for v in vals) / len(vals)) if vals else None


def cond_metrics(rows):
    return {
        "n": len(rows),
        "raw_compliance": rate([raw_compliant(r) for r in rows]),
        "effective_control": rate([effective(r) for r in rows]),
        "accuracy": rate([accuracy(r) for r in rows]),
        "meta_rate": rate([r.get("meta") for r in rows]),
        "genuine_rate": rate([r.get("genuine") for r in rows]),
        "malformed_rate": rate([r.get("malformed") for r in rows]),
        "truncated_rate": rate([r.get("truncated") for r in rows]),
        "degenerate_rate": rate([is_degenerate(r.get("analysis", "")) for r in rows]),
    }


def load_idx(path):
    idx = {}
    for line in open(path):
        r = json.loads(line)
        idx[(r["arm"], r["condition"], r["task_id"])] = r
    return idx


def _p(x):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:5.1f}%"


def _d(x):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:+5.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="heldout")
    p.add_argument("--ft-tag", default="cdel")
    p.add_argument("--ft-preset", default="heldout_full")
    args = p.parse_args()

    sidx = load_idx(f"results/steer_eval_{args.tag}_judged.jsonl")
    ft_suffix = "" if args.ft_preset == "smoke" else f"_{args.ft_preset}"
    fidx = load_idx(f"results/ft_eval_{args.ft_tag}{ft_suffix}_judged.jsonl")

    rand_arms = sorted({k[0] for k in sidx if k[0].startswith("rand")})
    # The held-out task subset (shared across all 9 held-out instructions in the steered eval).
    tasks = sorted({k[2] for k in sidx if k[0] == "real" and k[1] == HELDOUT_INSTRS[0]})
    print(f"steered arms: base/real/signrev + {rand_arms}; n_tasks={len(tasks)}")

    def rows_for(idx, arm, cond):
        return [idx[(arm, cond, t)] for t in tasks if (arm, cond, t) in idx]

    out = {"tag": args.tag, "ft_tag": args.ft_tag, "n_tasks": len(tasks),
           "rand_arms": rand_arms, "per_instruction": {}, "macros": {}}

    lines = [f"# Single-layer steering — held-out eval — `{args.tag}` vs base / random-null / "
             f"sign-reversed / FT(`{args.ft_tag}`)\n",
             f"n held-out tasks = {len(tasks)} (source-stratified; nested in the FT n=100 sample). "
             "Real direction = pooled TRAIN matched-pair contrast at the VAL-selected (layer, c, "
             "sign). Random null = {} matched-norm seeds. All arms same batching + Opus-meta scoring.\n"
             .format(len(rand_arms)),
             "## Per-instruction effective_control (base → real), with controls\n",
             "real ‖vec‖ matched by the random null; signrev = negated real vector.\n",
             "| instr | bucket | n | base eff | **real eff (Δ, McN p)** | rand null eff "
             "(mean[min,max]) | signrev eff | FT eff | real raw | real malf | real acc | real meta |",
             "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]

    metric_fn = effective
    for cond in HELDOUT_INSTRS:
        b = cond_metrics(rows_for(sidx, "base", cond))
        r = cond_metrics(rows_for(sidx, "real", cond))
        sr = cond_metrics(rows_for(sidx, "signrev", cond))
        ft = cond_metrics(rows_for(fidx, "ft", cond))
        rand_effs = [cond_metrics(rows_for(sidx, ra, cond))["effective_control"] for ra in rand_arms]
        rand_effs = [x if x is not None else 0.0 for x in rand_effs]
        # McNemar real vs base
        bvec = paired_metric_vec(sidx, "base", cond, tasks, metric_fn)
        rvec = paired_metric_vec(sidx, "real", cond, tasks, metric_fn)
        b01, b10, pmc, npair = mcnemar([(bvec[t], rvec[t]) for t in tasks])
        rec = {"bucket": bucket_of(cond), "base": b, "real": r, "signrev": sr, "ft_benchmark": ft,
               "rand_null_eff": rand_effs,
               "rand_null_eff_mean": float(np.mean(rand_effs)), "rand_null_eff_max": float(np.max(rand_effs)),
               "mcnemar_real_vs_base": {"gain": b01, "loss": b10, "p": pmc}}
        out["per_instruction"][cond] = rec
        sig = "***" if pmc < 0.001 else "**" if pmc < 0.01 else "*" if pmc < 0.05 else ""
        deff = (r["effective_control"] or 0) - (b["effective_control"] or 0)
        rn_mean, rn_min, rn_max = np.mean(rand_effs), np.min(rand_effs), np.max(rand_effs)
        lines.append(
            f"| {cond} | {rec['bucket']} | {r['n']} | {_p(b['effective_control'])} | "
            f"**{_p(r['effective_control'])} ({_d(deff)}, {pmc:.3f}{sig})** | "
            f"{_p(rn_mean)}[{_p(rn_min)},{_p(rn_max)}] | {_p(sr['effective_control'])} | "
            f"{_p(ft['effective_control'])} | {_p(r['raw_compliance'])} | {_p(r['malformed_rate'])} | "
            f"{_p(r['accuracy'])} | {_p(r['meta_rate'])} |")

    # ---- macros: cluster bootstrap real-vs-base, + null distribution of the macro ----
    buckets = [("all_heldout", HELDOUT_INSTRS),
               ("within_category", [c for c in HELDOUT_INSTRS if bucket_of(c) == "within_category"]),
               ("cross_category", [c for c in HELDOUT_INSTRS if bucket_of(c) == "cross_category"])]
    lines.append("\n## Aggregate / category-macro effective_control (task-level cluster bootstrap)\n")
    lines.append("Real-vs-base macro uplift with 95% CI (resampling held-out TASKS). Random-null "
                 "macro = each seed's macro uplift over base; we report the null mean/max and whether "
                 "the REAL macro uplift exceeds the MAX of the {} random draws (direction-specificity) "
                 "and whether it exceeds signrev (sign-specificity).\n".format(len(rand_arms)))
    lines.append("| bucket | n_instr | base macro | real macro | **real uplift (95% CI)** | "
                 "rand-null uplift mean[max] | real>null-max? | signrev macro | sign-specific? | FT macro |")
    lines.append("|---|--:|--:|--:|--:|--:|:--:|--:|:--:|--:|")

    def macro_uplift_vs(idx_a, arm_a, idx_b, arm_b, conds):
        """instruction-macro of (arm_a - arm_b) effective rates over conds."""
        ups = []
        for c in conds:
            ma = cond_metrics(rows_for(idx_a, arm_a, c))["effective_control"] or 0
            mb = cond_metrics(rows_for(idx_b, arm_b, c))["effective_control"] or 0
            ups.append(ma - mb)
        return float(np.mean(ups))

    for bname, conds in buckets:
        pit = {c: (paired_metric_vec(sidx, "base", c, tasks, metric_fn),
                   paired_metric_vec(sidx, "real", c, tasks, metric_fn)) for c in conds}
        up = cluster_bootstrap_macro(pit, conds, tasks)
        bl = cluster_bootstrap_macro_levels(pit, conds, tasks, "base")
        rl = cluster_bootstrap_macro_levels(pit, conds, tasks, "ft")  # 'ft' slot = real
        rand_macro_ups = [macro_uplift_vs(sidx, ra, sidx, "base", conds) for ra in rand_arms]
        sr_macro = macro_uplift_vs(sidx, "signrev", sidx, "base", conds)
        ft_macro_level = np.mean([cond_metrics(rows_for(fidx, "ft", c))["effective_control"] or 0
                                  for c in conds])
        beats_null = up["point"] > max(rand_macro_ups)
        sign_spec = up["point"] > sr_macro
        out["macros"][bname] = {
            "real_uplift": up, "base_level": bl, "real_level": rl,
            "rand_null_uplift": rand_macro_ups, "rand_null_uplift_mean": float(np.mean(rand_macro_ups)),
            "rand_null_uplift_max": float(np.max(rand_macro_ups)),
            "signrev_uplift": sr_macro, "ft_macro_level": float(ft_macro_level),
            "beats_null_max": bool(beats_null), "sign_specific": bool(sign_spec)}
        lines.append(
            f"| {bname} | {len(conds)} | {_p(bl['point'])} | {_p(rl['point'])} | "
            f"**{_d(up['point'])} ({_d(up['ci_lo'])},{_d(up['ci_hi'])})** | "
            f"{_d(np.mean(rand_macro_ups))}[{_d(np.max(rand_macro_ups))}] | "
            f"{'YES' if beats_null else 'no'} | {_d(sr_macro)} | "
            f"{'YES' if sign_spec else 'no'} | {_p(ft_macro_level)} |")

    # raw_compliance aggregate (METR-comparable) for real vs base + null
    lines.append("\n## Aggregate raw_compliance (METR-comparable) real vs base + null\n")
    lines.append("| bucket | base | real | rand-null mean[max] | signrev | FT |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for bname, conds in buckets:
        bl = np.mean([cond_metrics(rows_for(sidx, "base", c))["raw_compliance"] or 0 for c in conds])
        rl = np.mean([cond_metrics(rows_for(sidx, "real", c))["raw_compliance"] or 0 for c in conds])
        srl = np.mean([cond_metrics(rows_for(sidx, "signrev", c))["raw_compliance"] or 0 for c in conds])
        ftl = np.mean([cond_metrics(rows_for(fidx, "ft", c))["raw_compliance"] or 0 for c in conds])
        rn = [np.mean([cond_metrics(rows_for(sidx, ra, c))["raw_compliance"] or 0 for c in conds])
              for ra in rand_arms]
        lines.append(f"| {bname} | {_p(bl)} | {_p(rl)} | {_p(np.mean(rn))}[{_p(np.max(rn))}] | "
                     f"{_p(srl)} | {_p(ftl)} |")

    # joint comply(effective)-AND-correct macro (accuracy-aware read, like the FT deliverable)
    def cc(r):
        e = effective(r)
        if e is None:
            return None
        return bool(e) and bool(accuracy(r))
    lines.append("\n## Joint comply(effective)-AND-correct macro (accuracy-aware)\n")
    lines.append("effective_control does NOT require a correct answer; this requires BOTH (the honest "
                 "downstream read, since steering carries an accuracy cost).\n")
    lines.append("| bucket | base | real | Δ | signrev | FT |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for bname, conds in buckets:
        bl = np.mean([rate([cc(r) for r in rows_for(sidx, "base", c)]) or 0 for c in conds])
        rl = np.mean([rate([cc(r) for r in rows_for(sidx, "real", c)]) or 0 for c in conds])
        srl = np.mean([rate([cc(r) for r in rows_for(sidx, "signrev", c)]) or 0 for c in conds])
        ftl = np.mean([rate([cc(r) for r in rows_for(fidx, "ft", c)]) or 0 for c in conds])
        out["macros"].setdefault("comply_correct", {})[bname] = {"base": float(bl), "real": float(rl),
                                                                 "signrev": float(srl), "ft": float(ftl)}
        lines.append(f"| {bname} | {_p(bl)} | {_p(rl)} | {_d(rl-bl)} | {_p(srl)} | {_p(ftl)} |")

    # guardrails: malformed / accuracy / degenerate by arm (aggregate over 9 held-out instr)
    lines.append("\n## Guardrails (aggregate over 9 held-out instructions, micro-pooled)\n")
    lines.append("NOTE: the random-null arms (rand*) are judged ONLY on raw-compliant rows (cost saver "
                 "— non-complying rows can't be effective_control); their **genuine/meta columns are "
                 "NOT comparable** to base/real/signrev (which are fully judged). For the random arms "
                 "read only malformed/truncated/degenerate/accuracy (fully measured); their eff≈0 "
                 "follows from raw_compliance≈0, not from genuine.\n")
    lines.append("| arm | malformed | truncated | degenerate | accuracy | genuine | meta |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for arm, idx in [("base", sidx), ("real", sidx), ("signrev", sidx)] + \
                    [(ra, sidx) for ra in rand_arms] + [("ft", fidx)]:
        pooled = [r for c in HELDOUT_INSTRS for r in rows_for(idx, arm, c)]
        m = cond_metrics(pooled)
        lines.append(f"| {arm} | {_p(m['malformed_rate'])} | {_p(m['truncated_rate'])} | "
                     f"{_p(m['degenerate_rate'])} | {_p(m['accuracy'])} | {_p(m['genuine_rate'])} | "
                     f"{_p(m['meta_rate'])} |")

    # no-instruction default behaviour for the steered arms (garbling check)
    lines.append("\n## No-instruction default behaviour (steered garbling check)\n")
    lines.append("| arm | n | accuracy | malformed | degenerate | aw median |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for arm in ["base", "real", "signrev"] + rand_arms:
        nr = rows_for(sidx, arm, "none")
        if not nr:
            continue
        m = cond_metrics(nr)
        aws = [r["analysis_words"] for r in nr]
        lines.append(f"| {arm} | {len(nr)} | {_p(m['accuracy'])} | {_p(m['malformed_rate'])} | "
                     f"{_p(m['degenerate_rate'])} | {np.median(aws):.0f} |")

    # verdict
    em = out["macros"]
    allh = em["all_heldout"]
    verdict = ("BEATS the random null + sign-specific" if (allh["beats_null_max"] and allh["sign_specific"])
               else "does NOT clearly beat the null/sign controls")
    lines.append(f"\n## Verdict\n\nAll-held-out real effective_control macro uplift = "
                 f"{_d(allh['real_uplift']['point'])} "
                 f"(95% CI {_d(allh['real_uplift']['ci_lo'])},{_d(allh['real_uplift']['ci_hi'])}); "
                 f"random-null macro uplift mean {_d(allh['rand_null_uplift_mean'])} "
                 f"(max {_d(allh['rand_null_uplift_max'])}); signrev macro uplift "
                 f"{_d(allh['signrev_uplift'])}. **Real effect {verdict}.** "
                 f"FT benchmark macro level {_p(allh['ft_macro_level'])}.\n")

    json.dump(out, open(f"results/steer_eval_{args.tag}_analysis.json", "w"), indent=2)
    (RESULTS / f"steer_eval_{args.tag}_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote results/steer_eval_{args.tag}_analysis.md")


if __name__ == "__main__":
    main()
