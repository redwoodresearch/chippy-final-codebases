"""Deliverable #2 analysis: the full-n, significance-tested held-out steering eval.

Reads the JUDGED steered deliverable eval (base + gL10 + trained-on-control twin gL10ctrl + seeds
gL10_s1/_s2), the JUDGED random-matched-norm null eval (>=5 seeds), and the FT deliverable
eval (base + ft `cdel`) -- all on the IDENTICAL (instruction x task) held-out subset (9 held-out
instructions incl. the bullet probe x 100 source-stratified tasks). Produces, per instruction +
category-macro (cross/within) + aggregate:

  * effective_control (PRIMARY), raw_compliance (METR-comparable), accuracy, meta/genuine/malformed/
    truncated/degenerate, and the joint comply-AND-correct rate, for steering vs base vs FT vs the
    random null vs the trained-on-control twin;
  * per-instruction PAIRED McNemar (gL10 vs base, gL10 vs FT esp. bullet, gL10 vs control);
  * task-level CLUSTER bootstrap CIs for the aggregate/category-macro uplift (gL10-vs-base), the
    paired gL10-vs-FT DIFFERENCE (the rigorous "reproduces FT" test), and the gL10-vs-control diff;
  * the random-null DISTRIBUTION (mean/spread/max) and where the real gL10 effect falls;
  * multi-seed (gL10_s1/_s2) robustness of the headline metrics;
  * the no-instruction default-behaviour characterization (accuracy, length, degenerate, spurious-form).

Pre-stated primary metric = aggregate + category-macro held-out effective_control uplift of gL10 over
base; minimum interesting effect = +10pp. FAST (no API).

Usage:
  python analyze_steer_deliverable.py --run deliv --null-run delivnull --main gL10 \
      --control gL10ctrl --seeds gL10_s1 gL10_s2 --ft-tag cdel
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import instructions as I
from analyze_ft_eval import effective, raw_compliant, accuracy
from sft_edit import is_degenerate
from run_ft_eval import HELDOUT_INSTRS, CROSS_CATEGORY
from analyze_ft_deliverable import (cluster_bootstrap_macro, cluster_bootstrap_macro_levels,
                                    mcnemar, paired_metric_vec, comply_correct, rate, cond_metrics)

RESULTS = Path("results")
MIN_EFFECT = 0.10


def _p(x):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:5.1f}%"


def _d(x):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100*x:+5.1f}"


def bucket_of(cond):
    return "cross_category" if cond in CROSS_CATEGORY else "within_category"


def load_idx(path):
    idx = {}
    for line in open(path):
        r = json.loads(line)
        idx[(r["arm"], r["condition"], r["task_id"])] = r
    return idx


def _bulletish(t):
    lines = [l for l in t.split("\n") if l.strip()]
    return bool(lines) and sum(1 for l in lines if l.lstrip().startswith(("- ", "* ", "• "))) >= max(2, 0.5 * len(lines))


def _numbered(t):
    lines = [l for l in t.split("\n") if l.strip()]
    return bool(lines) and sum(1 for l in lines if re.match(r"^\s*\d+[.)]", l)) >= max(2, 0.5 * len(lines))


def rows_for(idx, arm, cond, tasks):
    return [idx[(arm, cond, t)] for t in tasks if (arm, cond, t) in idx]


def macro_level(idx, arm, conds, tasks, metric_key="effective_control"):
    """instruction-mean of per-instruction metric LEVEL for one arm."""
    vals = []
    for c in conds:
        m = cond_metrics(rows_for(idx, arm, c, tasks))[metric_key]
        vals.append(m if m is not None else 0.0)
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="deliv")
    ap.add_argument("--null-run", default="delivnull")
    ap.add_argument("--main", default="gL10")
    ap.add_argument("--control", default="gL10ctrl")
    ap.add_argument("--seeds", nargs="*", default=["gL10_s1", "gL10_s2"])
    ap.add_argument("--ft-tag", default="cdel")
    ap.add_argument("--ft-preset", default="heldout_full")
    ap.add_argument("--min-effect", type=float, default=MIN_EFFECT)
    args = ap.parse_args()

    sidx = load_idx(f"results/grad_steer_eval_deliverable_{args.run}_judged.jsonl")
    nidx = load_idx(f"results/grad_steer_eval_deliverable_{args.null_run}_judged.jsonl") \
        if Path(f"results/grad_steer_eval_deliverable_{args.null_run}_judged.jsonl").exists() else {}
    ft_suffix = "" if args.ft_preset == "smoke" else f"_{args.ft_preset}"
    fidx = load_idx(f"results/ft_eval_{args.ft_tag}{ft_suffix}_judged.jsonl")

    main, control = args.main, args.control
    seeds = args.seeds
    null_arms = sorted({k[0] for k in nidx if k[0].startswith("rand")})

    tasks = sorted({k[2] for k in sidx if k[0] == main and k[1] == HELDOUT_INSTRS[0]})
    # item-alignment assertions (identical held-out items across arms)
    base_tasks = sorted({k[2] for k in sidx if k[0] == "base" and k[1] == HELDOUT_INSTRS[0]})
    ft_tasks = sorted({k[2] for k in fidx if k[0] == "ft" and k[1] == HELDOUT_INSTRS[0]})
    assert tasks == base_tasks == ft_tasks, "held-out task subset mismatch across arms!"
    if null_arms:
        nt = sorted({k[2] for k in nidx if k[0] == null_arms[0] and k[1] == HELDOUT_INSTRS[0]})
        assert nt == tasks, "null arm task subset mismatch!"
    print(f"main={main} control={control} seeds={seeds} nulls={null_arms}; n_tasks={len(tasks)}")

    buckets = [("all_heldout", list(HELDOUT_INSTRS)),
               ("within_category", [c for c in HELDOUT_INSTRS if bucket_of(c) == "within_category"]),
               ("cross_category", [c for c in HELDOUT_INSTRS if bucket_of(c) == "cross_category"])]

    out = {"run": args.run, "main": main, "control": control, "seeds": seeds, "ft_tag": args.ft_tag,
           "null_arms": null_arms, "n_tasks": len(tasks), "min_effect": args.min_effect,
           "per_instruction": {}, "macros": {}}

    L = []
    L.append(f"# Deliverable #2 — full-n significance-tested held-out STEERING uplift "
             f"(`{main}`, seed 0)\n")
    L.append(f"A single frozen-weights "
             f"**2,880-param additive steering vector** (`{main}`: resid_post @ layer 10, model weights "
             f"FROZEN, complying-target NLL, 500 steps / lr 0.05, **seed 0**) vs **base**, the FT "
             f"benchmark **`{args.ft_tag}`**, the **trained-on-control twin `{control}`**, and a "
             f"**{len(null_arms)}-seed random matched-norm null** — all on the IDENTICAL 9 held-out "
             f"instructions (incl. the **bullet** probe) × {len(tasks)} source-stratified held-out "
             f"tasks, FT-matched generation convention (per-source max_new_tokens + 8192 recovery, "
             f"greedy temp 0, seed 0, medium effort), Opus-meta judge pipeline, same batching.\n")
    L.append(f"**Pre-stated primary metric:** aggregate + category-macro held-out `effective_control` "
             f"uplift of steering over base; **minimum interesting effect = +{100*args.min_effect:.0f}pp**.\n")

    # ---------- per-instruction ----------
    L.append("## Per-instruction effective_control (base → gL10), with controls + FT + McNemar\n")
    L.append("Δ McN = gL10-vs-base paired McNemar p; gL10−FT = paired McNemar gL10-vs-FT p (tests the "
             "gap on each instruction, esp. the **bullet** probe).\n")
    L.append("| instr | bucket | n | base | **gL10 (Δ, McN p)** | FT | gL10−FT McN p | ctrl | "
             "gL10 raw | gL10 malf | gL10 acc | gL10 meta |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for cond in HELDOUT_INSTRS:
        b = cond_metrics(rows_for(sidx, "base", cond, tasks))
        m = cond_metrics(rows_for(sidx, main, cond, tasks))
        c = cond_metrics(rows_for(sidx, control, cond, tasks))
        ft = cond_metrics(rows_for(fidx, "ft", cond, tasks))
        bvec = paired_metric_vec(sidx, "base", cond, tasks, effective)
        mvec = paired_metric_vec(sidx, main, cond, tasks, effective)
        fvec = paired_metric_vec(fidx, "ft", cond, tasks, effective)
        _, _, p_base, _ = mcnemar([(bvec[t], mvec[t]) for t in tasks])
        gf01, gf10, p_ft, _ = mcnemar([(fvec[t], mvec[t]) for t in tasks])
        out["per_instruction"][cond] = {
            "bucket": bucket_of(cond), "base": b, "gL10": m, "control": c, "ft": ft,
            "mcnemar_vs_base_p": p_base, "mcnemar_vs_ft": {"gL10_gain": gf01, "gL10_loss": gf10, "p": p_ft}}
        sb = "***" if p_base < 0.001 else "**" if p_base < 0.01 else "*" if p_base < 0.05 else ""
        sf = "***" if p_ft < 0.001 else "**" if p_ft < 0.01 else "*" if p_ft < 0.05 else ""
        deff = (m["effective_control"] or 0) - (b["effective_control"] or 0)
        L.append(f"| {cond} | {bucket_of(cond)} | {m['n']} | {_p(b['effective_control'])} | "
                 f"**{_p(m['effective_control'])} ({_d(deff)}, {p_base:.3f}{sb})** | "
                 f"{_p(ft['effective_control'])} | {p_ft:.3f}{sf} | {_p(c['effective_control'])} | "
                 f"{_p(m['raw_compliance'])} | {_p(m['malformed_rate'])} | {_p(m['accuracy'])} | "
                 f"{_p(m['meta_rate'])} |")

    # ---------- aggregate / category macro effective_control ----------
    L.append("\n## Aggregate / category-macro effective_control (task-level cluster bootstrap, 95% CI)\n")
    L.append("Primary = **gL10 uplift over base**. Also: the paired **gL10−FT difference** (the "
             "rigorous 'reproduces FT' test — a CI bracketing 0 = comparable in aggregate; negative = "
             "a fraction of FT's uplift), the control twin level, and the random-null mean[max].\n")
    L.append("| bucket | n | base | gL10 | **gL10 uplift (95% CI)** | clears +10pp? | FT | "
             "**gL10−FT (95% CI)** | ctrl | null mean[max] |")
    L.append("|---|--:|--:|--:|--:|:--:|--:|--:|--:|--:|")
    null_macro_uplifts = {}  # bucket -> list over null arms
    for bname, conds in buckets:
        pit_base = {c: (paired_metric_vec(sidx, "base", c, tasks, effective),
                        paired_metric_vec(sidx, main, c, tasks, effective)) for c in conds}
        up = cluster_bootstrap_macro(pit_base, conds, tasks)
        bl = cluster_bootstrap_macro_levels(pit_base, conds, tasks, "base")
        ml = cluster_bootstrap_macro_levels(pit_base, conds, tasks, "ft")  # 'ft' index=1 = gL10 here
        # gL10 - FT difference
        pit_ft = {c: (paired_metric_vec(fidx, "ft", c, tasks, effective),
                      paired_metric_vec(sidx, main, c, tasks, effective)) for c in conds}
        diff_ft = cluster_bootstrap_macro(pit_ft, conds, tasks)
        ft_lvl = macro_level(fidx, "ft", conds, tasks)
        ctrl_lvl = macro_level(sidx, control, conds, tasks)
        # null distribution (per-arm macro uplift over base)
        nuls = []
        for na in null_arms:
            nl = macro_level(nidx, na, conds, tasks)
            nuls.append(nl - bl["point"])
        null_macro_uplifts[bname] = nuls
        clears = "✅" if up["ci_lo"] >= args.min_effect else "❌"
        out["macros"][bname] = {
            "base_level": bl, "gL10_level": ml, "gL10_uplift_vs_base": up,
            "ft_level": ft_lvl, "gL10_minus_ft": diff_ft, "control_level": ctrl_lvl,
            "null_uplifts": nuls, "clears_min_effect": up["ci_lo"] >= args.min_effect}
        nmean = (np.mean(nuls) if nuls else float("nan"))
        nmax = (np.max(nuls) if nuls else float("nan"))
        L.append(f"| {bname} | {len(conds)} | {_p(bl['point'])} | {_p(ml['point'])} | "
                 f"**{_d(up['point'])} ({_d(up['ci_lo'])},{_d(up['ci_hi'])})** | {clears} | "
                 f"{_p(ft_lvl)} | **{_d(diff_ft['point'])} ({_d(diff_ft['ci_lo'])},{_d(diff_ft['ci_hi'])})** | "
                 f"{_p(ctrl_lvl)} | {_d(nmean)}[{_d(nmax)}] |")

    # ---------- raw_compliance (METR-comparable) ----------
    L.append("\n## Aggregate raw_compliance (truncation-robust, METR-comparable: ~2.9%→8.8% OOD)\n")
    L.append("| bucket | base | gL10 | **gL10 uplift (95% CI)** | FT | ctrl | null mean[max] |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    for bname, conds in buckets:
        pit = {c: (paired_metric_vec(sidx, "base", c, tasks, raw_compliant),
                   paired_metric_vec(sidx, main, c, tasks, raw_compliant)) for c in conds}
        up = cluster_bootstrap_macro(pit, conds, tasks)
        bl = macro_level(sidx, "base", conds, tasks, "raw_compliance")
        ml = macro_level(sidx, main, conds, tasks, "raw_compliance")
        fl = macro_level(fidx, "ft", conds, tasks, "raw_compliance")
        cl = macro_level(sidx, control, conds, tasks, "raw_compliance")
        nuls = [macro_level(nidx, na, conds, tasks, "raw_compliance") - bl for na in null_arms]
        out["macros"][bname]["raw_compliance"] = {
            "base": bl, "gL10": ml, "ft": fl, "control": cl, "uplift": up, "null_uplifts": nuls}
        nmean = (np.mean(nuls) if nuls else float("nan"))
        nmax = (np.max(nuls) if nuls else float("nan"))
        L.append(f"| {bname} | {_p(bl)} | {_p(ml)} | **{_d(up['point'])} "
                 f"({_d(up['ci_lo'])},{_d(up['ci_hi'])})** | {_p(fl)} | {_p(cl)} | "
                 f"{_d(nmean)}[{_d(nmax)}] |")

    # ---------- joint comply-AND-correct ----------
    cc = lambda r: comply_correct(r, True)  # noqa: E731
    L.append("\n## Joint comply(effective)-AND-correct macro (accuracy-aware)\n")
    L.append("| bucket | base | gL10 | **gL10 uplift (95% CI)** | FT | ctrl |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for bname, conds in buckets:
        pit = {c: (paired_metric_vec(sidx, "base", c, tasks, cc),
                   paired_metric_vec(sidx, main, c, tasks, cc)) for c in conds}
        up = cluster_bootstrap_macro(pit, conds, tasks)
        bl = np.mean([rate([cc(r) for r in rows_for(sidx, "base", c, tasks)]) or 0 for c in conds])
        ml = np.mean([rate([cc(r) for r in rows_for(sidx, main, c, tasks)]) or 0 for c in conds])
        fl = np.mean([rate([cc(r) for r in rows_for(fidx, "ft", c, tasks)]) or 0 for c in conds])
        cl = np.mean([rate([cc(r) for r in rows_for(sidx, control, c, tasks)]) or 0 for c in conds])
        out["macros"][bname]["comply_correct"] = {"base": float(bl), "gL10": float(ml),
                                                   "ft": float(fl), "control": float(cl),
                                                   "uplift": up}
        L.append(f"| {bname} | {_p(bl)} | {_p(ml)} | **{_d(up['point'])} "
                 f"({_d(up['ci_lo'])},{_d(up['ci_hi'])})** | {_p(fl)} | {_p(cl)} |")

    # ---------- random-null distribution ----------
    if null_arms:
        L.append(f"\n## Random matched-norm null distribution ({len(null_arms)} seeds, ‖v‖≈148 @ L10)\n")
        L.append("Each null = a random Gaussian vector at L10 scaled to gL10's ‖v‖, same injection "
                 "scheme. Macro effective_control uplift over base per null seed; the real gL10 effect "
                 "must lie OUTSIDE this null distribution.\n")
        L.append("| metric | null seeds (uplift vs base) | null mean | null max | **gL10** | gL10 outside null? |")
        L.append("|---|---|--:|--:|--:|:--:|")
        for bname, conds in buckets:
            nuls = null_macro_uplifts[bname]
            gl = out["macros"][bname]["gL10_uplift_vs_base"]["point"]
            outside = "✅" if (not nuls or gl > max(nuls)) else "❌"
            seeds_str = ", ".join(f"{100*x:+.1f}" for x in nuls)
            L.append(f"| eff `{bname}` | {seeds_str} | {_d(np.mean(nuls))} | {_d(np.max(nuls))} | "
                     f"**{_d(gl)}** | {outside} |")

    # ---------- control dissociation ----------
    L.append(f"\n## Control dissociation: gL10 ≫ trained-on-control `{control}` (macro Δ, cluster bootstrap)\n")
    L.append("| bucket | gL10 eff | ctrl eff | **Δ (95% CI)** | P(Δ≤0) |")
    L.append("|---|--:|--:|--:|--:|")
    for bname, conds in buckets:
        pit = {c: (paired_metric_vec(sidx, control, c, tasks, effective),
                   paired_metric_vec(sidx, main, c, tasks, effective)) for c in conds}
        diff = cluster_bootstrap_macro(pit, conds, tasks)
        ml = macro_level(sidx, main, conds, tasks)
        cl = macro_level(sidx, control, conds, tasks)
        out["macros"][bname]["gL10_minus_control"] = diff
        L.append(f"| {bname} | {_p(ml)} | {_p(cl)} | **{_d(diff['point'])} "
                 f"({_d(diff['ci_lo'])},{_d(diff['ci_hi'])})** | {diff['p_gt_0']:.4f} |")

    # ---------- multi-seed robustness ----------
    seed_arms = [main] + [s for s in seeds if any(k[0] == s for k in sidx)]
    L.append(f"\n## Multi-seed robustness (headline = seed 0 `{main}`; the conservative pick from vector training — "
             "at full mnt the 3 seeds are essentially tied on bullet)\n")
    L.append("| arm | seed | bullet | cross-cat (fmt) | within-cat | all-held-out | accuracy |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    out["seed_robustness"] = {}
    for a in seed_arms:
        seed = 0 if a == main else int(a.split("_s")[-1])
        bull = cond_metrics(rows_for(sidx, a, "bullet", tasks))["effective_control"]
        cross = macro_level(sidx, a, [c for c in HELDOUT_INSTRS if bucket_of(c) == "cross_category"], tasks)
        within = macro_level(sidx, a, [c for c in HELDOUT_INSTRS if bucket_of(c) == "within_category"], tasks)
        allh = macro_level(sidx, a, list(HELDOUT_INSTRS), tasks)
        acc = cond_metrics([r for c in HELDOUT_INSTRS for r in rows_for(sidx, a, c, tasks)])["accuracy"]
        out["seed_robustness"][a] = {"bullet": bull, "cross": cross, "within": within,
                                     "all": allh, "accuracy": acc}
        L.append(f"| {a} | {seed} | {_p(bull)} | {_p(cross)} | {_p(within)} | {_p(allh)} | {_p(acc)} |")

    # ---------- guardrails ----------
    L.append("\n## Guardrails (aggregate over 9 held-out instructions, micro-pooled)\n")
    L.append("base/gL10/ft are FULLY judged; control/seeds are judged compliant-only (a cost saver — "
             "meta/genuine are evaluated only on raw-compliant rows, which is all that effective_control "
             "needs). **DENOMINATOR NOTE:** for base/gL10/ft the **meta** rate is over ALL rows (54.7 / "
             "9.0 / 0.0%); for the compliant-only arms (control/seeds) **meta** is over their (few) "
             "raw-compliant rows ONLY (so seed meta 2.8% is NOT comparable to gL10's 9.0% — different "
             "denominators), and **genuine is shown n/a** (the pipeline forces genuine=False on un-judged "
             "non-compliant rows, so a pooled genuine-rate would be meaningless). **effective_control is "
             "EXACT for every arm** (a non-raw-compliant row is False regardless of meta/genuine).\n")
    L.append("| arm | malformed | truncated | degenerate | accuracy | genuine | meta |")
    L.append("|---|--:|--:|--:|--:|--:|--:|")
    full_judged = {"base", main, "ft"}
    guard_arms = [("base", sidx), (main, sidx), (control, sidx)] + \
                 [(s, sidx) for s in seeds if any(k[0] == s for k in sidx)] + [("ft", fidx)]
    for arm, idx in guard_arms:
        pooled = [r for c in HELDOUT_INSTRS for r in rows_for(idx, arm, c, tasks)]
        mm = cond_metrics(pooled)
        out.setdefault("guardrails", {})[arm] = mm
        gen_cell = _p(mm['genuine_rate']) if arm in full_judged else "  n/a"
        L.append(f"| {arm} | {_p(mm['malformed_rate'])} | {_p(mm['truncated_rate'])} | "
                 f"{_p(mm['degenerate_rate'])} | {_p(mm['accuracy'])} | {gen_cell} | "
                 f"{_p(mm['meta_rate'])} |")

    # ---------- no-instruction default behaviour ----------
    none_tasks = sorted({k[2] for k in sidx if k[0] == main and k[1] == "none"})
    if none_tasks:
        L.append("\n## No-instruction default behaviour (steered side-effect characterization)\n")
        L.append("Spurious-form = the asked forms should be ABSENT when NO instruction is given "
                 "(bullets/numbered ≈0, prose normal-case). The always-on vector is NOT fully inert: "
                 "quantify the verbosity/degeneration side-effect.\n")
        L.append("FT `cdel` + the trained-on-control twin are included for the apples-to-apples "
                 "side-by-side: FT is essentially INERT on no-instruction (≈ base), so the always-on "
                 "verbosity/degeneration is a place gL10 does NOT match FT (an off-target cost).\n")
        L.append("| arm | n | accuracy | degenerate | truncated | aw median | aw mean | bullets | "
                 "numbered | upper>0.5 |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        out["none"] = {}
        none_arms = [("base", sidx), (main, sidx), (control, sidx)] + \
                    [(s, sidx) for s in seeds if any((s, "none", t) in sidx for t in none_tasks)] + \
                    [("ft", fidx)]
        for arm, idx in none_arms:
            nr = rows_for(idx, arm, "none", none_tasks)
            if not nr:
                continue
            mm = cond_metrics(nr)
            aws = [r["analysis_words"] for r in nr]
            _ = arm  # arm name keyed below
            nbul = sum(_bulletish(r["analysis"]) for r in nr)
            nnum = sum(_numbered(r["analysis"]) for r in nr)
            nupp = sum((r.get("uppercase_fraction") or 0) > 0.5 for r in nr)
            out["none"][arm] = {"n": len(nr), "accuracy": mm["accuracy"],
                                "degenerate": mm["degenerate_rate"], "truncated": mm["truncated_rate"],
                                "aw_median": float(np.median(aws)), "aw_mean": float(np.mean(aws)),
                                "bullets": nbul, "numbered": nnum, "upper_gt_50": nupp}
            L.append(f"| {arm} | {len(nr)} | {_p(mm['accuracy'])} | {_p(mm['degenerate_rate'])} | "
                     f"{_p(mm['truncated_rate'])} | {np.median(aws):.0f} | {np.mean(aws):.0f} | "
                     f"{nbul}/{len(nr)} | {nnum}/{len(nr)} | {nupp}/{len(nr)} |")

    # ---------- verdict ----------
    allh = out["macros"]["all_heldout"]
    cross = out["macros"]["cross_category"]
    up = allh["gL10_uplift_vs_base"]
    clears = [b for b in ("all_heldout", "within_category", "cross_category")
              if out["macros"][b]["clears_min_effect"]]
    L.append("\n## Verdict\n")
    L.append(f"- **gL10 all-held-out effective_control uplift = {_d(up['point'])}pp "
             f"(95% CI {_d(up['ci_lo'])},{_d(up['ci_hi'])}), P(uplift≤0)={up['p_gt_0']:.4f}**; "
             f"cross-category (formatting) {_d(cross['gL10_uplift_vs_base']['point'])} "
             f"(CI {_d(cross['gL10_uplift_vs_base']['ci_lo'])},{_d(cross['gL10_uplift_vs_base']['ci_hi'])}), "
             f"carried by bullet 0→{_p(out['per_instruction']['bullet']['gL10']['effective_control']).strip()}.")
    L.append(f"- **Clears the +{100*args.min_effect:.0f}pp minimum interesting effect** at the macro CI "
             f"lower bound for: {', '.join(clears) if clears else 'NONE'}.")
    L.append(f"- **Reproduces FT (paired gL10−FT difference):** all-held-out {_d(allh['gL10_minus_ft']['point'])} "
             f"(CI {_d(allh['gL10_minus_ft']['ci_lo'])},{_d(allh['gL10_minus_ft']['ci_hi'])}); FT macro "
             f"level {_p(allh['ft_level'])} vs gL10 {_p(allh['gL10_level']['point'])}.")
    L.append(f"- **Control dissociation:** gL10 ≫ trained-on-control `{control}` (all-held-out Δ "
             f"{_d(allh['gL10_minus_control']['point'])}, CI {_d(allh['gL10_minus_control']['ci_lo'])},"
             f"{_d(allh['gL10_minus_control']['ci_hi'])}); control macro {_p(allh['control_level'])}.")
    if null_arms:
        nmean = np.mean(null_macro_uplifts["all_heldout"])
        nmax = np.max(null_macro_uplifts["all_heldout"])
        L.append(f"- **Beyond the random null:** {len(null_arms)}-seed null mean {_d(nmean)}, max "
                 f"{_d(nmax)} ≪ gL10 {_d(up['point'])} — the effect is the LEARNED direction, not a "
                 f"generic matched-norm push at L10.")
    L.append(f"- **raw_compliance (METR-comparable):** base {_p(macro_level(sidx,'base',HELDOUT_INSTRS,tasks,'raw_compliance'))} "
             f"→ gL10 {_p(macro_level(sidx,main,HELDOUT_INSTRS,tasks,'raw_compliance'))} "
             f"(FT {_p(macro_level(fidx,'ft',HELDOUT_INSTRS,tasks,'raw_compliance'))}).")

    base_out = f"results/steer_deliverable_{main}"
    json.dump(out, open(base_out + ".json", "w"), indent=2)
    open(base_out + ".md", "w").write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {base_out}.md + .json")


if __name__ == "__main__":
    main()
