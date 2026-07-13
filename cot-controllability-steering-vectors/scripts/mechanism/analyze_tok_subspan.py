"""Analysis: WHICH instruction tokens.
  (a) per-sub-span steered−base attention mass of the recruited late full-attention heads
      (results/tok_subspan_attn.npz), per instruction + the fraction of gL10's induced
      attention-increment that lands on each sub-span (with task-cluster CIs).
  (b) per-sub-span CAUSAL masking: how much of the induced form-logit shift each sub-span removes
      (results/tok_subspan_mask_raw.json), per instruction with CIs — the necessary-sub-span result.
Fast (no Modal).  python analyze_tok_subspan.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import tok_lib as T
from analyze_tok_verify import cluster_boot_ratio, MARKER, FORM_CONDS

RESULTS = Path("results")
PLOTS = Path("plots")
SUBSPANS = ["spec", "cot_target", "directive", "rest"]
LATE_FULL = [13, 15, 17, 19, 21, 23]


def main():
    meta = json.load(open(RESULTS / "tok_subspan_attn_meta.json"))
    npz = np.load(RESULTS / "tok_subspan_attn.npz")
    conds = np.array(meta["conds"]); tasks = np.array(meta["tasks"])
    has_instr = np.array(meta["has_instr"])

    out = {"attn_mass": {}, "attn_increment_share": {}, "causal_mask": {}}
    md = ["# WHICH instruction tokens the recruited late heads attend to, and which are NECESSARY\n",
          "Sub-spans (clean partition of the in-context instruction, validated by decoding in "
          "`results/tok_subspan_inspection.md`): **spec** = the literal FORMAT SPECIFIER (marker chars "
          "+ form-name word), **cot_target** = references to the model's reasoning ('chain of thought'/"
          "'your reasoning'/'step-by-step'), **directive** = the imperative verb(s), **rest** = "
          "preamble/connectives. Recruited late full-attention layers: L13/15/17/19/21/23.\n"]

    # ---- (a) per-sub-span attention mass: sum over heads within a late full-attn layer, mean over
    #          those layers; steered vs base; per instruction. ----
    def layer_head_sum(arm, span):
        # npz[f"{arm}_{span}"] : [n, 24, 64] per-head mass; sum over heads, mean over LATE_FULL layers
        a = npz[f"{arm}_{span}"][:, LATE_FULL, :]      # [n, |LATE|, 64]
        return np.nansum(a, axis=2).mean(axis=1)        # [n] mass-onto-span summed over heads, mean over late layers

    md.append("## (a) Attention mass onto each sub-span (sum over heads, mean over late full-attn "
              "layers): base → gL10-steered\n")
    md.append("| instruction | span | base mass | steered mass | Δ (steer−base) [95% CI] |")
    md.append("|---|---|--:|--:|--:|")
    for cond in FORM_CONDS + ["initial_caps"]:
        m = conds == cond
        if m.sum() == 0:
            continue
        out["attn_mass"][cond] = {}
        for span in SUBSPANS + ["instruction", "sink", "self_prefix"]:
            b = layer_head_sum("base", span)[m]
            s = layer_head_sum("steer", span)[m]
            tk = tasks[m]
            # Δ mean with cluster bootstrap (ratio trick: num=Δ, den=ones)
            d_pt = float(np.nanmean(s - b))
            # bootstrap mean Δ by clusters
            uniq = np.unique(tk)
            idxby = {t: np.where(tk == t)[0] for t in uniq}
            rng = np.random.default_rng(1)
            boots = []
            dd = s - b
            for _ in range(2000):
                samp = rng.choice(uniq, size=len(uniq), replace=True)
                selv = np.concatenate([idxby[t] for t in samp])
                boots.append(np.nanmean(dd[selv]))
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            out["attn_mass"][cond][span] = {"base": float(np.nanmean(b)), "steer": float(np.nanmean(s)),
                                            "delta": d_pt, "lo": float(lo), "hi": float(hi)}
            md.append(f"| {cond} | {span} | {np.nanmean(b):.3f} | {np.nanmean(s):.3f} | "
                      f"{d_pt:+.3f} [{lo:+.3f},{hi:+.3f}] |")
        md.append("|  |  |  |  |  |")

    # ---- share of the induced instruction-attention increment that lands on each sub-span ----
    md.append("\n## (a') Of gL10's induced attention INCREMENT onto the instruction, what share lands "
              "on each sub-span?\n")
    md.append("| instruction | spec share | cot_target share | directive share | rest share |")
    md.append("|---|--:|--:|--:|--:|")
    for cond in FORM_CONDS + ["initial_caps"]:
        m = conds == cond
        if m.sum() == 0:
            continue
        deltas = {}
        for span in SUBSPANS:
            b = layer_head_sum("base", span)[m]
            s = layer_head_sum("steer", span)[m]
            deltas[span] = float(np.nansum(s - b))
        tot = sum(max(0.0, deltas[s]) for s in SUBSPANS) or 1e-9
        share = {s: max(0.0, deltas[s]) / tot for s in SUBSPANS}
        out["attn_increment_share"][cond] = share
        md.append(f"| {cond} | {100*share['spec']:.0f}% | {100*share['cot_target']:.0f}% | "
                  f"{100*share['directive']:.0f}% | {100*share['rest']:.0f}% |")

    # ---- (b) per-sub-span CAUSAL mask: % of induced form-logit shift removed ----
    d = json.load(open(RESULTS / "tok_subspan_mask_raw.json"))
    sm = d["seq_meta"]; cand = d["cand_ids"]
    cidx = {t: i for i, t in enumerate(cand)}
    _, _, label2id, _ = T.candidate_tokens()
    mconds = np.array([x["cond"] for x in sm])
    mtasks = np.array([x["task_id"] for x in sm])

    def col(name, key, c):
        res = d["results"][name]
        return np.array([r[key][c] if r.get(key) is not None else np.nan for r in res])

    masks = ["mask_spec", "mask_cot_target", "mask_directive", "mask_rest", "mask_nonspec",
             "mask_whole"]
    md.append("\n## (b) CAUSAL sub-span masking: % of the induced form-logit shift REMOVED by zeroing "
              "attention onto each sub-span (full-attn heads, steered run) [95% CI]\n")
    md.append("| instruction | full shift | " + " | ".join(m.replace("mask_", "") for m in masks) + " |")
    md.append("|---|--:|" + "--:|" * len(masks))
    for cond in FORM_CONDS:
        m = mconds == cond
        c = cidx[label2id[MARKER[cond]]]
        base = col("mask_spec", "base", c)[m]
        steer = col("mask_spec", "steer", c)[m]
        tk = mtasks[m]
        shift = float((steer - base).mean())
        den = steer - base
        out["causal_mask"][cond] = {"shift": shift}
        cells = [f"{shift:+.2f}"]
        for name in masks:
            removed = steer - col(name, "patched", c)[m]   # how much the shift is REMOVED
            r_pt, r_lo, r_hi = cluster_boot_ratio(removed, den, tk)
            out["causal_mask"][cond][name] = [r_pt, r_lo, r_hi]
            cells.append(f"{100*r_pt:.0f} [{100*r_lo:.0f},{100*r_hi:.0f}]")
        md.append(f"| {cond} | " + " | ".join(cells) + " |")
    md.append("\n**Read:** % REMOVED = (steered marker logit − masked marker logit) / (steered − base). "
              "`mask_whole` ≈ 100% (replicates the whole-instruction knockout). The question is whether **spec** alone removes most "
              "of it (the format-specifier hypothesis) or whether the CoT-target / the broader span is "
              "needed. `mask_nonspec` = mask everything EXCEPT the specifier. Established-vs-suggestive "
              "+ per-instruction scoping in the write-up.\n")

    (RESULTS / "tok_subspan.md").write_text("\n".join(md) + "\n")
    json.dump(out, open(RESULTS / "tok_subspan.json", "w"), indent=2)
    print("wrote results/tok_subspan.md + .json")
    _plot(out)


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))

    # left: share of induced attention increment per sub-span (stacked bars)
    ax = axes[0]
    conds = [c for c in FORM_CONDS + ["initial_caps"] if c in out["attn_increment_share"]]
    bottoms = np.zeros(len(conds))
    colors = {"spec": "#c0392b", "cot_target": "#2980b9", "directive": "#e67e22", "rest": "#95a5a6"}
    x = np.arange(len(conds))
    for span in SUBSPANS:
        vals = [100 * out["attn_increment_share"][c][span] for c in conds]
        ax.bar(x, vals, 0.6, bottom=bottoms, color=colors[span], label=span)
        bottoms += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=15)
    ax.set_ylabel("% of gL10's induced attention increment")
    ax.set_title("Where the induced attention lands (by sub-span)")
    ax.legend(fontsize=8)

    # right: causal mask % removed per sub-span. Include `rest` so the xml_steps counter-case
    # (rest > spec) is VISIBLE, not just in the table.
    ax = axes[1]
    cm_conds = [c for c in ["bullet", "numbered", "section_headers", "xml_steps"]
                if c in out["causal_mask"]]
    masks = ["mask_spec", "mask_cot_target", "mask_rest", "mask_nonspec", "mask_whole"]
    mcolors = ["#c0392b", "#2980b9", "#95a5a6", "#8e44ad", "#000000"]
    nb = len(masks)
    x = np.arange(len(cm_conds)); w = 0.16
    for j, (name, col) in enumerate(zip(masks, mcolors)):
        v, lo, hi = [], [], []
        for c in cm_conds:
            p, l, h = out["causal_mask"][c][name]
            v.append(100 * p); lo.append(100 * (p - l)); hi.append(100 * (h - p))
        ax.bar(x + (j - (nb - 1) / 2) * w, v, w, yerr=[lo, hi], capsize=2, color=col,
               label=name.replace("mask_", ""))
    ax.axhline(100, color="gray", ls=":", lw=1)
    ax.set_ylim(top=200)
    # flag the section_headers spec bar as noise (tiny +2.78 form-logit shift => unstable ratio)
    if "section_headers" in cm_conds:
        sh = cm_conds.index("section_headers")
        ax.annotate("section_headers spec >100%:\ntiny +2.8 shift → ratio noisy",
                    xy=(sh - 0.32, 149), xytext=(0.65, 168), fontsize=7, color="#555",
                    ha="left", arrowprops=dict(arrowstyle="->", color="#888", lw=0.8))
    # annotate the xml counter-case (rest > spec)
    if "xml_steps" in cm_conds:
        xi = cm_conds.index("xml_steps")
        ax.annotate("xml: rest 55% > spec 39%\n(counter-case)", xy=(xi, 55),
                    xytext=(xi - 0.55, 95), fontsize=7, color="#333", ha="left",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels(cm_conds, rotation=15)
    ax.set_ylabel("% of induced form-logit shift REMOVED")
    ax.set_title("Causal sub-span masking (full-attn heads):\n"
                 "spec is the largest single share for bullet/numbered; xml is rest-dominant")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "tok_subspan_attribution.png", dpi=130)
    plt.close(fig)
    print("wrote plots/tok_subspan_attribution.png")


if __name__ == "__main__":
    main()
