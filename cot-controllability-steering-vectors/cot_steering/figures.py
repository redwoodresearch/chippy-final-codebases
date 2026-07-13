"""Regenerate the release figures from the released summary artifacts.

Three main figures (the graphs in the post) plus one supplementary figure:
fig1 (headline: fine-tune vs steering vector on held-out CoT control), fig2 (attention onto
each instruction part, base vs steered), fig3 (attention concentrates on the format-specifier
tokens), and fig4 (supplementary: the difference-of-means and random-vector comparisons).

Each ``figN_*`` function loads only small JSON summaries (via :mod:`cot_steering.artifacts`,
which fetches from Hugging Face with a local fallback), builds the matplotlib figure, and
returns ``(fig, key_numbers)``. ``key_numbers`` is a small dict of the load-bearing values
plotted, so callers can *assert* the figure reproduces the reference numerically.

These figures plot saved summary metrics only -- there is **no model generation, no GPU,
no training**. See ``generate_figures.py`` for the CPU entry point and ``figures/REPRODUCTION.md``
for the numeric/visual comparison against the reference figures.

Figure -> data map
------------------
* fig1  steer_deliverable_gL10.json + ft_deliverable_cdel_vs_ctrldel.json
* fig2  fig2_subspan_attention.json (base vs steered attention per instruction part, bootstrap CIs)
* fig3  fig3_token_shading.json (per-instruction-part attention deltas; tokens shaded by part avg)
* fig4  steer_deliverable_gL10.json (bullet base/vector/FT) + steer_eval_heldout_analysis.json
        (average-difference direction, n=39) + fig4_random_null.json (random vectors)

fig1 additionally reports the no-instruction control from steer_deliverable_gL10.json
(``none``): the vector produces no spurious bullets/numbering/casing when no formatting
instruction is given, at the cost of increased verbosity and degeneration.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colormaps, colors  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .artifacts import load_figure_json  # noqa: E402

# ---- shared styling ---------------------------------------------------------------------
C_BASE = "#9aa0a6"   # grey  - base model
C_FT = "#1f77b4"     # blue  - fine-tuned (LoRA)
C_VEC = "#d62728"    # red   - steering vector
C_DOM = "#c8b9a6"    # tan   - average-difference direction
C_NULL = "#dcdcdc"   # light - random vector

_RC = {
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "figure.dpi": 130,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
}

# Held-out instruction display order + labels (fig1 rows).
FIG1_ORDER = ["bullet", "terse_25w", "numbered", "xml_steps", "no_word_so",
              "initial_caps", "include_exactly_twice", "section_headers", "child_explanation"]
FIG1_LABELS = {
    "bullet": "bullet lines", "terse_25w": "terse (\u226425 words)", "numbered": "numbered lines",
    "xml_steps": "XML step tags", "no_word_so": 'no word "so"', "initial_caps": "Capitalize Each Word",
    "include_exactly_twice": "include a word \u00d72", "section_headers": "section headers",
    "child_explanation": "explain to a child",
}


def _wilson(p: float, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# =========================================================================================
# FIG 1 -- a frozen-weights steering vector reproduces fine-tuning's held-out CoT control
# =========================================================================================
def fig1_headline(source=None):
    sd = load_figure_json("steer_deliverable_gL10.json", source=source)
    ftd = load_figure_json("ft_deliverable_cdel_vs_ctrldel.json", source=source)
    pi = sd["per_instruction"]
    base = [pi[k]["base"]["effective_control"] * 100 for k in FIG1_ORDER]
    ft = [pi[k]["ft"]["effective_control"] * 100 for k in FIG1_ORDER]
    vec = [pi[k]["gL10"]["effective_control"] * 100 for k in FIG1_ORDER]

    # Aggregate = macro average over the 9 held-out instructions (n=100 each) -- the first bar.
    agg = {"base": float(np.mean(base)), "ft": float(np.mean(ft)), "vec": float(np.mean(vec))}
    # The aggregate-uplift CIs + the paired difference (quoted in the post; not plotted).
    m = sd["macros"]["all_heldout"]
    ftup = ftd["macros"]["all_heldout"]["uplift"]
    gup = m["gL10_uplift_vs_base"]
    diff = m["gL10_minus_ft"]

    labels = ["all 9 held-out\n(aggregate)"] + [FIG1_LABELS[k] for k in FIG1_ORDER]
    b_vals = [agg["base"]] + base
    f_vals = [agg["ft"]] + ft
    v_vals = [agg["vec"]] + vec
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(9.5, 7.0))
        y = np.arange(len(labels))[::-1]  # aggregate on top
        h = 0.26
        ax.barh(y + h, b_vals, h, label="base model", color=C_BASE)
        ax.barh(y, f_vals, h, label="fine-tuned (LoRA)", color=C_FT)
        ax.barh(y - h, v_vals, h, label="steering vector", color=C_VEC)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlabel("strict CoT-control compliance (%)")
        ax.set_xlim(0, 76)
        ax.axhline(y[0] - 0.55, color="#bbb", lw=0.8, ls=":")  # aggregate | per-instruction
        ax.legend(loc="lower right")
        ax.set_title("Activation Steering Can Increase CoT Controllability\n"
                     "(9 held-out instructions, n=100 tasks each)", fontsize=12.5)

    keys = {
        "agg": {k: round(v, 2) for k, v in agg.items()},
        "bullet": {"base": base[0], "ft": ft[0], "vec": vec[0]},
        "terse": {"base": base[1], "ft": ft[1], "vec": vec[1]},
        "numbered": {"base": base[2], "ft": ft[2], "vec": vec[2]},
        "ft_uplift_pp": ftup["point"] * 100,
        "ft_uplift_ci": [ftup["ci_lo"] * 100, ftup["ci_hi"] * 100],
        "vec_uplift_pp": gup["point"] * 100,
        "vec_uplift_ci": [gup["ci_lo"] * 100, gup["ci_hi"] * 100],
        "paired_diff_pp": diff["point"] * 100,
        "paired_diff_ci": [diff["ci_lo"] * 100, diff["ci_hi"] * 100],
    }
    # No-instruction control (not plotted): with no formatting instruction the vector produces no
    # spurious bullets/numbering/casing, but it does increase verbosity and degeneration.
    nc = sd["none"]
    keys["none_control"] = {
        "vec_bullets": nc["gL10"]["bullets"], "vec_numbered": nc["gL10"]["numbered"],
        "vec_upper_gt_50": nc["gL10"]["upper_gt_50"], "n": nc["gL10"]["n"],
        "base_aw_mean": nc["base"]["aw_mean"], "vec_aw_mean": nc["gL10"]["aw_mean"],
        "base_degenerate": nc["base"]["degenerate"], "vec_degenerate": nc["gL10"]["degenerate"],
    }
    return fig, keys


# =========================================================================================
# FIG 2 -- attention onto each instruction part, base vs steered (bullet & numbered)
# =========================================================================================
_FIG2_SPANS = ["spec", "cot_target", "directive", "rest"]
_FIG2_SPAN_LABEL = {
    "spec": "format specifier\n(\u201cbulleted\u201d, \u201c'- '\u201d)",
    "cot_target": "\u201cyour reasoning\u201d\nreference",
    "directive": "directive verbs\n(\u201cwrite\u201d, \u201cstart\u201d)",
    "rest": "other words",
}


def fig2_attention_subspan(source=None):
    fa = load_figure_json("fig2_subspan_attention.json", source=source)
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
        fig.subplots_adjust(bottom=0.30, top=0.86, wspace=0.08)
        for ax, cond in zip(axes, ["bullet", "numbered"]):
            spans_d = fa[cond]["spans"]
            xc = np.arange(len(_FIG2_SPANS)); w = 0.38
            for j, (arm, col, lab) in enumerate([("base", C_BASE, "base model"),
                                                 ("steer", C_VEC, "with steering vector")]):
                pts = [spans_d[s][arm]["mean"] for s in _FIG2_SPANS]
                los = [spans_d[s][arm]["mean"] - spans_d[s][arm]["lo"] for s in _FIG2_SPANS]
                his = [spans_d[s][arm]["hi"] - spans_d[s][arm]["mean"] for s in _FIG2_SPANS]
                ax.bar(xc + (j - 0.5) * w, pts, w, yerr=[los, his], capsize=4, color=col, label=lab)
            ax.set_xticks(xc)
            ax.set_xticklabels([_FIG2_SPAN_LABEL[s] for s in _FIG2_SPANS], fontsize=9)
            ax.set_title(f"{cond} instruction")
        axes[0].set_ylabel("attention onto the sub-span\n(recruited late heads)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=10)
        fig.suptitle("Adding the steering vector raises attention on the instruction",
                     fontsize=12, y=0.97)
    keys = {cond: {s: {"base": round(fa[cond]["spans"][s]["base"]["mean"], 2),
                       "steer": round(fa[cond]["spans"][s]["steer"]["mean"], 2)}
                   for s in _FIG2_SPANS} for cond in ("bullet", "numbered")}
    return fig, keys


# =========================================================================================
# FIG 3 -- instruction text shaded by the per-part attention increase (steered - base)
# =========================================================================================
# The token layout + per-token shading value is precomputed into ``fig3_token_shading.json`` by
# ``precompute_figure_data.py`` (which uses the o200k_base tokenizer once). Plotting needs no
# tokenizer and no network, so this figure is fully offline like the others.
def fig3_attention_tokens(source=None):
    data = load_figure_json("fig3_token_shading.json", source=source)
    norm = colors.Normalize(vmin=0.0, vmax=data["vmax"])
    cmap = colormaps["Reds"]

    CW, LH, WRAP = 1.0, 2.0, 70
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(12.5, 5.2))
        y = 0.0
        for c in ("bullet", "numbered"):
            ax.text(0, y + 1.0, f"{c} instruction", fontsize=11, fontweight="bold", va="bottom")
            y -= LH * 0.7
            x = 0.0
            for tok in data[c]:
                t, val = tok["t"], max(0.0, tok["v"])
                w = len(t) * CW
                if x + w > WRAP:
                    x = 0.0; y -= LH
                fc = cmap(norm(val))
                ax.add_patch(Rectangle((x, y - 0.45), w, 1.35, fc=fc, ec="none", zorder=1))
                lum = 0.299 * fc[0] + 0.587 * fc[1] + 0.114 * fc[2]
                ax.text(x + 0.05, y + 0.2, t.replace(" ", "\u00a0"), fontsize=11, va="center",
                        ha="left", family="DejaVu Sans Mono",
                        color="white" if lum < 0.5 else "#111111", zorder=2)
                x += w
            y -= LH * 1.7
        ax.set_xlim(-1, WRAP + 1)
        ax.set_ylim(y + LH * 0.6, 2.0)
        ax.axis("off")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        cax = fig.add_axes([0.30, 0.07, 0.40, 0.03])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label("increase in attention to the token, steered \u2212 base "
                     "(per-token average within each instruction part)", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    keys = {c: {g: round(v, 4) for g, v in data["part_values"][c].items()}
            for c in ("bullet", "numbered")}
    return fig, keys


# =========================================================================================
# FIG 4 (supplementary) -- diff-of-means: only the gradient objective (or fine-tuning) unlocks formatting
# =========================================================================================
def fig4_diff_of_means(source=None):
    sd = load_figure_json("steer_deliverable_gL10.json", source=source)
    se = load_figure_json("steer_eval_heldout_analysis.json", source=source)
    rn = load_figure_json("fig4_random_null.json", source=source)
    b = sd["per_instruction"]["bullet"]
    avg_diff = se["per_instruction"]["bullet"]["real"]  # single-layer diff-of-means direction (n=39)

    arms = [
        ("base model", b["base"]["effective_control"] * 100, b["base"]["n"], C_BASE),
        ("average-difference\ndirection", avg_diff["effective_control"] * 100, avg_diff["n"], C_DOM),
        ("random vector\n(same size)", rn["p"] * 100, rn["n"], C_NULL),
        ("gradient-trained\nsteering vector", b["gL10"]["effective_control"] * 100, b["gL10"]["n"], C_VEC),
        ("fine-tuned\n(LoRA)", b["ft"]["effective_control"] * 100, b["ft"]["n"], C_FT),
    ]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for i, (lab, p, n, c) in enumerate(arms):
            lo, hi = _wilson(p / 100, n)
            ax.bar(i, p, 0.62, color=c, edgecolor="#666", lw=0.5)
            ax.errorbar(i, p, yerr=[[(p / 100 - lo) * 100], [(hi - p / 100) * 100]], fmt="none",
                        ecolor="#333", capsize=4, lw=1)
            ax.text(i, p + (3 if p > 1 else 2), f"{p:.0f}%", ha="center", fontsize=10)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels([a[0] for a in arms], fontsize=9.5)
        ax.set_ylabel("bullet-format CoT compliance (%)")
        ax.set_ylim(0, 62)
        ax.set_title("CoT controllability on bullet formatting by intervention")

    keys = {lab.replace("\n", " "): {"pct": round(p, 1), "n": n} for lab, p, n, _ in arms}
    return fig, keys


# =========================================================================================
# Registry + driver
# =========================================================================================
FIGURES = {
    "fig1_headline": fig1_headline,
    "fig2_attention_subspan": fig2_attention_subspan,
    "fig3_attention_tokens": fig3_attention_tokens,
    "fig4_diff_of_means": fig4_diff_of_means,
}


def generate_all(out_dir, source=None, names=None, formats=("png", "pdf"), verbose=True):
    """Regenerate figures ``names`` (default all) to ``out_dir``; return ``{name: key_numbers}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_keys = {}
    for name in (names or list(FIGURES)):
        if name not in FIGURES:
            raise KeyError(f"unknown figure '{name}'. Choices: {list(FIGURES)}")
        fig, keys = FIGURES[name](source=source)
        for ext in formats:
            fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        all_keys[name] = keys
        if verbose:
            print(f"[figures] saved {name} -> {out_dir}")
    return all_keys
