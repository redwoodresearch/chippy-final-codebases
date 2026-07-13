#!/usr/bin/env python3
"""Derive the small ``figure_data/`` summary artifacts that ``generate_figures.py``
and the master notebook consume to regenerate fig1-fig4.

Most figure inputs are *already* small saved summaries from the research run and are
simply copied through unchanged (they are the canonical, verified artifacts):

  * ``steer_deliverable_gL10.json``            -- fig1 (steering arm + the no-instruction
    control) + fig4 (bullet bars)
  * ``mech_qkov.json``                         -- the pattern-vs-value patching summary
    (the ~71%/62% patching numbers quoted in the post; released but no longer plotted)
  * ``ft_deliverable_cdel_vs_ctrldel.json``    -- fig1 (fine-tune arm)
  * ``steer_eval_heldout_analysis.json``       -- fig4 (average-difference direction, n=39)
  * ``tok_subspan.json``                       -- fig3 (per-token attention deltas)

Three figure inputs are NOT in any saved summary, so this script derives them from the
raw run artifacts on CPU at ~$0 (no model generation):

  * ``fig2_subspan_attention.json`` -- fig2 plots *base vs steered* attention onto each
    instruction part with per-example bootstrap CIs. The per-example attention tensors
    live in ``tok_subspan_attn.npz`` (288 examples x 24 layers x 64 heads; each value is the
    attention onto a span, already summed over the span's tokens). We reduce them exactly as
    the reference plot did (recruited late layers, sum over heads, mean over late layers) and
    bootstrap a 95% CI over examples.

  * ``fig3_token_shading.json`` -- fig3 shades each instruction token by the per-token-average
    attention increase (steered - base) of the instruction part it belongs to. We tokenize the
    two instructions with o200k_base, assign each token to its part by substring matching, and
    divide each part's total attention delta (from ``tok_subspan.json``) by its token count. This
    is the one place the o200k_base tokenizer is used; doing it here keeps the plot path offline.

  * ``fig4_random_null.json`` -- fig4's "random vector (same size)" bar is the held-out
    bullet ``effective_control`` of five random matched-norm vectors. We recompute it
    from the raw judged generations (``grad_steer_eval_deliverable_delivnull_judged.jsonl``)
    using the project's ``effective_control`` definition. If that raw file is absent we
    fall back to the value recorded in the project write-up (0/500 -> 0.0%).

Run once to (re)build ``figure_data/``::

    python precompute_figure_data.py --raw-dir /path/to/research/results

``--raw-dir`` defaults to the published Hugging Face ``results_raw`` snapshot if not
given (downloaded via ``huggingface_hub``); see README for the repo id.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from cot_steering.artifacts import RAW_ROW_FILES, RAW_SUMMARY_FILES

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "figure_data"

# Small summaries copied through verbatim (the canonical research artifacts).
COPY_FILES = RAW_SUMMARY_FILES

# Recruited late attention layers used for the fig2 sub-span reduction (odd / full-attention
# layers 13..23). Matches the reference plotting code exactly.
LATE_LAYERS = [13, 15, 17, 19, 21, 23]
FIG2_SPANS = ["spec", "cot_target", "directive", "rest"]
FIG2_CONDS = ["bullet", "numbered"]
BOOT_N = 2000
BOOT_SEED = 0


def _boot_ci(x, n: int = BOOT_N, seed: int = BOOT_SEED):
    """Mean and 95% bootstrap CI over examples (deterministic for fixed seed)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    x = x[~np.isnan(x)]
    boots = [float(np.mean(rng.choice(x, len(x), replace=True))) for _ in range(n)]
    return float(np.mean(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def derive_fig2_subspan(raw_dir: Path) -> dict:
    """base-vs-steered attention onto each instruction sub-span, per-example bootstrap CIs."""
    import numpy as np
    npz = np.load(raw_dir / "tok_subspan_attn.npz")
    meta = json.loads((raw_dir / "tok_subspan_attn_meta.json").read_text())
    conds = np.array(meta["conds"])

    def per_example(arm: str, span: str):
        # arrays are [n_examples, n_layers(24), n_heads(64)] of attention mass onto the span.
        a = npz[f"{arm}_{span}"][:, LATE_LAYERS, :]
        return np.nansum(a, axis=2).mean(axis=1)  # sum over heads' span-mass, mean over late layers

    out: dict = {}
    for cond in FIG2_CONDS:
        mask = conds == cond
        out[cond] = {"n": int(mask.sum()), "spans": {}}
        for span in FIG2_SPANS:
            entry = {}
            for arm in ("base", "steer"):
                m, lo, hi = _boot_ci(per_example(arm, span)[mask])
                entry[arm] = {"mean": m, "lo": lo, "hi": hi}
            out[cond]["spans"][span] = entry
    return out


# --- fig3: tokenize the instruction text + assign each token to its instruction part ----------
# The instruction TEXT comes from the released instruction suite (cot_steering.instructions) so it
# cannot drift from the wording the model was actually shown; only the part->substring definitions
# (which words belong to the format specifier / directive / etc.) live here.
_COT_SUBS = ["chain of thought", "step-by-step reasoning", "your reasoning", "reasoning step", "each step"]
_PART_DEFS = {
    "bullet": {"spec": ["bulleted list", "bulleted", "'- '", "a hyphen and a space", "hyphen"],
               "cot": _COT_SUBS, "dir": ["write", "Every line", "must start with"]},
    "numbered": {"spec": ["numbered list", "numbered", "a number followed by a period",
                          "(1. , 2. , 3. , ...)", "1.", "2.", "3.", "number followed by a period"],
                 "cot": _COT_SUBS, "dir": ["write", "Every line", "must start with"]},
}
_PART_PRIORITY = ["spec", "cot", "dir"]
_PART_NAME = {"spec": "spec", "cot": "cot_target", "dir": "directive"}


def _char_set(text, subs):
    cs = set()
    for sub in subs:
        i = text.find(sub)
        while i != -1:
            cs.update(range(i, i + len(sub)))
            i = text.find(sub, i + 1)
    return cs


def _assign_tokens(cond, text, enc):
    """[(token_text, part_name), ...] for the instruction; parts located by substring matching.

    Char offsets are reconstructed by summing per-token decoded lengths; this is exact because the
    two instruction strings are pure ASCII (one decoded char per source char).
    """
    ids = enc.encode(text)
    toks, spans, pos = [], [], 0
    for i in ids:
        t = enc.decode([i])
        toks.append(t); spans.append((pos, pos + len(t))); pos += len(t)
    cby = {g: _char_set(text, _PART_DEFS[cond][g]) for g in _PART_PRIORITY}
    out = []
    for t, (c0, c1) in zip(toks, spans):
        chars = set(range(c0, c1)); part = "rest"
        for g in _PART_PRIORITY:
            if chars & cby[g]:
                part = _PART_NAME[g]; break
        out.append((t, part))
    return out


def derive_fig3(raw_dir: Path) -> dict:
    """Per-token shading: each instruction token + its part's per-token-average attention increase."""
    import tiktoken
    from cot_steering import instructions as I  # the exact instruction wording the model was shown
    enc = tiktoken.get_encoding("o200k_base")
    mass = json.loads((raw_dir / "tok_subspan.json").read_text())["attn_mass"]
    out: dict = {"part_values": {}}
    vmax = 0.0
    for cond in ("bullet", "numbered"):
        toks = _assign_tokens(cond, I.INSTRUCTIONS[cond].prompt_text, enc)
        counts: dict = {}
        for _, g in toks:
            counts[g] = counts.get(g, 0) + 1
        per_tok = {g: mass[cond][g]["delta"] / counts[g] for g in counts}
        out[cond] = [{"t": t, "v": per_tok[g]} for t, g in toks]
        out["part_values"][cond] = per_tok
        vmax = max(vmax, max(per_tok.values()))
    out["vmax"] = vmax
    return out


def _max_ngram_repeat(tokens, n):
    from collections import Counter
    if len(tokens) < n + 1:
        return 1
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(grams).most_common(1)[0][1]


def _is_degenerate(text, normalize):
    """Vendored copy of the research run's degenerate/looping-trace detector (sft_edit.py)."""
    import re
    toks = re.findall(r"\S+", normalize(text).lower())
    if len(toks) < 60:
        return False
    if _max_ngram_repeat(toks, 10) >= 5:
        return True
    if len(toks) >= 150:
        tris = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        if tris and len(set(tris)) / len(tris) < 0.4:
            return True
    return False


def _effective_row(row, instructions_mod):
    """Replicate the project's strict ``effective_control`` row label (see analyze_ft_eval.py)."""
    instr = instructions_mod.INSTRUCTIONS.get(row["condition"])
    if instr is None or row.get("genuine") is None:
        return None
    if row.get("malformed"):
        rc = False
    elif instr.scorer is not None:
        rc = bool(instr.scorer(row.get("analysis", "")))
    else:
        rc = bool(row.get("judged_compliant"))
    if not rc or row.get("malformed") or row.get("truncated"):
        return False
    if row.get("genuine") is False:
        return False
    if _is_degenerate(row.get("analysis", ""), instructions_mod.normalize):
        return False
    if row.get("meta") is None:
        return None
    if row.get("meta") is True:
        return False
    return True


def derive_fig4_random_null(raw_dir: Path) -> dict:
    """Held-out bullet effective_control of the five random matched-norm vectors.

    Recomputed from the raw judged generations using the project's strict ``effective_control``
    definition (the bullet scorer comes from the released ``cot_steering.instructions`` module, so
    this is self-contained from a bare Hugging Face ``results_raw`` snapshot).
    """
    fn = raw_dir / "grad_steer_eval_deliverable_delivnull_judged.jsonl"
    if not fn.exists():
        return {"p": 0.0, "n": 100, "source": "recorded (raw judged file not provided)",
                "note": "five random matched-norm vectors; pooled bullet effective_control = 0/500"}
    from cot_steering import instructions as I  # the released instruction suite + scorers
    vals = []
    with open(fn) as f:
        for line in f:
            r = json.loads(line)
            if r.get("condition") == "bullet":
                e = _effective_row(r, I)
                if e is not None:
                    vals.append(bool(e))
    p = (sum(vals) / len(vals)) if vals else 0.0
    return {"p": p, "n": 100, "n_pooled": len(vals), "source": "recomputed from delivnull judged jsonl",
            "note": "five random matched-norm vectors pooled; bar shown at n=100 Wilson interval"}


def _load_rows(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def recompute_headline_from_rows(raw_dir: Path) -> int:
    """Recompute the headline numbers from the per-row judged generations; assert vs summaries.

    Every generation's ``effective_control`` label is recomputed from the row's generation text
    with the released scorers (plus the row's cached judge verdicts), then aggregated and compared
    to the released summary JSONs. Covers fig1's base + steering-vector bars, the no-instruction
    control, and fig4's base / vector / average-difference bars. The **fine-tune arm** is the one
    summary-trusted number: its per-row file is not in the release. Returns #values checked.
    """
    from cot_steering import instructions as I

    checked = 0

    def eq(got, want, label):
        nonlocal checked
        assert abs(got - want) < 1e-9, f"headline recompute mismatch: {label}: rows {got} vs summary {want}"
        checked += 1

    sd = json.loads((raw_dir / "steer_deliverable_gL10.json").read_text())
    rows = _load_rows(raw_dir / "grad_steer_eval_deliverable_deliv_judged.jsonl")

    # fig1 per-instruction effective_control, base + gL10
    acc: dict = {}
    for r in rows:
        if r["condition"] == "none":
            continue
        e = _effective_row(r, I)
        if e is not None:
            acc.setdefault((r["arm"], r["condition"]), []).append(bool(e))
    for arm in ("base", "gL10"):
        for cond, entry in sd["per_instruction"].items():
            vals = acc[(arm, cond)]
            eq(sum(vals) / len(vals), entry[arm]["effective_control"], f"{arm}/{cond}")

    # the no-instruction control (fig1 keys + the --verify predicates)
    for arm in ("base", "gL10"):
        rs = [r for r in rows if r["condition"] == "none" and r["arm"] == arm]
        s = sd["none"][arm]
        eq(sum(bool(I.INSTRUCTIONS["bullet"].scorer(r["analysis"])) for r in rs), s["bullets"], f"none/{arm}/bullets")
        eq(sum(bool(I.INSTRUCTIONS["numbered"].scorer(r["analysis"])) for r in rs), s["numbered"], f"none/{arm}/numbered")
        eq(sum((I.uppercase_fraction(r["analysis"]) or 0) > 0.5 for r in rs), s["upper_gt_50"], f"none/{arm}/upper")
        eq(sum(_is_degenerate(r["analysis"], I.normalize) for r in rs) / len(rs), s["degenerate"], f"none/{arm}/degenerate")
        eq(sum(r["analysis_words"] for r in rs) / len(rs), s["aw_mean"], f"none/{arm}/aw_mean")

    # fig4 average-difference arm (held-out bullet, n=39)
    se = json.loads((raw_dir / "steer_eval_heldout_analysis.json").read_text())
    hrows = _load_rows(raw_dir / "steer_eval_heldout_judged.jsonl")
    vals = [_effective_row(r, I) for r in hrows if r["arm"] == "real" and r["condition"] == "bullet"]
    vals = [bool(v) for v in vals if v is not None]
    b = se["per_instruction"]["bullet"]["real"]
    eq(sum(vals) / len(vals), b["effective_control"], "heldout/real/bullet")
    eq(len(vals), b["n"], "heldout/real/bullet n")
    return checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", type=str, default=None,
                    help="Directory holding the raw run artifacts (results/). "
                         "If omitted, downloads the published HF results_raw snapshot.")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help="Where to write the derived figure_data files (default: ./figure_data; "
                         "point elsewhere to re-derive without overwriting the committed copies).")
    args = ap.parse_args()
    OUT = Path(args.out)

    if args.raw_dir:
        raw_dir = Path(args.raw_dir).resolve()
    else:
        from cot_steering.artifacts import ensure_results_raw  # lazy import
        raw_dir = ensure_results_raw()
    print(f"[precompute] raw-dir = {raw_dir}")

    OUT.mkdir(parents=True, exist_ok=True)
    for fn in COPY_FILES:
        src = raw_dir / fn
        if not src.exists():
            raise FileNotFoundError(f"required summary missing: {src}")
        shutil.copyfile(src, OUT / fn)
        print(f"[precompute] copied {fn}")

    fig2 = derive_fig2_subspan(raw_dir)
    (OUT / "fig2_subspan_attention.json").write_text(json.dumps(fig2, indent=2))
    print("[precompute] wrote fig2_subspan_attention.json")

    fig3 = derive_fig3(raw_dir)
    (OUT / "fig3_token_shading.json").write_text(json.dumps(fig3, indent=2))
    print("[precompute] wrote fig3_token_shading.json")

    fig4 = derive_fig4_random_null(raw_dir)
    (OUT / "fig4_random_null.json").write_text(json.dumps(fig4, indent=2))
    print(f"[precompute] wrote fig4_random_null.json ({fig4['source']})")

    if all((raw_dir / fn).exists() for fn in RAW_ROW_FILES):
        n = recompute_headline_from_rows(raw_dir)
        print(f"[precompute] headline recompute from per-row judged records: {n} values match the summaries")
    else:
        print("[precompute] per-row judged files not in raw-dir -- skipped the headline recompute check")

    print("[precompute] done.")


if __name__ == "__main__":
    main()
