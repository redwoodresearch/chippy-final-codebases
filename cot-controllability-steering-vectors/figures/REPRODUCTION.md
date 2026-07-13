# Figure reproduction — verification

The figures in this release are **regenerated from the released data artifacts by clean code**
(`cot_steering/figures.py`, driven by `generate_figures.py` / the notebook) — never by copying the
reference image/PDF files. This document records the numeric + visual comparison against the
reference figures from the blog post *"A 2,880-number steering vector gives a reasoning model the
chain-of-thought control that fine-tuning does."*

Reproduce + check the numbers yourself:

```bash
python generate_figures.py --verify          # auto-loads figure_data (HF, local fallback), asserts the numbers
python generate_figures.py --source hf --verify     # specifically exercise the Hugging Face load path
python generate_figures.py --source local --verify  # fully offline, committed figure_data/
python tests/test_release.py                 # the full test suite (verify both sources, parity, consistency)
```

`--verify` runs **40 checks**: 28 plotted-value assertions (summarized in the table below) plus 12 structural/relative
checks that guard the headline *messages* (the paired vector−fine-tune CI brackets 0; both uplift CIs
clear +10pp; the format specifier is the most-attended instruction part for both bullet & numbered;
the no-instruction control shows no spurious formatting but does show the verbosity/degeneration
side-effect).

## Figure → data map (each figure's exact source + plotted quantities)

| figure | source artifact(s) | plotted quantity |
|---|---|---|
| **fig1_headline** | `steer_deliverable_gL10.json`, `ft_deliverable_cdel_vs_ctrldel.json` | (L) per-held-out-instruction strict CoT-control compliance (`effective_control`) for base / fine-tune / steering vector, n=100 each, 9 held-out instructions; (R) aggregate uplift over base for FT and the vector with 95% cluster-bootstrap CIs, the paired vector−FT difference, and the +10pp dashed line |
| **fig2_attention_subspan** | `fig2_subspan_attention.json` (derived from `tok_subspan_attn.npz` per-example tensors) | attention onto each instruction part (format-specifier / "your reasoning" / directive verbs / other), base vs steered, bullet & numbered, pooled over recruited late heads, bootstrap CIs |
| **fig3_attention_tokens** | `fig3_token_shading.json` (precomputed from `tok_subspan.json` + the o200k_base token layout) | the bullet & numbered instruction text, each token shaded by the per-part average attention increase (steered − base) |
| **fig4_diff_of_means** | `steer_deliverable_gL10.json` (bullet base/vector/FT), `steer_eval_heldout_analysis.json` (average-difference direction, n=39), `fig4_random_null.json` (random vectors, derived) | held-out bullet `effective_control` for base / average-difference direction / random vector / gradient-trained vector / fine-tune, Wilson intervals |

Three figure-data files are not verbatim summaries; they are **derived from the raw run artifacts on
CPU at ~$0** by `precompute_figure_data.py` (no model generation): the fig2 base-vs-steered per-part
attention CIs (from the per-example `tok_subspan_attn.npz`); the fig3 per-token shading layout (the
o200k_base tokenization of the two instructions + each part's per-token-average delta from
`tok_subspan.json`); and the fig4 random-vector bullet bar (`effective_control` of five random
matched-norm vectors, pooled 0/500 → 0.0%, recomputed from the raw judged generations with the
project's strict `effective_control` definition). Doing the fig3 tokenization at precompute time keeps
the plot path **offline** (no tokenizer download); `tiktoken` is only needed to re-run precompute.

## Numeric verification (`generate_figures.py --verify`, all PASS)

| figure | quantity | regenerated | reference |
|---|---|--:|--:|
| fig1 | aggregate: base / FT / vector | 1.6 / 13.9 / 14.3 % | 1.6 / 13.9 / 14.3 % |
| fig1 | bullet: base / FT / vector | 0 / 52 / 48 % | 0 / 52 / 48 % |
| fig1 | aggregate uplift: FT | +12.3pp [+10.4,+14.2] | +12.3pp [+10.4,+14.2] |
| fig1 | aggregate uplift: steering vector | +12.8pp [+10.7,+14.9] | +12.8pp [+10.7,+14.9] |
| fig1 | paired vector − FT difference | +0.4pp [−1.9,+2.8] | +0.4pp [−1.9,+2.8] |
| fig2 | bullet format-specifier attention: base → steered | 2.29 → 6.52 | 2.3 → 6.5 |
| fig2 | numbered format-specifier attention: base → steered | 3.95 → 8.59 | 3.9 → 8.6 |
| fig3 | bullet per-part Δattn: spec > "your reasoning" | spec 0.39 ≫ cot −0.05 | specifier darkest |
| fig4 | bullet: base / avg-diff(n=39) / random / vector / FT | 0 / 0 / 0 / 48 / 52 % | 0 / 0 / 0 / 48 / 52 % |

All 28 asserted key numbers match the published reference (see `generate_figures.py --verify`); the
remaining table rows above are read directly from the released summaries and are consistent with them.

## Visual comparison

The regenerated figures were eyeballed side-by-side against the reference images: same bars,
ordering, axes, tick labels, legends, titles, error bars, and the same qualitative message.
Differences are **cosmetic only** (exact font/DPI/whitespace), as expected for a clean re-plot.
fig1 follows the published post layout (a single horizontal bar chart with the aggregate as the
first bar); fig3 uses the token-level attention artifacts (the published per-token attention figure).

## Honesty notes

- fig3: per-token attention was logged only for the bullet instruction at the format onset; both the
  bullet and numbered instructions are shaded by their instruction part's per-token-average increase
  (parts located by substring matching), exactly as in the published figure. This is the one figure
  whose shading is a part-average rather than a literal per-token value for every token — stated in
  the figure caption and the blog post. The token→part assignment is precomputed into
  `fig3_token_shading.json` (the two instruction strings are pure ASCII, so the o200k_base
  token→char alignment is exact).
- fig2 bars carry 95% bootstrap CIs over examples (deterministic, seed 0). The bullet/numbered
  "total instruction attention" (12.3→18.4 / 12.8→18.5 in `tok_subspan.json`) is consistent with
  the plotted per-part bars but is not one of the asserted `--verify` checks (which cover the
  format-specifier sub-span).
- Provenance: `precompute_figure_data.py` recomputes fig1's base/vector bars, the no-instruction
  control, and fig4's base/vector/average-difference bars **from the per-row judged generations**
  in `results_raw/` (re-scoring each generation's text with the released scorers; judge verdicts
  are cached in the rows) and asserts they equal the summaries. The fine-tune arm and the
  cluster-bootstrap CIs are taken from the released summaries (the FT per-row file is not in the
  release).
- No reproduction gap: every plotted quantity is regenerated from the released artifacts and matches
  the reference numerically and visually.
