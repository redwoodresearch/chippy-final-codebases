# Research-run scripts (organized for reference)

These are the **curated** training / data-generation / evaluation / mechanism scripts from the
research run, kept here so the pipeline behind the headline results is readable. **They are NOT
invoked by the master notebook or `generate_figures.py`** and they are not needed to reproduce the
figures (which plot saved summary metrics on CPU).

Running them from scratch requires the original heavy infrastructure: a **Modal** H100 GPU container
serving `gpt-oss-20b` (the harness in `harness/gpt_oss_infer.py`), the **Tinker** fine-tuning API,
and a Claude (Anthropic) API key for the LLM judges. The released artifacts on Hugging Face are the
outputs of these scripts.

This is a *subset* of the ~190 scripts in the working run — the ones that matter for the published
results. Each file keeps its module docstring; one-line summaries below.

> These scripts assume the **original flat module layout** of the research run (e.g. `import
> instructions`, `import answer_scoring`, `import gpt_oss_infer`), not the release's `cot_steering/`
> package layout — they are preserved essentially as-run for reference and are not import-compatible with the
> release package as-is. The cleaned, import-ready versions of the instruction suite + accuracy scorer
> live in `cot_steering/instructions.py` and `cot_steering/scoring.py`.

## harness/ — the generation + intervention engine (GPU / Modal)
| file | what it does |
|---|---|
| `gpt_oss_infer.py` | Modal H100 harness: Harmony-format generation for `gpt-oss-20b`, the residual-stream **steering hooks**, and activation/attention **capture + causal-patch** methods. |
| `harmony_utils.py` | Render/parse the Harmony chat format (prompt token rendering for training; parse the `analysis`/`final` channels). |

## training/ — the two interventions
| file | what it does |
|---|---|
| `run_grad_steer_train.py` | **Train the additive steering vector** (the deliverable): gradient descent on the complying-target NLL wrt one residual-stream bias, model weights frozen. Produces `grad_steer_<tag>.npz` (e.g. `gL10`). |
| `grad_steer_lib.py` | Helpers for the steering-vector training (build train sequences, save/load `.npz`, apply-spec). |
| `run_ft_train.py` | Train the rank-32 **LoRA fine-tune** on edited complying reasoning traces (via Tinker), completion-only loss. |
| `ft_data.py` | Load/render the SFT datasets (compliant / raw-trace control / plain) into training tokens with a completion-only loss mask. |
| `ft_merge_modal.py`, `run_ft_merge.py` | Merge the LoRA adapter into servable bf16 weights on a Modal Volume (for steering-hookable inference). |

## data_generation/ — the novel datasets
| file | what it does |
|---|---|
| `run_build_tasks.py` | Build + score the source-stratified task pool (GSM8K, MATH, MMLU-Pro, OpenBookQA, ARC-Challenge, ReasonIF). |
| `build_source_traces.py` | Collect the base model's natural reasoning traces that qualify as edit sources. |
| `build_sft.py` | Build the **edited-reasoning** SFT targets (base traces edited to comply, programmatically or with Claude Opus). |
| `build_control.py` | Build the **raw-trace control** SFT set (same prompts, unedited non-complying traces) for the dissociation control. |
| `build_plain.py` | Build the small "plain" (no-instruction) mix used to keep default behaviour stable. |

## evaluation/ — held-out evals + the deliverable analyses (produce fig1 / fig4 data)
| file | what it does |
|---|---|
| `run_grad_steer_eval.py` | Held-out steering eval (n=100/instruction): base + steering vector + control twin + random-null seeds, FT-matched generation convention. |
| `analyze_steer_deliverable.py` | **Deliverable #2 analysis** → `steer_deliverable_gL10.json` (fig1 steering arm + fig4 bullet bars): per-instruction + aggregate `effective_control`, cluster-bootstrap CIs, paired vector−FT difference. |
| `run_ft_eval.py`, `run_ft_eval_judges.py` | Held-out fine-tune eval + the Claude Opus/Haiku judge pipeline (meta / genuineness / style compliance). |
| `analyze_ft_eval.py` | Core eval scoring: `effective_control` / `raw_compliance` / accuracy row labels (the strict CoT-control metric). |
| `analyze_ft_deliverable.py` | **Deliverable #1 analysis** → `ft_deliverable_cdel_vs_ctrldel.json` (fig1 fine-tune arm): FT vs matched control, per-instruction + aggregate uplift with CIs. |
| `run_steer_eval.py`, `analyze_steer_eval.py` | Single-layer **diff-of-means** ("average-difference direction") held-out eval → `steer_eval_heldout_analysis.json` (fig4 avg-diff bar, n=39). |
| `run_derive_directions.py`, `steering_lib.py` | Derive the diff-of-means steering directions (pooled / per-category) from matched complying-vs-non-complying activations. |
| `steer_eval_lib.py` | Shared steered-evaluation machinery (steering-aware generation, arm bookkeeping). |
| `judges.py` | The Claude Opus/Haiku CoT-compliance + genuineness judges. |

## mechanism/ — fig2 / fig3 data
| file | what it does |
|---|---|
| `run_qkov_patch.py`, `analyze_qkov.py` | Causal patch of the steered attention **pattern** vs **values** → `mech_qkov.json` (the ~71%/62% pattern-vs-value numbers; released, not plotted). |
| `run_tok_subspan.py`, `analyze_tok_subspan.py` | **fig2 / fig3**: per-instruction-part attention (format-specifier / directive / "your reasoning" / rest), base vs steered → `tok_subspan.json` + the per-example `tok_subspan_attn.npz`. |
| `run_tok_verify.py`, `analyze_tok_verify.py` | Broadened (48-task) replication of the pattern-vs-value attribution. |
| `mech_lib.py`, `tok_lib.py` | Shared mechanism helpers (instruction-span location, sub-span decomposition, recruited-head selection). |
