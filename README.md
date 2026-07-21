# Automation final codebases

Reproducible release packages for research projects run on Redwood Research's
automated research scaffold. Each project folder is self-contained: minimal cleaned code, a
master Jupyter notebook, and a figure-generation entry point that regenerate the project's main
results on CPU from artifacts hosted on Hugging Face, plus the curated research-run scripts for
reference.

## Projects

- **[`cot-controllability-steering-vectors/`](cot-controllability-steering-vectors/)** —
  *Activation steering can increase chain-of-thought controllability* (`gpt-oss-20b`). A single
  frozen-weights steering vector (2,880 numbers added to one layer's residual stream) matches what
  a LoRA fine-tune does to the model's CoT controllability on held-out instructions, and works by
  raising the late attention heads' attention onto the in-context instruction — concentrated on
  the format-specifier tokens. The package regenerates the three main figures (plus a
  supplementary difference-of-means comparison) from released artifacts on CPU, recomputes the
  headline numbers from per-row judged generations, and verifies every plotted value against the
  published reference.

## Layout

Each project directory carries its own `README.md` (results, quickstart, artifact index),
`LICENSE`, tests, and a `figures/REPRODUCTION.md` documenting the numeric verification against
the published figures.
