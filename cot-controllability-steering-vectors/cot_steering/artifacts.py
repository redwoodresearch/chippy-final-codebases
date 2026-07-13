"""Locate the released artifacts the figures and notebook need.

By default we **load the small figure-summary JSONs from Hugging Face** (per the
publication request) and fall back to the copies committed under ``figure_data/`` so the
notebook still runs fully offline. Set ``COT_ARTIFACT_SOURCE=local`` (or pass
``source="local"``) to force the local copies, or ``=hf`` to require Hugging Face.

Hugging Face layout (user ``ejcgan``):

  dataset repo ``cot-controllability-steering-vectors``
    figure_data/<name>.json     small summaries the figures plot (this loader)
    steering_vectors/<tag>.npz  the steering vectors (gL10 + family)
    datasets/<file>             the novel datasets (task pool, SFT data, ...)
    results_raw/<file>          large raw artifacts

  model repo  ``cot-controllability-gpt-oss-20b-lora``  (the LoRA fine-tune, optional)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HF_DATASET_REPO = "ejcgan/cot-controllability-steering-vectors"
HF_MODEL_REPO = "ejcgan/cot-controllability-gpt-oss-20b-lora"

_PKG_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_FIGURE_DATA = _PKG_ROOT / "figure_data"

# Canonical lists of the raw artifacts needed to *re-derive* figure_data/ (used by
# precompute_figure_data.py and ensure_results_raw). Defined once here so they cannot drift.
#  - RAW_SUMMARY_FILES: small summaries copied through verbatim (the canonical research artifacts).
#  - RAW_DERIVE_INPUTS: the large raw artifacts the two derived summaries are computed from.
RAW_SUMMARY_FILES = [
    "steer_deliverable_gL10.json",
    "ft_deliverable_cdel_vs_ctrldel.json",
    "steer_eval_heldout_analysis.json",
    "mech_qkov.json",
    "tok_subspan.json",
]
RAW_DERIVE_INPUTS = [
    "tok_subspan_attn.npz",            # fig2 per-example attention tensors
    "tok_subspan_attn_meta.json",
    "grad_steer_eval_deliverable_delivnull_judged.jsonl",  # fig4 random-null per-row generations
]
# Per-row judged generations for the headline evals. Not needed to plot, but
# precompute_figure_data.py recomputes the headline numbers from these and asserts they
# equal the released summaries (judge verdicts are cached in the rows).
RAW_ROW_FILES = [
    "grad_steer_eval_deliverable_deliv_judged.jsonl",   # fig1 base/vector + no-instruction control
    "steer_eval_heldout_judged.jsonl",                  # fig4 average-difference arm (n=39)
]


def _source(source: str | None) -> str:
    return (source or os.environ.get("COT_ARTIFACT_SOURCE", "auto")).lower()


def _hf_download_figure_data(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=f"figure_data/{name}",
        token=token,
    )
    return Path(path)


def figure_data_path(name: str, source: str | None = None) -> Path:
    """Return a local filesystem path to figure-summary file ``name``.

    ``source`` is one of ``auto`` (HF then local), ``hf`` (HF only) or ``local``.
    """
    src = _source(source)
    local = _LOCAL_FIGURE_DATA / name
    if src == "local":
        if not local.exists():
            raise FileNotFoundError(f"local figure_data missing: {local}")
        return local
    if src == "hf":
        return _hf_download_figure_data(name)
    # auto: try HF, fall back to the committed local copy.
    try:
        return _hf_download_figure_data(name)
    except Exception as exc:  # network / auth / repo-not-yet-created
        if local.exists():
            return local
        raise FileNotFoundError(
            f"could not fetch '{name}' from Hugging Face ({exc}) and no local copy at {local}"
        ) from exc


def load_figure_json(name: str, source: str | None = None) -> Any:
    """Load and parse a figure-summary JSON by file name."""
    return json.loads(figure_data_path(name, source=source).read_text())


def steering_vector_path(tag: str = "gL10", source: str | None = None) -> Path:
    """Local path to a steering-vector ``.npz`` (``gL10`` is the headline vector).

    Tries Hugging Face first (unless ``source='local'``); falls back to a ``data/``
    directory next to the package if present.
    """
    src = _source(source)
    fname = f"grad_steer_{tag}.npz"
    local = _PKG_ROOT / "data" / fname
    if src != "local":
        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(
                repo_id=HF_DATASET_REPO, repo_type="dataset",
                filename=f"steering_vectors/{fname}", token=os.environ.get("HF_TOKEN")))
        except Exception:
            if src == "hf":
                raise
    if local.exists():
        return local
    raise FileNotFoundError(f"steering vector '{tag}' not found (HF or {local})")


def ensure_results_raw(filenames: list[str] | None = None) -> Path:
    """Download the raw artifacts needed to *re-derive* figure_data and return their dir.

    Used by ``precompute_figure_data.py``. Downloads the per-example attention tensors
    (``tok_subspan_attn.npz`` + meta), the random-null judged generations, and the verbatim
    summaries, all into one ``results_raw/`` snapshot directory.
    """
    from huggingface_hub import hf_hub_download

    if filenames is None:
        filenames = RAW_DERIVE_INPUTS + RAW_ROW_FILES + RAW_SUMMARY_FILES
    token = os.environ.get("HF_TOKEN")
    out_dir = None
    for fn in filenames:
        p = Path(hf_hub_download(
            repo_id=HF_DATASET_REPO, repo_type="dataset",
            filename=f"results_raw/{fn}", token=token))
        out_dir = p.parent
    assert out_dir is not None
    return out_dir
