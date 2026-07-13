"""Load a steering vector and apply it to a model's residual stream.

The headline artifact is ``grad_steer_gL10.npz`` -- a single 2,880-dimensional vector added
to layer 10's residual stream of ``gpt-oss-20b`` with **zero weight change**. Each ``.npz``
stores ``layers`` (the resid_post layer indices) and ``vec_<i>`` (one float32 vector per layer);
a sibling ``*_meta.json`` records the training provenance (norm, loss, config, git hash).

This module is intentionally framework-light: :func:`load_steering_vector` needs only numpy and
returns plain arrays. :func:`make_residual_steering_hook` is a readable reference for *how* the
vector is applied at inference -- a PyTorch forward hook on ``model.model.layers[L]`` that adds
``vector`` to the layer's residual-stream output (resid_post) on every forward pass, so it fires
on the prefill AND on every decode step. (The research run applies the same hook inside a Modal
GPU harness; that GPU harness is *not* needed to regenerate the figures, which plot saved metrics.)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_steering_vector(path_or_tag, data_dir: str | Path | None = None):
    """Load a steering vector ``.npz``.

    Accepts a direct path to a ``grad_steer_<tag>.npz`` file, or a bare ``tag`` (e.g. ``"gL10"``)
    resolved against ``data_dir`` / Hugging Face (via :mod:`cot_steering.artifacts`).

    Returns ``(layers, vectors, meta)`` where ``layers`` is a list of int layer indices,
    ``vectors`` is a list of ``float32`` arrays (one per layer, shape ``[hidden]``), and ``meta``
    is the provenance dict (or ``{}`` if the meta JSON is absent).
    """
    p = Path(path_or_tag)
    if p.suffix == ".npz" and p.exists():
        npz_path = p
    else:
        tag = str(path_or_tag)
        if data_dir is not None:
            npz_path = Path(data_dir) / f"grad_steer_{tag}.npz"
        else:
            from .artifacts import steering_vector_path
            npz_path = steering_vector_path(tag)
    z = np.load(npz_path)
    layers = [int(L) for L in z["layers"]]
    vectors = [np.asarray(z[f"vec_{i}"], dtype=np.float32) for i in range(len(layers))]
    meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return layers, vectors, meta


def make_residual_steering_hook(vector, attention_mask=None):
    """Reference PyTorch forward hook that adds ``vector`` to a layer's residual-stream output.

    Register it on the target decoder layer::

        import torch
        layers, vectors, _ = load_steering_vector("gL10")
        vec = torch.tensor(vectors[0])                       # layer-10 vector, shape [hidden]
        handle = model.model.layers[layers[0]].register_forward_hook(
            make_residual_steering_hook(vec))
        # ... model.generate(...) ; the CoT now obeys the in-context format instruction ...
        handle.remove()

    During prefill the hidden states are padded; if an ``attention_mask`` ``[B, T]`` is supplied we
    add the vector only at real (non-pad) positions, which makes steering padding-invariant. During
    decode (KV cache, ``seq_len == 1``) the single new token is always real.
    """
    import torch

    vec = torch.as_tensor(vector)

    def hook(module, args, output):
        hs = output[0] if isinstance(output, tuple) else output
        add = vec.to(hs.dtype).to(hs.device)
        if attention_mask is not None and hs.shape[1] == attention_mask.shape[1] and hs.shape[1] > 1:
            m = attention_mask.to(hs.dtype).unsqueeze(-1)  # [B, T, 1]
            hs = hs + add * m
        else:
            hs = hs + add
        return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

    return hook
