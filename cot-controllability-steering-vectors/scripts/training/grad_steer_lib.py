"""Helpers for the gradient-trained additive steering bias.

A gradient-trained intervention is "fine-tuning, but the only thing it can change is a fixed additive
bias in the residual stream": for each TRAIN (task x instruction) example we teacher-force
``prompt + complying target`` and minimize the completion-only NLL (prompt masked) wrt the steering
vector(s), with the MODEL WEIGHTS FROZEN (so no weight gradients -> memory modest). Trained on TRAIN
instructions ONLY, so held-out (esp. formatting/bullet) is a genuine generalization test mirroring
FT and the diff-of-means.

Vectors persist to ``data/grad_steer_<tag>.npz`` (layers + per-layer vector) + a provenance JSON.
At apply time the vector goes straight into the existing steering path:
    steering = [{"layer": L, "vector": vec.tolist()} for L, vec in zip(layers, vectors)]
(no per-layer-norm rescaling — the gradient learns the magnitude itself).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

import ft_data
import instructions as I

DATA = Path("data")
RESULTS = Path("results")
TRAIN_INSTRS = [i.id for i in I.INSTRUCTIONS.values() if i.split == "train"]


def load_train_rows(which: str, match_control: bool):
    """Compliant or control SFT rows restricted to TRAIN instructions. If match_control, restrict
    to the prompts SHARED by both sets (so compliant-trained vs control-trained differ ONLY in the
    target compliance — the steering analogue of the FT raw-trace control)."""
    rows = ft_data.load_rows(which)
    rows = [r for r in rows if r["instruction_id"] in TRAIN_INSTRS]
    if match_control:
        pairs = {(r["task_id"], r["instruction_id"])
                 for r in ft_data.load_rows("control")
                 if r["instruction_id"] in TRAIN_INSTRS}
        rows = [r for r in rows if (r["task_id"], r["instruction_id"]) in pairs]
    return sorted(rows, key=lambda r: (r["instruction_id"], r["task_id"]))


def build_sequences(rows, max_length: int):
    """Return (sequences, comp_starts, metas, n_dropped). sequence = prompt+completion token ids
    (the FT rendering convention); comp_start = len(prompt) (completion-only loss span)."""
    seqs, starts, metas, dropped = [], [], [], 0
    for r in rows:
        pt, ct = ft_data.render_full_tokens(r)
        if len(pt) + len(ct) > max_length:
            dropped += 1
            continue
        seqs.append(pt + ct)
        starts.append(len(pt))
        metas.append({"task_id": r["task_id"], "instruction_id": r["instruction_id"],
                      "category": r["category"]})
    return seqs, starts, metas, dropped


def data_hash(seqs, starts, max_length: int) -> str:
    h = hashlib.sha256()
    h.update(str(max_length).encode())
    for s, cs in zip(seqs, starts):
        h.update(np.asarray(s, dtype=np.int64).tobytes())
        h.update(str(int(cs)).encode())
    return h.hexdigest()[:16]


def save_vectors(tag: str, layers, vectors, meta: dict):
    arrays = {"layers": np.asarray(layers, dtype=np.int32)}
    for i, v in enumerate(vectors):
        arrays[f"vec_{i}"] = np.asarray(v, dtype=np.float32)
    os.makedirs(DATA, exist_ok=True)
    np.savez(DATA / f"grad_steer_{tag}.npz", **arrays)
    json.dump(meta, open(DATA / f"grad_steer_{tag}_meta.json", "w"), indent=2)


def load_vectors(tag: str):
    z = np.load(DATA / f"grad_steer_{tag}.npz")
    layers = [int(L) for L in z["layers"]]
    vectors = [z[f"vec_{i}"] for i in range(len(layers))]
    meta = json.load(open(DATA / f"grad_steer_{tag}_meta.json"))
    return layers, vectors, meta


def make_steering(tag: str):
    """Build steering=[{layer, vector}] from a persisted trained-vector tag, + total ||vector||."""
    layers, vectors, meta = load_vectors(tag)
    steering = [{"layer": int(L), "vector": np.asarray(v, dtype=np.float32).tolist()}
                for L, v in zip(layers, vectors)]
    total_norm = float(np.sqrt(sum(float(np.linalg.norm(v)) ** 2 for v in vectors)))
    return steering, total_norm, layers, vectors


def make_scaled_steering(tag: str, scale: float):
    """Like make_steering but scales every per-layer vector by ``scale`` (for the dose-response
    magnitude sweep). scale=1.0 reproduces the trained vector; scale=0.0 == base (no steering)."""
    layers, vectors, _ = load_vectors(tag)
    steering = [{"layer": int(L), "vector": (scale * np.asarray(v, dtype=np.float32)).tolist()}
                for L, v in zip(layers, vectors)]
    total_norm = float(scale) * float(np.sqrt(sum(float(np.linalg.norm(v)) ** 2 for v in vectors)))
    return steering, total_norm, layers, vectors


def param_count(vectors) -> int:
    return int(sum(np.asarray(v).size for v in vectors))
