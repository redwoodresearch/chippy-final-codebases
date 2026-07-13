"""Parametrized Modal app to CPU-MERGE a Tinker LoRA adapter into an MXFP4 gpt-oss-20b HF
checkpoint on the shared volume. Kept separate from the orchestrator so Modal's remote
import only pulls container-safe deps (no local harmony_utils / gpt_oss_infer).

Why CPU: build_hf_model re-quantizes the MoE experts back to MXFP4, which OOMs on GPU while the model
is loaded. The merged MXFP4 dir (~13.7GB) then loads into the standard HF harness with
Mxfp4Config(dequantize=True) — same recipe as the GptOss eval/steering harness.

The adapter is uploaded to the volume (under /cache/ft_adapters/<tag>) by the orchestrator and read
from there (the 922MB rank-32 adapter is too big to bake into the image).
"""
from __future__ import annotations

import os

import modal

BASE_MODEL = "openai/gpt-oss-20b"

merge_app = modal.App("gpt-oss-ft-merge")
hf_cache_vol = modal.Volume.from_name("gpt-oss-hf-cache", create_if_missing=True)

# tinker_cookbook==0.4.1 needs transformers>=4.57.6 -> don't pin transformers (let pip resolve).
merge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0", "triton==3.4.0", "kernels==0.10.3", "peft", "safetensors",
        "huggingface_hub[hf_transfer]", "tinker", "tinker_cookbook==0.4.1",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _merge_complete(merged_path) -> bool:
    """A merge is complete only if config.json + a real (non-.tmp) safetensors are present.
    (A leftover `shard-*.tmp.safetensors` from a crashed/interrupted merge must NOT count as done.)"""
    if not os.path.isdir(merged_path):
        return False
    files = os.listdir(merged_path)
    has_cfg = "config.json" in files
    has_real_st = any(f.endswith(".safetensors") and not f.endswith(".tmp.safetensors")
                      for f in files)
    return has_cfg and has_real_st


@merge_app.function(image=merge_image, cpu=8.0, memory=131072,
                    volumes={"/cache": hf_cache_vol}, timeout=5400)
def merge_to_volume(adapter_vol_path: str, merged_path: str, clobber: bool = False) -> dict:
    """CPU-only merge -> writes the merged MXFP4 HF model to the shared volume.

    Idempotent: if merged_path already has a COMPLETE merge (config.json + a real safetensors),
    returns without re-merging. A partial/interrupted merge (only a *.tmp.safetensors) does NOT
    count as complete (that bug served s_e2 a broken checkpoint). clobber=True forces re-merge."""
    import shutil
    import time as _t
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU merge (avoids GPU OOM in re-quant)

    hf_cache_vol.reload()
    if not clobber and _merge_complete(merged_path):
        return {"merged_files": sorted(os.listdir(merged_path)), "merge_seconds": 0.0,
                "already_present": True}

    from tinker_cookbook.weights import build_hf_model
    t0 = _t.time()
    shutil.rmtree(merged_path, ignore_errors=True)
    build_hf_model(base_model=BASE_MODEL, adapter_path=adapter_vol_path, output_path=merged_path,
                   dtype="bfloat16")  # default strategy: merges LoRA, re-quantizes experts to MXFP4
    if not _merge_complete(merged_path):
        raise RuntimeError(f"merge incomplete: {sorted(os.listdir(merged_path))}")
    hf_cache_vol.commit()
    return {"merged_files": sorted(os.listdir(merged_path)), "merge_seconds": _t.time() - t0,
            "already_present": False}


@merge_app.function(image=merge_image, cpu=8.0, memory=262144,
                    volumes={"/cache": hf_cache_vol}, timeout=7200)
def merge_to_volume_bf16(adapter_vol_path: str, merged_path: str) -> dict:
    """CPU-only UNQUANTIZED merge (merge-fidelity).

    Loads the base gpt-oss-20b with ``Mxfp4Config(dequantize=True)`` -- i.e. the EXACT recipe the
    serving harness uses to get a clean bf16 residual stream -- merges the LoRA adapter into those
    bf16 weights, strips the mxfp4 quantization_config, and saves a plain bf16 HF checkpoint. The
    LoRA delta therefore NEVER passes through an MXFP4 quantization round-trip (unlike
    merge_to_volume, which re-quantizes the merged experts back to MXFP4). The harness loads this
    plain-bf16 checkpoint without dequantizing (it detects the absence of an mxfp4 config), so the
    served weights are faithful to the trained adapter.

    Idempotent: if merged_path already has safetensors, returns without re-merging."""
    import shutil
    import time as _t
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU merge

    hf_cache_vol.reload()
    import torch
    from transformers import AutoModelForCausalLM, Mxfp4Config
    from tinker_cookbook.weights._artifacts import load_adapter_weights
    from tinker_cookbook.weights._merge import merge_adapter_weights

    t0 = _t.time()
    # Always re-merge (no early-return): a stale/broken checkpoint must be overwritten cleanly.
    shutil.rmtree(merged_path, ignore_errors=True)
    print("[bf16-merge] loading base dequantized to bf16 (CPU) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
        device_map="cpu", attn_implementation="eager")
    model.eval()
    print("[bf16-merge] loading adapter + merging in bf16 ...", flush=True)
    from pathlib import Path as _P
    adapter_weights, adapter_config = load_adapter_weights(_P(adapter_vol_path))
    merge_adapter_weights(model, adapter_weights, adapter_config)

    # transformers 5.x routes save_pretrained through a regex weight-conversion that mangles the
    # dequantized expert names (e.g. `...experts.gate_up_proj$`), so the experts fail to reload.
    # Bypass save_pretrained: write the in-memory state_dict directly via safetensors (clean keys),
    # plus a config.json with the mxfp4 quantization_config dropped -> a standard bf16 gpt-oss the
    # harness loads as plain bf16 (the LoRA delta never sees an MXFP4 round-trip).
    import json as _json
    from safetensors.torch import save_file
    os.makedirs(merged_path, exist_ok=True)
    sd = model.state_dict()
    expert_keys = [k for k in sd if "experts.gate_up_proj" in k][:2]
    print(f"[bf16-merge] sample expert state_dict keys: {expert_keys}", flush=True)
    cleaned = {}
    for k, v in sd.items():
        k2 = k[:-1] if k.endswith("$") else k
        # clone() breaks any shared storage (tied embeddings) so safetensors can serialize each key.
        cleaned[k2] = v.detach().to(torch.bfloat16).clone().contiguous()
    have_experts = any(k.endswith("mlp.experts.gate_up_proj") for k in cleaned)
    bad = [k for k in cleaned if k.endswith("$")]
    if bad or not have_experts:
        raise RuntimeError(f"bf16 state_dict bad keys: bad={bad[:3]} have_experts={have_experts}")
    print(f"[bf16-merge] saving {len(cleaned)} tensors -> safetensors ...", flush=True)
    save_file(cleaned, os.path.join(merged_path, "model.safetensors"), metadata={"format": "pt"})
    cfgd = model.config.to_dict()
    cfgd.pop("quantization_config", None)
    _json.dump(cfgd, open(os.path.join(merged_path, "config.json"), "w"), indent=2, default=str)
    try:
        model.generation_config.save_pretrained(merged_path)
    except Exception:
        pass
    hf_cache_vol.commit()
    return {"merged_files": sorted(os.listdir(merged_path)), "merge_seconds": _t.time() - t0,
            "already_present": False, "merge_kind": "bf16_unquantized",
            "n_weight_keys": len(cleaned), "have_experts": have_experts,
            "sample_expert_keys": expert_keys}
