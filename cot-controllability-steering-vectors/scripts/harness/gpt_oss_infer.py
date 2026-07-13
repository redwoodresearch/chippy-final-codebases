"""Modal GPU inference for gpt-oss-20b with Harmony format, steering hooks, and capture.

This is the reusable inference harness for the whole project. Design:
  * Serving: HuggingFace ``transformers`` with the MXFP4 MoE weights *dequantized to bf16*
    (``Mxfp4Config(dequantize=True)``), so the residual stream is a clean bf16 tensor we can
    hook for steering. Runs on a Modal H100 (80GB) via a warm ``@app.cls`` container that
    loads the model once in ``@modal.enter()`` and serves many ``generate`` / ``capture`` /
    ``info`` calls.
  * Steering: forward hooks on ``model.model.layers[L]`` add ``alpha * v`` to the residual
    stream (resid_post) at *every* forward pass -> fires on prefill AND every decode step.
  * Capture: forward hooks record resid_post for activation-direction work.

The remote class returns *token ids only*; rendering (Harmony) and channel parsing happen
locally via ``harmony_utils`` so the slow GPU path stays minimal and the parsing primitive
can be iterated without touching the GPU. Local helpers (``generate``, ``capture_activations``,
``model_info``) wrap the remote calls with FileCache caching + Modal cost tracking.

Layer/residual conventions for gpt-oss-20b (from config.json):
  * 24 decoder layers (``model.model.layers[0..23]``), hidden_size = 2880 (residual dim).
  * MoE: 32 experts, 4 active/token; alternating sliding(window=128)/full attention + sinks.
  * Hooking ``model.model.layers[L]`` output gives resid_post of layer L (bf16).
"""

# NOTE: do NOT add `from __future__ import annotations` here -- PEP 563 stringizes the
# `model_path: str` class annotation, which breaks modal.parameter()'s type resolver.
import argparse
import os
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal app / image / volume
# ---------------------------------------------------------------------------
MODEL_NAME = "openai/gpt-oss-20b"
GPU_TYPE = "H100"
N_LAYERS = 24
HIDDEN_SIZE = 2880
# Assistant-turn stop tokens (Harmony): <|call|>=200012, <|return|>=200002.
ASSISTANT_STOP_TOKENS = [200012, 200002]
PAD_TOKEN_ID = 199999

app = modal.App("gpt-oss-infer")

# HF weights cached on a Volume so containers don't re-download ~13GB each cold start.
hf_cache_vol = modal.Volume.from_name("gpt-oss-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.57.1",
        "accelerate==1.10.1",
        "triton==3.4.0",
        "kernels==0.10.3",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "numpy",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/cache": hf_cache_vol},
    timeout=3600,
    scaledown_window=300,  # keep container warm 5 min after last call (warm-container pattern)
)
class GptOss:
    # Optional fine-tuned/merged checkpoint to load instead of the base HF model. Default = base.
    # A volume path (e.g. /cache/merged_ft_c32) loads a CPU-merged MXFP4 FT model; the load recipe
    # (Mxfp4Config(dequantize=True), eager) is IDENTICAL to base, so steering hooks apply unchanged.
    model_path: str = modal.parameter(default=MODEL_NAME)

    @modal.enter()
    def load(self):
        # Reduce CUDA allocator fragmentation on long warm-container runs (avoids OOM after many
        # variable-length generation batches). Set BEFORE torch initializes CUDA. This only
        # changes the allocator strategy, not numerics/outputs, and does NOT change the Modal
        # image_id (the source file is mounted, not baked into the image), so cached generations
        # remain valid.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

        os.environ.setdefault("HF_HOME", "/cache/hf")
        t0 = time.monotonic()
        self.torch = torch
        # Always load the BASE tokenizer (the gpt-oss tokenizer is identical for the merged FT model;
        # generation works on token ids directly so the tokenizer is not on the hot path).
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        src = self.model_path
        if src != MODEL_NAME and src.startswith("/cache"):
            hf_cache_vol.reload()  # see freshly-merged FT weights committed by the merge container
        # Detect whether the checkpoint is MXFP4-quantized. The base + the (re-quantized) MXFP4 merge
        # have a mxfp4 quantization_config and must be dequantized to bf16 for a clean residual
        # stream; an UNQUANTIZED bf16 merge (the faithful serving path) has NO mxfp4 config
        # and is loaded as plain bf16 (the LoRA delta never saw an MXFP4 round-trip). Either way the
        # residual stream is bf16 and steering hooks apply unchanged.
        cfg = AutoConfig.from_pretrained(src)
        qc = getattr(cfg, "quantization_config", None)
        method = None
        if isinstance(qc, dict):
            method = qc.get("quant_method")
        elif qc is not None:
            method = getattr(qc, "quant_method", None)
        is_mxfp4 = (method == "mxfp4")
        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="cuda",
                           attn_implementation="eager")
        if is_mxfp4:
            load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        self.serving_kind = "mxfp4_dequant" if is_mxfp4 else "bf16_plain"
        self.model = AutoModelForCausalLM.from_pretrained(src, **load_kwargs)
        self.model.eval()
        self.layers = self.model.model.layers
        self.load_seconds = time.monotonic() - t0
        # cumulative GPU compute seconds spent in method bodies (for cost cross-checks)
        self.compute_seconds = 0.0
        print(f"[load] model loaded from {src} ({self.serving_kind}) in {self.load_seconds:.1f}s; "
              f"n_layers={len(self.layers)}")

    # -- hook helpers -------------------------------------------------------
    def _make_steering_hook(self, vec, counter, mask):
        """Forward hook that adds ``vec`` (scaled bf16 tensor, shape [hidden]) to the layer's
        residual-stream output at every forward pass, but ONLY at real (non-pad) positions.

        ``mask`` is the [B, T_prefill] attention mask. During prefill the hidden states have the
        padded length, so we zero the addition at pad positions (left-padding); during decode
        (KV cache, seq_len=1) the single new token is always real. Masking makes steering
        padding-invariant (adding pad tokens is then a true no-op). Increments counter[0]/call."""

        def hook(module, args, output):
            counter[0] += 1
            hs = output[0] if isinstance(output, tuple) else output
            add = vec.to(hs.dtype)
            if hs.shape[1] == mask.shape[1] and hs.shape[1] > 1:  # prefill: skip pad positions
                m = mask.to(hs.dtype).unsqueeze(-1)  # [B, T, 1]
                hs = hs + add * m
            else:  # decode step (or unpadded single token): all real
                hs = hs + add
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        return hook

    def _make_capture_hook(self, store, key):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            store[key] = hs.detach()
        return hook

    # -- diagnostics --------------------------------------------------------
    @modal.method()
    def info(self, sample_prompt_token_ids: list[int]) -> dict:
        """Return architecture + activation-sanity diagnostics for hooking de-risking."""
        torch = self.torch
        import torch as _t

        store: dict = {}
        handles = []
        capture_layers = list(range(0, N_LAYERS, 4)) + [N_LAYERS - 1]
        for L in capture_layers:
            handles.append(self.layers[L].register_forward_hook(self._make_capture_hook(store, L)))
        try:
            ids = torch.tensor([sample_prompt_token_ids], device="cuda")
            with torch.no_grad():
                out = self.model(ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        stats = {}
        for L, hs in store.items():
            t = hs.float()
            stats[str(L)] = {
                "shape": list(hs.shape),
                "dtype": str(hs.dtype),
                "mean_abs": t.abs().mean().item(),
                "l2_norm_last_tok": t[0, -1].norm().item(),
                "max_abs": t.abs().max().item(),
                "has_nan": bool(torch.isnan(t).any().item()),
                "has_inf": bool(torch.isinf(t).any().item()),
            }
        return {
            "n_layers": len(self.layers),
            "hidden_size": self.model.config.hidden_size,
            "model_dtype": str(next(self.model.parameters()).dtype),
            "resid_dtype": str(out.logits.dtype),
            "logits_shape": list(out.logits.shape),
            "layer_module_type": type(self.layers[0]).__name__,
            "load_seconds": self.load_seconds,
            "activation_stats": stats,
        }

    # -- architecture probe (confirm submodule/return structure) -----
    @modal.method()
    def arch_probe(self, sample_prompt_token_ids: list[int]) -> dict:
        """Inspect the gpt-oss decoder-layer internals we will hook for mechanistic attribution: per-layer attention type (sliding vs full), the self_attn / mlp / router
        forward-output STRUCTURE (so we can hook output[0]=sub-block contribution, output[1]=attn
        weights / router scores), the attention-weight shape (heads, with/without the sink column),
        and config dims (heads, kv-heads, experts, top-k, sliding_window)."""
        torch = self.torch
        cfg = self.model.config
        layer_types = list(getattr(cfg, "layer_types", []))
        probe = {}
        L0 = 0
        attn_mod = self.layers[L0].self_attn
        mlp_mod = self.layers[L0].mlp
        router_mod = mlp_mod.router

        captured = {}

        def cap(name):
            def hook(module, args, output):
                if isinstance(output, tuple):
                    captured[name] = ("tuple", len(output),
                                      [list(o.shape) if hasattr(o, "shape") else type(o).__name__
                                       for o in output])
                else:
                    captured[name] = ("tensor", 1, [list(output.shape)])
            return hook

        h = [attn_mod.register_forward_hook(cap("self_attn")),
             mlp_mod.register_forward_hook(cap("mlp")),
             router_mod.register_forward_hook(cap("router"))]
        try:
            ids = torch.tensor([sample_prompt_token_ids], device="cuda")
            with torch.no_grad():
                self.model(ids, use_cache=False)
        finally:
            for hh in h:
                hh.remove()

        return {
            "n_layers": len(self.layers),
            "layer_types": layer_types,
            "num_attention_heads": int(getattr(cfg, "num_attention_heads", -1)),
            "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", -1)),
            "head_dim": int(getattr(cfg, "head_dim", -1)),
            "num_local_experts": int(getattr(cfg, "num_local_experts", -1)),
            "num_experts_per_tok": int(getattr(cfg, "num_experts_per_tok", -1)),
            "sliding_window": int(getattr(cfg, "sliding_window", -1)),
            "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-6)),
            "captured_output_structure": captured,
            "self_attn_type": type(attn_mod).__name__,
            "mlp_type": type(mlp_mod).__name__,
            "router_type": type(router_mod).__name__,
            "has_sinks": hasattr(attn_mod, "sinks"),
            "attn_implementation": str(getattr(cfg, "_attn_implementation", "?")),
        }

    # -- generation ---------------------------------------------------------
    @modal.method()
    def generate(self, request: dict) -> dict:
        torch = self.torch
        t_start = time.monotonic()
        prompts = request["prompt_token_ids"]  # list[list[int]]
        max_new_tokens = int(request.get("max_new_tokens", 512))
        temperature = float(request.get("temperature", 0.0))
        top_p = float(request.get("top_p", 1.0))
        seed = int(request.get("seed", 0))
        do_sample = temperature > 0.0
        steering = request.get("steering")  # None or list of {"layer", "vector"}
        # Optional EXTRA left-padding (pad tokens, attn mask 0) prepended to every prompt. Used to
        # isolate left-padding correctness from batch-size effects (run batch=1 with/without it).
        extra_left_pad = int(request.get("extra_left_pad", 0))

        # Left-pad batch.
        maxlen = max(len(p) for p in prompts) + extra_left_pad
        input_ids = torch.full((len(prompts), maxlen), PAD_TOKEN_ID, dtype=torch.long)
        attn = torch.zeros((len(prompts), maxlen), dtype=torch.long)
        for i, p in enumerate(prompts):
            input_ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, maxlen - len(p):] = 1
        input_ids = input_ids.to("cuda")
        attn = attn.to("cuda")

        # Register steering hooks.
        handles = []
        counter = [0]
        if steering:
            for inj in steering:
                L = int(inj["layer"])
                vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                handles.append(self.layers[L].register_forward_hook(self._make_steering_hook(vec, counter, attn)))

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=ASSISTANT_STOP_TOKENS,
            pad_token_id=PAD_TOKEN_ID,
            use_cache=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
            torch.manual_seed(seed)
        try:
            with torch.no_grad():
                out = self.model.generate(input_ids, attention_mask=attn, **gen_kwargs)
        finally:
            for h in handles:
                h.remove()

        gen = out[:, maxlen:].tolist()  # completion tokens only
        results = []
        for seq in gen:
            # Cut at first stop token (inclusive); drop trailing pads.
            cut = None
            for j, tk in enumerate(seq):
                if tk in ASSISTANT_STOP_TOKENS:
                    cut = j + 1
                    break
            comp = seq[:cut] if cut is not None else [t for t in seq if t != PAD_TOKEN_ID]
            results.append({"token_ids": comp, "n_tokens": len(comp)})

        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        total_gen_tokens = sum(r["n_tokens"] for r in results)
        return {
            "generations": results,
            "hook_fire_count": counter[0],
            "n_prompts": len(prompts),
            "total_gen_tokens": total_gen_tokens,
            "elapsed": elapsed,
            "tokens_per_sec": total_gen_tokens / elapsed if elapsed > 0 else 0.0,
            "steering_active": bool(steering),
        }

    # -- teacher-forced analysis-span capture (steering directions) ----
    @modal.method()
    def capture_spans(self, request: dict) -> dict:
        """Teacher-force a batch of full token sequences (prompt + assistant completion) and
        capture resid_post at ``capture_layers``, mean-pooled over a per-sequence TOKEN SPAN
        (the analysis-channel content tokens of the completion).

        request = {
          "sequences": list[list[int]],     # full prompt+completion token ids (no padding)
          "spans": list[[start, end]],       # analysis-token span (indices into the unpadded seq)
          "capture_layers": list[int],
        }

        Uses RIGHT padding so real tokens sit at indices 0..len-1 (default position_ids are then
        correct for the real tokens) and the trailing pad positions are masked out of attention.
        Because attention is causal, a real token at position i (< len) only attends to positions
        0..i (all real), so its resid_post equals the unpadded value -> the captured per-token
        residuals are padding/batch-invariant (verified by a batch-1-vs-batched check). Returns per
        (layer, sequence): the mean residual vector over the span + the mean per-token L2 norm over
        the span (the latter is the per-layer typical ||resid|| used to scale steering magnitude).
        """
        torch = self.torch
        t_start = time.monotonic()
        sequences = request["sequences"]
        spans = request["spans"]
        capture_layers = request["capture_layers"]

        steering = request.get("steering")  # optional: [{"layer","vector"}] added at real positions

        maxlen = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences), maxlen), PAD_TOKEN_ID, dtype=torch.long)
        attn = torch.zeros((len(sequences), maxlen), dtype=torch.long)
        for i, s in enumerate(sequences):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, :len(s)] = 1
        input_ids = input_ids.to("cuda")
        attn = attn.to("cuda")

        store: dict = {}
        handles = []
        for L in capture_layers:
            handles.append(self.layers[L].register_forward_hook(self._make_capture_hook(store, L)))
        if steering:  # inject (e.g. gL10) during the teacher-forced pass, masking pad positions
            cnt = [0]
            for inj in steering:
                vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                handles.append(self.layers[int(inj["layer"])].register_forward_hook(
                    self._make_steering_hook(vec, cnt, attn)))
        try:
            with torch.no_grad():
                self.model(input_ids, attention_mask=attn, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        out = {str(L): [] for L in capture_layers}
        for L in capture_layers:
            hs = store[L].float()  # [B, T, H]
            for i, sp in enumerate(spans):
                s_start, s_end = int(sp[0]), int(sp[1])
                seg = hs[i, s_start:s_end]  # [span_len, H]
                mean_vec = seg.mean(dim=0)
                tok_norms = seg.norm(dim=-1)  # per-token L2 norm
                out[str(L)].append({
                    "mean": mean_vec.cpu().tolist(),
                    "mean_token_norm": float(tok_norms.mean().item()),
                    "span_len": int(seg.shape[0]),
                    "has_nan": bool(torch.isnan(seg).any().item()),
                })
        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"layers": out, "elapsed": elapsed, "n": len(sequences)}

    # -- gradient-trained additive steering bias ----
    @modal.method()
    def train_steer(self, request: dict) -> dict:
        """Train a small ADDITIVE steering bias (a learned vector added to resid_post at one or a
        small set of layers) by gradient descent, with the MODEL WEIGHTS FROZEN, minimizing the NLL
        of teacher-forced complying-target completions (completion-only loss, prompt masked — the
        SAME objective family as the LoRA fine-tune, but the only trainable parameter is the fixed
        additive bias). The vector is added to resid_post at EVERY real position (matching how the
        eval steering hook adds it at every prefill/decode step), so the trained vector transfers
        directly to generation via the existing steering path.

        request = {
          "sequences":  list[list[int]],   # full prompt+completion token ids (RIGHT-padded here)
          "comp_starts": list[int],        # index where the completion (loss) span begins per seq
          "layers": list[int],             # which resid_post layer(s) get a learned vector
          "n_steps": int, "lr": float, "batch_size": int, "seed": int,
          "warmup_frac": float, "weight_decay": float, "grad_clip": float,
          "max_length": int,               # drop sequences longer than this (memory)
          "grad_ckpt": bool,               # gradient checkpointing (memory; needs train mode)
          "kl_coef": float,                # optional KL(steered||base) reg on completion tokens
          "init_vectors": list[list[float]] | None,  # optional warm-start (e.g. diff-of-means)
        }
        Returns {"vectors": [[...H] per layer], "layers": [...], "losses": [...],
                 "kl_losses": [...], "n_examples": int, "grad_norms": [...], "elapsed": float}.
        """
        torch = self.torch
        import math as _math
        import random as _random
        import torch.nn.functional as Fnn
        t_start = time.monotonic()

        sequences = request["sequences"]
        comp_starts = request["comp_starts"]
        layers = [int(L) for L in request["layers"]]
        n_steps = int(request["n_steps"])
        lr = float(request["lr"])
        batch_size = int(request["batch_size"])
        seed = int(request.get("seed", 0))
        warmup_frac = float(request.get("warmup_frac", 0.05))
        weight_decay = float(request.get("weight_decay", 0.0))
        grad_clip = float(request.get("grad_clip", 1.0))
        max_length = int(request.get("max_length", 1024))
        grad_ckpt = bool(request.get("grad_ckpt", True))
        kl_coef = float(request.get("kl_coef", 0.0))
        log_every = int(request.get("log_every", 10))
        init_vectors = request.get("init_vectors")

        data = [(s, int(cs)) for s, cs in zip(sequences, comp_starts)
                if len(s) <= max_length and 0 < int(cs) < len(s)]
        n = len(data)
        print(f"[train_steer] {n}/{len(sequences)} seqs (<= {max_length} tok); layers={layers} "
              f"steps={n_steps} lr={lr} bs={batch_size} grad_ckpt={grad_ckpt} kl={kl_coef}")

        torch.manual_seed(seed)
        Hd = self.model.config.hidden_size
        if init_vectors is not None:
            vecs = [torch.tensor(v, device="cuda", dtype=torch.float32).requires_grad_(True)
                    for v in init_vectors]
        else:
            vecs = [torch.zeros(Hd, device="cuda", dtype=torch.float32, requires_grad=True)
                    for _ in layers]
        opt = torch.optim.Adam(vecs, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

        for p in self.model.parameters():
            p.requires_grad_(False)
        # Disable any dropout so train() mode (needed by HF gradient checkpointing) is numerically
        # identical to eval() (gpt-oss has p=0 dropout by default; this is defensive).
        for mod in self.model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        if grad_ckpt:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.train()

        cur_mask = [None]

        def make_hook(vec):
            def hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs + vec.to(hs.dtype) * cur_mask[0]
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return hook

        handles = [self.layers[L].register_forward_hook(make_hook(v)) for L, v in zip(layers, vecs)]

        rng = _random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        ptr = [0]

        def next_batch():
            idxs = []
            while len(idxs) < batch_size:
                if ptr[0] >= n:
                    rng.shuffle(order)
                    ptr[0] = 0
                idxs.append(order[ptr[0]])
                ptr[0] += 1
            return [data[i] for i in idxs]

        losses, kl_losses, grad_norms = [], [], []
        warmup = max(1, int(warmup_frac * n_steps))
        PAD = PAD_TOKEN_ID
        try:
            for step in range(n_steps):
                batch = next_batch()
                maxlen = max(len(s) for s, _ in batch)
                input_ids = torch.full((len(batch), maxlen), PAD, dtype=torch.long, device="cuda")
                attn = torch.zeros((len(batch), maxlen), dtype=torch.long, device="cuda")
                loss_mask = torch.zeros((len(batch), maxlen), dtype=torch.float32, device="cuda")
                for i, (s, cs) in enumerate(batch):
                    input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device="cuda")
                    attn[i, :len(s)] = 1
                    loss_mask[i, cs:len(s)] = 1.0
                cur_mask[0] = attn.to(torch.bfloat16).unsqueeze(-1)

                if step < warmup:
                    cur_lr = lr * (step + 1) / warmup
                else:
                    prog = (step - warmup) / max(1, n_steps - warmup)
                    cur_lr = lr * max(0.0, 1.0 - prog)
                for g in opt.param_groups:
                    g["lr"] = cur_lr

                # optional base (unsteered) logits for the KL regularizer
                base_logp = None
                if kl_coef > 0:
                    for h in handles:
                        h.remove()
                    with torch.no_grad():
                        base_out = self.model(input_ids, attention_mask=attn, use_cache=False)
                        base_logp = Fnn.log_softmax(base_out.logits[:, :-1, :].float(), dim=-1).detach()
                    handles[:] = [self.layers[L].register_forward_hook(make_hook(v))
                                  for L, v in zip(layers, vecs)]

                opt.zero_grad(set_to_none=True)
                out = self.model(input_ids, attention_mask=attn, use_cache=False)
                shift_logits = out.logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                shift_mask = loss_mask[:, 1:]
                logp = Fnn.log_softmax(shift_logits.float(), dim=-1)
                nll = -logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                denom = shift_mask.sum().clamp_min(1.0)
                loss = (nll * shift_mask).sum() / denom
                kl_val = 0.0
                if kl_coef > 0 and base_logp is not None:
                    # KL(base || steered) on completion tokens encourages staying close to base
                    kl_tok = (base_logp.exp() * (base_logp - logp)).sum(-1)
                    kl_val = (kl_tok * shift_mask).sum() / denom
                    total = loss + kl_coef * kl_val
                else:
                    total = loss
                total.backward()
                gn = torch.nn.utils.clip_grad_norm_(vecs, grad_clip).item()
                opt.step()
                losses.append(float(loss.item()))
                kl_losses.append(float(kl_val) if kl_coef > 0 else 0.0)
                grad_norms.append(float(gn))
                if step % log_every == 0 or step == n_steps - 1:
                    vn = [round(float(v.norm().item()), 1) for v in vecs]
                    print(f"  step {step:4d}/{n_steps} lr={cur_lr:.2e} loss={loss.item():.4f} "
                          f"kl={kl_val if kl_coef>0 else 0:.4f} gnorm={gn:.3f} ||v||={vn}")
        finally:
            for h in handles:
                h.remove()
            if grad_ckpt:
                self.model.gradient_checkpointing_disable()
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"vectors": [v.detach().float().cpu().tolist() for v in vecs], "layers": layers,
                "losses": losses, "kl_losses": kl_losses, "grad_norms": grad_norms,
                "n_examples": n, "elapsed": elapsed,
                "vector_norms": [float(v.norm().item()) for v in vecs]}

    # -- greedy generation with sub-block ablation (behavioral spot-check) ----
    @modal.method()
    def gen_ablate(self, request: dict) -> dict:
        """Greedy generation (no KV cache; recompute each step) with optional sub-block ABLATION:
        at each step, run a BASE (unsteered) forward to capture the target sub-blocks' outputs, then
        a STEERED forward in which those sub-blocks are FROZEN to the base outputs (severing their
        response to the steering), and take the argmax next token. This makes the logit-level
        ablation result BEHAVIORAL — does freezing the mediating components actually stop gL10 from
        generating the asked form?

        request = {sequences(prompt+header ids), steering, ablate_targets([[L,'attn'|'mlp'],...] or
                   []), n_new, with_steer(bool), mask_spec(optional)}

        mask_spec (behavioral sub-span knockout) = {layers:[...], heads:"all"|[...],
        span_ranges:{str(seq_idx): [a,b] or {"ids":[...]}}}: in the steered forward, zero the target
        full-attention heads' post-softmax attention onto the given key span(s) and renormalize
        (sink unchanged), then GENERATE — does masking a NECESSARY sub-span revert the generated CoT
        toward base prose? Combinable with ablate_targets (usually used alone).
        """
        torch = self.torch
        t_start = time.monotonic()
        seqs = [list(s) for s in request["sequences"]]
        steering = request.get("steering")
        targets = [(int(L), k) for L, k in request.get("ablate_targets", [])]
        n_new = int(request.get("n_new", 64))
        with_steer = bool(request.get("with_steer", True))
        mask_spec = request.get("mask_spec")
        mask_layers = [int(L) for L in mask_spec["layers"]] if mask_spec else []
        mask_heads = (mask_spec.get("heads", "all") if mask_spec else "all")
        mask_sr = (mask_spec.get("span_ranges", {}) if mask_spec else {})
        done = [False] * len(seqs)

        for _step in range(n_new):
            lens = [len(s) for s in seqs]
            maxlen = max(lens)
            input_ids = torch.full((len(seqs), maxlen), PAD_TOKEN_ID, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for r, s in enumerate(seqs):
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[r, :len(s)] = 1
            input_ids = input_ids.to("cuda")
            attn = attn.to("cuda")

            store = {}
            if targets:  # base forward to capture sub-block outputs to freeze to
                handles = []
                for L, kind in targets:
                    mod = self.layers[L].self_attn if kind == "attn" else self.layers[L].mlp

                    def cap_hook(module, args, output, _k=(L, kind)):
                        store[_k] = (output[0] if isinstance(output, tuple) else output).detach()
                    handles.append(mod.register_forward_hook(cap_hook))
                try:
                    with torch.no_grad():
                        self.model(input_ids, attention_mask=attn, use_cache=False)
                finally:
                    for h in handles:
                        h.remove()

            handles = []
            cnt = [0]
            if with_steer and steering:
                for inj in steering:
                    vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                    handles.append(self.layers[int(inj["layer"])].register_forward_hook(
                        self._make_steering_hook(vec, cnt, attn)))
            for L, kind in targets:
                mod = self.layers[L].self_attn if kind == "attn" else self.layers[L].mlp

                def froze(module, args, output, _k=(L, kind)):
                    b = store[_k]
                    return (b,) + tuple(output[1:]) if isinstance(output, tuple) else b
                handles.append(mod.register_forward_hook(froze))
            # Surgical attention-sub-span masking during generation
            if mask_layers:
                def make_mask(_L):
                    am = self.layers[_L].self_attn
                    cur = {}

                    def vhook(module, args, output):
                        b, t, _ = output.shape
                        hd = am.head_dim
                        nkv = am.config.num_key_value_heads
                        vv = output.view(b, t, nkv, hd).transpose(1, 2)
                        nrep = am.num_key_value_groups
                        cur["V"] = (vv[:, :, None, :, :].expand(b, nkv, nrep, t, hd)
                                    .reshape(b, nkv * nrep, t, hd) if nrep > 1 else vv)

                    def ohook(module, args, output):
                        A = output[1].clone()
                        V = cur["V"]
                        B, Hh, Tq, Tkv = A.shape
                        hm = (range(Hh) if mask_heads == "all" else [int(x) for x in mask_heads])
                        for r in range(B):
                            sp = mask_sr.get(str(r))
                            if sp is None:
                                continue
                            if isinstance(sp, dict):
                                col = torch.tensor([int(x) for x in sp["ids"]],
                                                   dtype=torch.long, device=A.device)
                                if col.numel() == 0:
                                    continue
                            else:
                                col = torch.arange(int(sp[0]), int(sp[1]), device=A.device)
                            for h in hm:
                                row = A[r, h]
                                old = row.sum(-1, keepdim=True)
                                row[:, col] = 0.0
                                new = row.sum(-1, keepdim=True).clamp_min(1e-9)
                                A[r, h] = row * (old / new)
                        ctx = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, Tq, -1)
                        return (am.o_proj(ctx),) + tuple(output[1:])
                    return am, vhook, ohook
                for L in mask_layers:
                    am, vhook, ohook = make_mask(L)
                    handles.append(am.v_proj.register_forward_hook(vhook))
                    handles.append(am.register_forward_hook(ohook))
            try:
                with torch.no_grad():
                    out = self.model(input_ids, attention_mask=attn, use_cache=False)
            finally:
                for h in handles:
                    h.remove()
            for r in range(len(seqs)):
                if done[r]:
                    continue
                nxt = int(out.logits[r, lens[r] - 1].argmax().item())
                seqs[r].append(nxt)
                if nxt in ASSISTANT_STOP_TOKENS:
                    done[r] = True
            del out, store
            torch.cuda.empty_cache()
            if all(done):
                break

        n_prompt = [len(s) for s in request["sequences"]]
        gens = [seqs[r][n_prompt[r]:] for r in range(len(seqs))]
        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"generations": gens, "elapsed": elapsed}

    # -- activation capture -------------------------------------------------
    @modal.method()
    def capture(self, request: dict) -> dict:
        """Forward pass (no generation) over a batch of prompts; capture resid_post at
        ``capture_layers``. Returns per-prompt last-token + mean activations (over non-pad
        positions) for each layer, plus norm/NaN stats."""
        torch = self.torch
        t_start = time.monotonic()
        prompts = request["prompt_token_ids"]
        capture_layers = request["capture_layers"]

        maxlen = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), maxlen), PAD_TOKEN_ID, dtype=torch.long)
        attn = torch.zeros((len(prompts), maxlen), dtype=torch.long)
        for i, p in enumerate(prompts):
            input_ids[i, maxlen - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[i, maxlen - len(p):] = 1
        input_ids = input_ids.to("cuda")
        attn = attn.to("cuda")

        store: dict = {}
        handles = []
        for L in capture_layers:
            handles.append(self.layers[L].register_forward_hook(self._make_capture_hook(store, L)))
        try:
            with torch.no_grad():
                self.model(input_ids, attention_mask=attn, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        out_layers = {}
        for L in capture_layers:
            hs = store[L].float()  # [B, T, H]
            per_prompt = []
            for i in range(len(prompts)):
                mask = attn[i].bool()
                toks = hs[i][mask]  # [valid_T, H]
                mean_vec = toks.mean(dim=0)
                last_vec = hs[i, -1]
                per_prompt.append({
                    "mean": mean_vec.cpu().tolist(),
                    "last": last_vec.cpu().tolist(),
                    "mean_norm": mean_vec.norm().item(),
                    "last_norm": last_vec.norm().item(),
                    "has_nan": bool(torch.isnan(toks).any().item()),
                })
            out_layers[str(L)] = per_prompt

        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"layers": out_layers, "elapsed": elapsed, "n_prompts": len(prompts)}

    # -- logit lens / vocab projection of residual-stream directions ----
    @modal.method()
    def lens_project(self, request: dict) -> dict:
        """Project residual-stream DIRECTIONS through the model's output (unembedding) and input
        (token-embedding) heads, for the logit lens (what vocab tokens a direction promotes/
        suppresses) + token-embedding alignment.

        For each input vector v (shape [hidden]):
          * logit_lens = lm_head.weight @ (final_norm.weight ⊙ v)   -- the DIRECT logit lens, folding
            the final RMSNorm's diagonal weight into the unembedding. (The RMSNorm's per-token rms is
            a POSITIVE scalar that rescales but does not reorder the vocab, so it is omitted; v is a
            mid-layer direction so this is an APPROXIMATE read of its output effect — validated
            separately against the actual induced logit shift.)
          * raw_unembed = lm_head.weight @ v                        -- logit lens WITHOUT the final
            norm (sanity comparison; shows how much the norm-weight diagonal matters).
          * embed_cos   = cosine(v, embed_tokens.weight[row])       -- input-embedding alignment.

        request = {"vecs": [[...hidden], ...], "topk": int, "full_idx": [int,...]}
        Returns per vec: top/bottom-`topk` (token_id, value) for logit_lens / raw_unembed / embed_cos,
        and (for indices in `full_idx`) the FULL logit_lens + embed_cos vectors (for correlation /
        local re-ranking). Plus global meta (vocab size, tied-unembedding flag, norm-weight stats).
        """
        torch = self.torch
        t_start = time.monotonic()
        vecs = request["vecs"]
        topk = int(request.get("topk", 100))
        full_idx = set(int(i) for i in request.get("full_idx", []))

        W = self.model.lm_head.weight  # [V, H]
        E = self.model.model.embed_tokens.weight  # [V, H]
        nrm = self.model.model.norm.weight  # [H]
        tied = bool(W.data_ptr() == E.data_ptr())
        Wf = W.float()
        Ef = E.float()
        nrmf = nrm.float()
        E_norms = Ef.norm(dim=-1)  # [V]
        V = Wf.shape[0]

        def tb(t, k):  # top/bottom-k (id, value)
            kk = min(k, t.shape[0])
            tv, ti = torch.topk(t, kk)
            bv, bi = torch.topk(-t, kk)
            return ([[int(i), float(v)] for i, v in zip(ti.tolist(), tv.tolist())],
                    [[int(i), float(-v)] for i, v in zip(bi.tolist(), bv.tolist())])

        out = []
        for j, v in enumerate(vecs):
            vt = torch.tensor(v, dtype=torch.float32, device="cuda")
            vn = float(vt.norm().item())
            ll = Wf @ (nrmf * vt)           # [V]
            raw = Wf @ vt                   # [V]
            ecos = (Ef @ vt) / (E_norms * (vn + 1e-8))  # [V]
            ll_top, ll_bot = tb(ll, topk)
            raw_top, raw_bot = tb(raw, topk)
            ec_top, ec_bot = tb(ecos, topk)
            entry = {"vec_norm": vn, "ll_top": ll_top, "ll_bot": ll_bot,
                     "raw_top": raw_top, "raw_bot": raw_bot,
                     "ecos_top": ec_top, "ecos_bot": ec_bot}
            if j in full_idx:
                entry["ll_full"] = ll.cpu().tolist()
                entry["ecos_full"] = ecos.cpu().tolist()
            out.append(entry)

        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"vecs": out, "vocab_size": int(V), "tied_unembedding": tied,
                "norm_weight_mean": float(nrmf.mean().item()),
                "norm_weight_l2": float(nrmf.norm().item()),
                "hidden": int(Wf.shape[1]), "elapsed": elapsed}

    # -- actual induced next-token logit shift under steering (the reliable read) --
    @modal.method()
    def induced_shift(self, request: dict) -> dict:
        """Teacher-force a batch of sequences (prompt + a short fixed continuation) and read the
        next-token logit distribution WITH vs WITHOUT a steering injection, at requested positions.
        This measures what `gL10` ACTUALLY does to the next-token logits (downstream of all 13 blocks
        + final norm) — the reliable validation of the (approximate, mid-layer) direct logit lens.

        request = {
          "sequences": list[list[int]],          # right-padded internally
          "positions": list[list[int]],           # per-seq positions to read logits AT (logits[pos]
                                                   #   predicts the token at pos+1)
          "steering":  [{"layer", "vector"}],      # the injection (gL10)
          "candidate_ids": list[int],              # token ids to report base+steered logit for
          "topk": int,                             # global top/bottom-k of the induced delta
        }
        Returns per (seq, pos): global top/bottom-`topk` (token_id, delta), the base + steered top-30
        predicted tokens, and base/steered logits at `candidate_ids` (for correlation vs the lens).
        Base and steered are computed in the SAME call (same padding/batching) so the delta is exact.
        """
        torch = self.torch
        t_start = time.monotonic()
        sequences = request["sequences"]
        positions = request["positions"]
        steering = request["steering"]
        cand = request.get("candidate_ids", [])
        topk = int(request.get("topk", 200))
        mb = int(request.get("micro_batch", 4))  # mini-batch to bound the [B,T,V] logits memory
        cand_t = torch.tensor(cand, dtype=torch.long, device="cuda") if cand else None

        def tb(t, k):
            kk = min(k, t.shape[0])
            tv, ti = torch.topk(t, kk)
            bv, bi = torch.topk(-t, kk)
            return ([[int(i), float(v)] for i, v in zip(ti.tolist(), tv.tolist())],
                    [[int(i), float(-v)] for i, v in zip(bi.tolist(), bv.tolist())])

        # Process in length-sorted mini-batches to keep the per-call [bs, T, vocab] logits small.
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        per_seq = {}  # i -> {pos -> {base_logit_row, steer_logit_row}}
        for b0 in range(0, len(order), mb):
            idxs = order[b0:b0 + mb]
            maxlen = max(len(sequences[i]) for i in idxs)
            input_ids = torch.full((len(idxs), maxlen), PAD_TOKEN_ID, dtype=torch.long)
            attn = torch.zeros((len(idxs), maxlen), dtype=torch.long)
            for r, i in enumerate(idxs):
                s = sequences[i]
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)  # RIGHT pad (positions ok)
                attn[r, :len(s)] = 1
            input_ids = input_ids.to("cuda")
            attn = attn.to("cuda")

            def forward(with_steer):
                handles = []
                if with_steer:
                    for inj in steering:
                        L = int(inj["layer"])
                        vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                        cnt = [0]
                        handles.append(self.layers[L].register_forward_hook(
                            self._make_steering_hook(vec, cnt, attn)))
                try:
                    with torch.no_grad():
                        out = self.model(input_ids, attention_mask=attn, use_cache=False)
                    return out.logits.float()  # [bs, T, V]
                finally:
                    for h in handles:
                        h.remove()

            bl_all = forward(False)
            for r, i in enumerate(idxs):
                per_seq.setdefault(i, {})
                for p in positions[i]:
                    per_seq[i].setdefault(p, {})["base"] = bl_all[r, p].clone()
            del bl_all
            sl_all = forward(True)
            for r, i in enumerate(idxs):
                for p in positions[i]:
                    per_seq[i][p]["steer"] = sl_all[r, p].clone()
            del sl_all
            torch.cuda.empty_cache()

        results = []
        for i, poss in enumerate(positions):
            for p in poss:
                bl = per_seq[i][p]["base"]
                sl = per_seq[i][p]["steer"]
                delta = sl - bl
                d_top, d_bot = tb(delta, topk)
                b_top, _ = tb(bl, 30)
                s_top, _ = tb(sl, 30)
                entry = {"seq_idx": i, "pos": int(p), "d_top": d_top, "d_bot": d_bot,
                         "base_top": b_top, "steer_top": s_top}
                if cand_t is not None:
                    entry["cand_base"] = bl[cand_t].cpu().tolist()
                    entry["cand_steer"] = sl[cand_t].cpu().tolist()
                results.append(entry)

        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"results": results, "candidate_ids": cand, "elapsed": elapsed}

    # -- mechanistic internals: resid-lens trajectory + sub-block DLA + MoE router --
    @modal.method()
    def capture_internals(self, request: dict) -> dict:
        """Teacher-force a batch of sequences and capture, at requested positions, the per-layer
        mechanistic internals WITH vs WITHOUT a steering injection, in ONE micro-batched call.

        Per (seq, pos, layer ℓ, arm∈{base,steer}) it returns:
          * resid_lens[ℓ]      = candidate-token logits from the INTERMEDIATE logit lens of resid_post
                                 at layer ℓ: (RMSNorm(resid_post_ℓ) ⊙ norm_w) @ Wᵤ[cand]. (Where does
                                 the form-logit / meta-suppression effect become token-level?)
          * attn_dla[ℓ]        = Direct Logit Attribution of layer ℓ's ATTENTION sub-block output to
                                 each candidate's FINAL logit (frozen final-LN): (attnΔ_ℓ/rms_final ⊙
                                 norm_w) @ Wᵤ[cand].
          * mlp_dla[ℓ]         = same for the MLP/MoE sub-block output. (attention-vs-MLP/MoE split)
          * router_topk[ℓ]     = the top-k expert ids + gate scores the router selects at this pos.
        Plus per (seq,pos,arm): the actual final-logit row at `cand` (DLA sanity + induced shift),
        rms_final, and the injected steering vector's OWN direct DLA (the gL10 direct path).

        Capture/steering hook order: the steering hook is registered FIRST so the resid_post capture
        at the injected layer sees the POST-injection value (so the L10 lens includes gL10 directly).

        request = {sequences, positions(per-seq), steering, candidate_ids, capture_layers(default all),
                   micro_batch}
        """
        torch = self.torch
        import torch.nn.functional as Fnn
        t_start = time.monotonic()
        sequences = request["sequences"]
        positions = request["positions"]
        steering = request["steering"]
        cand = list(request["candidate_ids"])
        cap_layers = request.get("capture_layers") or list(range(N_LAYERS))
        cap_layers = [int(L) for L in cap_layers]
        mb = int(request.get("micro_batch", 2))
        eps = float(self.model.config.rms_norm_eps)
        steer_layers = {int(inj["layer"]) for inj in steering} if steering else set()

        Wf = self.model.lm_head.weight.float()                 # [V, H]
        nrm = self.model.model.norm.weight.float()             # [H]
        cand_t = torch.tensor(cand, dtype=torch.long, device="cuda")
        Wc = Wf[cand_t]                                         # [n_cand, H]
        Wc_n = Wc * nrm.unsqueeze(0)                            # fold final-norm diagonal: [n_cand, H]
        # gL10's own direct DLA contribution (a fixed component added at its layer)
        steer_vecs = ([torch.tensor(inj["vector"], dtype=torch.float32, device="cuda")
                       for inj in steering] if steering else [])

        def lens_logits(resid_vec):  # full RMSNorm lens of a residual vector -> candidate logits
            v = resid_vec.float()
            rms = torch.sqrt(v.pow(2).mean() + eps)
            normed = (v / rms) * nrm
            return (Wc @ normed)  # [n_cand]

        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        per = {}  # i -> pos -> arm -> dict
        for b0 in range(0, len(order), mb):
            idxs = order[b0:b0 + mb]
            maxlen = max(len(sequences[i]) for i in idxs)
            input_ids = torch.full((len(idxs), maxlen), PAD_TOKEN_ID, dtype=torch.long)
            attn = torch.zeros((len(idxs), maxlen), dtype=torch.long)
            for r, i in enumerate(idxs):
                s = sequences[i]
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)  # RIGHT pad
                attn[r, :len(s)] = 1
            input_ids = input_ids.to("cuda")
            attn = attn.to("cuda")

            def run(with_steer):
                store = {}
                handles = []
                cnt = [0]
                # steering FIRST so the resid_post capture at the injected layer is post-injection
                if with_steer and steering:
                    for inj in steering:
                        vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                        handles.append(self.layers[int(inj["layer"])].register_forward_hook(
                            self._make_steering_hook(vec, cnt, attn)))
                for L in cap_layers:
                    handles.append(self.layers[L].register_forward_hook(
                        self._make_capture_hook(store, ("resid", L))))
                    handles.append(self.layers[L].self_attn.register_forward_hook(
                        self._make_capture_hook(store, ("attn", L))))
                    handles.append(self.layers[L].mlp.register_forward_hook(
                        self._make_capture_hook(store, ("mlp", L))))

                    def router_hook(module, args, output, _L=L):
                        store[("ridx", _L)] = output[1].detach()   # [B*T, top_k]
                        store[("rsc", _L)] = output[0].detach()    # [B*T, n_experts]
                    handles.append(self.layers[L].mlp.router.register_forward_hook(router_hook))
                # capture resid_post of the LAST layer for rms_final (input to final norm)
                if (N_LAYERS - 1) not in cap_layers:
                    handles.append(self.layers[N_LAYERS - 1].register_forward_hook(
                        self._make_capture_hook(store, ("resid", N_LAYERS - 1))))
                try:
                    with torch.no_grad():
                        out = self.model(input_ids, attention_mask=attn, use_cache=False)
                    logits = out.logits.float()  # [bs, T, V]
                finally:
                    for h in handles:
                        h.remove()
                return store, logits

            for arm, with_steer in (("base", False), ("steer", True)):
                store, logits = run(with_steer)
                T = maxlen
                for r, i in enumerate(idxs):
                    for p in positions[i]:
                        d = per.setdefault(i, {}).setdefault(p, {}).setdefault(arm, {})
                        resid_final = store[("resid", N_LAYERS - 1)][r, p].float()
                        rms_final = torch.sqrt(resid_final.pow(2).mean() + eps)
                        d["rms_final"] = float(rms_final.item())
                        d["cand_logit"] = logits[r, p][cand_t].cpu().tolist()
                        rl, ad, md = {}, {}, {}
                        for L in cap_layers:
                            rl[str(L)] = lens_logits(store[("resid", L)][r, p]).cpu().tolist()
                            aΔ = store[("attn", L)][r, p].float()
                            mΔ = store[("mlp", L)][r, p].float()
                            ad[str(L)] = ((Wc_n @ aΔ) / rms_final).cpu().tolist()
                            md[str(L)] = ((Wc_n @ mΔ) / rms_final).cpu().tolist()
                        d["resid_lens"] = rl
                        d["attn_dla"] = ad
                        d["mlp_dla"] = md
                        # router top-k (flat index = r*T + p)
                        flat = r * T + p
                        rt = {}
                        for L in cap_layers:
                            ids_ = store[("ridx", L)][flat].cpu().tolist()
                            sc_ = store[("rsc", L)][flat]
                            rt[str(L)] = {"experts": [int(x) for x in ids_],
                                          "gates": [float(sc_[int(x)].item()) for x in ids_]}
                        d["router"] = rt
                        # gL10 direct DLA (only meaningful for the steer arm)
                        if arm == "steer" and steer_vecs:
                            d["steer_dla"] = [((Wc_n @ sv) / rms_final).cpu().tolist()
                                              for sv in steer_vecs]
                del store, logits
                torch.cuda.empty_cache()

        # flatten into a list aligned with (seq,pos)
        results = []
        for i, poss in enumerate(positions):
            for p in poss:
                results.append({"seq_idx": i, "pos": int(p), **per[i][p]})
        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"results": results, "candidate_ids": cand, "capture_layers": cap_layers,
                "full_attention_layers": [L for L in range(N_LAYERS)
                                          if self.model.config.layer_types[L] == "full_attention"],
                "steer_layers": sorted(steer_layers), "elapsed": elapsed}

    # -- per-head attention mass onto key spans + the sink (the gating test) ----
    @modal.method()
    def capture_attn(self, request: dict) -> dict:
        """Teacher-force a batch and capture, at requested query positions, each head's POST-SOFTMAX
        attention mass onto named KEY spans (the in-context instruction, the prompt, the model's own
        reasoning prefix, the first token) and onto the per-head ATTENTION SINK, WITH vs WITHOUT a
        steering injection. The gating hypothesis ("gL10 makes the model attend to & act on the
        instruction it under-uses") predicts gL10 RAISES attention to the instruction span at the
        full-attention layers. (gpt-oss eager attention returns the post-softmax weights with the SINK
        column dropped, so per query: sink_mass = 1 − Σ_kv weights.)

        request = {sequences, positions(per-seq), steering, span_ranges{seq->{name:[a,b]}},
                   capture_layers(default = full-attention layers), micro_batch}
        Returns per (seq,pos,layer): per-head mass to each named span + sink + rowsum, base & steer.
        """
        torch = self.torch
        t_start = time.monotonic()
        sequences = request["sequences"]
        positions = request["positions"]
        steering = request.get("steering")
        span_ranges = request["span_ranges"]   # {str(seq_idx): {name: [a,b]}}
        full_layers = [L for L in range(N_LAYERS)
                       if self.model.config.layer_types[L] == "full_attention"]
        cap_layers = [int(L) for L in (request.get("capture_layers") or full_layers)]
        mb = int(request.get("micro_batch", 2))

        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        per = {}
        for b0 in range(0, len(order), mb):
            idxs = order[b0:b0 + mb]
            maxlen = max(len(sequences[i]) for i in idxs)
            input_ids = torch.full((len(idxs), maxlen), PAD_TOKEN_ID, dtype=torch.long)
            attn = torch.zeros((len(idxs), maxlen), dtype=torch.long)
            for r, i in enumerate(idxs):
                s = sequences[i]
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[r, :len(s)] = 1
            input_ids = input_ids.to("cuda")
            attn = attn.to("cuda")

            def run(with_steer):
                store = {}
                handles = []
                cnt = [0]
                if with_steer and steering:
                    for inj in steering:
                        vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                        handles.append(self.layers[int(inj["layer"])].register_forward_hook(
                            self._make_steering_hook(vec, cnt, attn)))
                for L in cap_layers:
                    def aw_hook(module, args, output, _L=L):
                        store[_L] = output[1].detach()   # [B, n_heads, q, kv] (sink dropped)
                    handles.append(self.layers[L].self_attn.register_forward_hook(aw_hook))
                try:
                    with torch.no_grad():
                        self.model(input_ids, attention_mask=attn, use_cache=False)
                finally:
                    for h in handles:
                        h.remove()
                return store

            for arm in ("base", "steer"):
                if arm == "steer" and not steering:
                    continue
                store = run(arm == "steer")
                for r, i in enumerate(idxs):
                    sr = span_ranges[str(i)]
                    for p in positions[i]:
                        d = per.setdefault(i, {}).setdefault(p, {}).setdefault(arm, {})
                        for L in cap_layers:
                            aw = store[L][r, :, p, :].float()   # [n_heads, kv]
                            rowsum = aw.sum(-1)                  # [n_heads]
                            entry = {"sink": (1.0 - rowsum).cpu().tolist(),
                                     "rowsum": rowsum.cpu().tolist()}
                            for name, v in sr.items():
                                # v is either a contiguous [a,b] span or {"ids":[...]} (explicit,
                                # possibly disjoint token indices).
                                if isinstance(v, dict):
                                    idx = torch.tensor([int(x) for x in v["ids"]],
                                                       dtype=torch.long, device=aw.device)
                                    entry[name] = (aw[:, idx].sum(-1).cpu().tolist()
                                                   if idx.numel() else [0.0] * aw.shape[0])
                                else:
                                    a, b = int(v[0]), int(v[1])
                                    entry[name] = aw[:, a:b].sum(-1).cpu().tolist()
                            d[str(L)] = entry
                del store
                torch.cuda.empty_cache()

        results = []
        for i, poss in enumerate(positions):
            for p in poss:
                results.append({"seq_idx": i, "pos": int(p), **per.get(i, {}).get(p, {})})
        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"results": results, "capture_layers": cap_layers,
                "full_attention_layers": full_layers, "elapsed": elapsed}

    # -- full attention-weight ROWS for chosen (layer,head) — for eyeballing real maps --
    @modal.method()
    def attn_row(self, request: dict) -> dict:
        """For chosen (layer, head) pairs, return the FULL post-softmax attention weight row at the
        read position (over all key positions), base vs gL10-steered, for a few sequences — so the
        actual attention map can be inspected by eye (which tokens a recruited head attends to).
        request = {sequences, positions, steering, lh_pairs:[[layer,head],...]}"""
        torch = self.torch
        t_start = time.monotonic()
        sequences = request["sequences"]
        positions = request["positions"]
        steering = request.get("steering")
        lh = [(int(L), int(h)) for L, h in request["lh_pairs"]]
        layers = sorted({L for L, _ in lh})
        out = []
        for i, seq in enumerate(sequences):
            input_ids = torch.tensor([seq], device="cuda")
            attn = torch.ones((1, len(seq)), dtype=torch.long, device="cuda")

            def run(with_steer):
                store = {}
                handles = []
                cnt = [0]
                if with_steer and steering:
                    for inj in steering:
                        vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                        handles.append(self.layers[int(inj["layer"])].register_forward_hook(
                            self._make_steering_hook(vec, cnt, attn)))
                for L in layers:
                    def aw(module, args, output, _L=L):
                        store[_L] = output[1].detach()
                    handles.append(self.layers[L].self_attn.register_forward_hook(aw))
                try:
                    with torch.no_grad():
                        self.model(input_ids, attention_mask=attn, use_cache=False)
                finally:
                    for h in handles:
                        h.remove()
                return store
            sb = run(False)
            ss = run(True) if steering else sb
            for p in positions[i]:
                for (L, h) in lh:
                    out.append({"seq_idx": i, "pos": int(p), "layer": L, "head": h,
                                "base_row": sb[L][0, h, p, :].float().cpu().tolist(),
                                "steer_row": ss[L][0, h, p, :].float().cpu().tolist()})
        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"results": out, "elapsed": elapsed}

    # -- QK-pattern vs OV-value patching + sub-block ablation ----
    @modal.method()
    def patch_attn(self, request: dict) -> dict:
        """Causal patching to resolve modulation (attention gating/routing) vs a downstream-gated
        additive feature, and to test mediating-component NECESSITY.

        patch_spec modes:
          * "qkov": capture the STEERED attention PATTERN (post-softmax weights) and post-repeat
            VALUE states at the target full-attention layers/heads, then run an UNSTEERED (base) pass
            in which those heads use {steered pattern + base values} ("pattern") or {base pattern +
            steered values} ("value"). If the form-logit shift follows the PATTERN ⇒ gating/routing;
            if it follows the VALUES ⇒ downstream-gated additive feature. patch_spec = {mode:"qkov",
            which:"pattern"|"value"|"both", layers:[...], heads:"all"|[...]}.
          * "ablate": in the STEERED run, FREEZE the listed sub-blocks to their BASE-run output
            (sever their response to the steering) and measure how much of the form-logit shift
            remains (necessity). patch_spec = {mode:"ablate", targets:[[layer,"attn"|"mlp"], ...]}.
          * "expert": in the STEERED run, at the listed (layer, expert) pairs, FORCE-DROP or FORCE-IN
            those experts in the router top-k and renormalize gates (MoE causal). patch_spec =
            {mode:"expert", drop:[[layer,expert],...], add:[[layer,expert],...]}.
        A passthrough self-check (recompute attn output from base pattern+values) asserts the
        recompute reproduces the model's own attention output.
        request = {sequences, positions, steering, candidate_ids, patch_spec, micro_batch}
        """
        torch = self.torch
        t_start = time.monotonic()
        sequences = request["sequences"]
        positions = request["positions"]
        steering = request["steering"]
        cand = list(request["candidate_ids"])
        spec = request["patch_spec"]
        mode = spec["mode"]
        mb = int(request.get("micro_batch", 2))
        cand_t = torch.tensor(cand, dtype=torch.long, device="cuda")
        eps = float(self.model.config.rms_norm_eps)

        def steer_handles(attn_mask):
            hs = []
            cnt = [0]
            for inj in steering:
                vec = torch.tensor(inj["vector"], dtype=torch.bfloat16, device="cuda")
                hs.append(self.layers[int(inj["layer"])].register_forward_hook(
                    self._make_steering_hook(vec, cnt, attn_mask)))
            return hs

        def cand_rows(logits, idxs, positions_):
            out = {}
            for r, i in enumerate(idxs):
                for p in positions_[i]:
                    out[(i, p)] = logits[r, p][cand_t].cpu().tolist()
            return out

        def repeat_kv(x, n_rep):
            b, nkv, t, hd = x.shape
            if n_rep == 1:
                return x
            return x[:, :, None, :, :].expand(b, nkv, n_rep, t, hd).reshape(b, nkv * n_rep, t, hd)

        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        rec = {"base": {}, "steer": {}, "patched": {}}
        passthrough_err = 0.0
        for b0 in range(0, len(order), mb):
            idxs = order[b0:b0 + mb]
            maxlen = max(len(sequences[i]) for i in idxs)
            input_ids = torch.full((len(idxs), maxlen), PAD_TOKEN_ID, dtype=torch.long)
            attn = torch.zeros((len(idxs), maxlen), dtype=torch.long)
            for r, i in enumerate(idxs):
                s = sequences[i]
                input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
                attn[r, :len(s)] = 1
            input_ids = input_ids.to("cuda")
            attn = attn.to("cuda")

            def plain(with_steer):
                hs = steer_handles(attn) if with_steer else []
                try:
                    with torch.no_grad():
                        out = self.model(input_ids, attention_mask=attn, use_cache=False)
                    return out.logits.float()
                finally:
                    for h in hs:
                        h.remove()

            base_logits = plain(False)
            rec["base"].update(cand_rows(base_logits, idxs, positions))
            del base_logits
            steer_logits = plain(True)
            rec["steer"].update(cand_rows(steer_logits, idxs, positions))
            del steer_logits
            torch.cuda.empty_cache()

            if mode == "qkov":
                which = spec.get("which", "both")
                tgt_layers = [int(L) for L in spec["layers"]]
                heads = spec.get("heads", "all")
                # 1) STEERED capture of pattern (scores) + post-repeat values at target layers
                cap = {}
                hs = steer_handles(attn)
                for L in tgt_layers:
                    am = self.layers[L].self_attn

                    def vhook(module, args, output, _L=L, _am=am):
                        v = output  # v_proj output [B, T, n_kv*hd]
                        b, t, _ = v.shape
                        hd = _am.head_dim
                        nkv = _am.config.num_key_value_heads
                        vv = v.view(b, t, nkv, hd).transpose(1, 2)  # [B, nkv, T, hd]
                        cap.setdefault(_L, {})["V"] = repeat_kv(vv, _am.num_key_value_groups).detach()
                    hs.append(am.v_proj.register_forward_hook(vhook))

                    def ahook(module, args, output, _L=L):
                        cap.setdefault(_L, {})["A"] = output[1].detach()  # [B, H, T, T]
                    hs.append(am.register_forward_hook(ahook))
                try:
                    with torch.no_grad():
                        self.model(input_ids, attention_mask=attn, use_cache=False)
                finally:
                    for h in hs:
                        h.remove()

                # 2) BASE patched run: recompute target-head attn output with mixed pattern/values
                def make_patch(_L):
                    am = self.layers[_L].self_attn
                    cur = {}

                    def vhook(module, args, output):
                        b, t, _ = output.shape
                        hd = am.head_dim
                        nkv = am.config.num_key_value_heads
                        vv = output.view(b, t, nkv, hd).transpose(1, 2)
                        cur["Vbase"] = repeat_kv(vv, am.num_key_value_groups)

                    def ohook(module, args, output):
                        nonlocal passthrough_err
                        A_base = output[1]                 # [B, H, Tq, Tkv]
                        V_base = cur["Vbase"]              # [B, H, Tkv, hd]
                        A_st = cap[_L]["A"]
                        V_st = cap[_L]["V"]
                        B, Hh, Tq, Tkv = A_base.shape
                        hd = V_base.shape[-1]
                        A_use = A_base.clone()
                        V_use = V_base.clone()
                        hmask = (range(Hh) if heads == "all" else [int(x) for x in heads])
                        if which in ("pattern", "both"):
                            for h in hmask:
                                A_use[:, h] = A_st[:, h]
                        if which in ("value", "both"):
                            for h in hmask:
                                V_use[:, h] = V_st[:, h]
                        ctx = torch.matmul(A_use, V_use)          # [B, H, Tq, hd]
                        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, Hh * hd)
                        new_out = am.o_proj(ctx)
                        # passthrough check (no override would reproduce output[0])
                        if which == "checkonly":
                            chk = torch.matmul(A_base, V_base).transpose(1, 2).contiguous().view(B, Tq, Hh * hd)
                            chk = am.o_proj(chk)
                            passthrough_err = max(passthrough_err,
                                                  float((chk - output[0]).abs().max().item()))
                            return output
                        return (new_out,) + tuple(output[1:])
                    return am, vhook, ohook

                handles = []
                for L in tgt_layers:
                    am, vhook, ohook = make_patch(L)
                    handles.append(am.v_proj.register_forward_hook(vhook))
                    handles.append(am.register_forward_hook(ohook))
                try:
                    with torch.no_grad():
                        out = self.model(input_ids, attention_mask=attn, use_cache=False)
                    rec["patched"].update(cand_rows(out.logits.float(), idxs, positions))
                finally:
                    for h in handles:
                        h.remove()
                del cap
                torch.cuda.empty_cache()

            elif mode == "mask_instr":
                # Surgical necessity test: in the STEERED run, zero the target heads' attention onto
                # the instruction span and renormalize (preserve rowsum → sink unchanged), measuring
                # whether the form-logit shift collapses. Tests attention-TO-THE-INSTRUCTION
                # specifically (vs freezing the whole attention sub-block).
                tgt_layers = [int(L) for L in spec["layers"]]
                heads = spec.get("heads", "all")
                span_ranges = spec["span_ranges"]   # {str(seq_idx): [a,b]}
                handles = steer_handles(attn)

                def make_mask(_L):
                    am = self.layers[_L].self_attn
                    cur = {}

                    def vhook(module, args, output):
                        b, t, _ = output.shape
                        hd = am.head_dim
                        nkv = am.config.num_key_value_heads
                        vv = output.view(b, t, nkv, hd).transpose(1, 2)
                        cur["V"] = repeat_kv(vv, am.num_key_value_groups)

                    def ohook(module, args, output):
                        A = output[1].clone()      # [B, H, Tq, Tkv]
                        V = cur["V"]
                        B, Hh, Tq, Tkv = A.shape
                        hmask = (range(Hh) if heads == "all" else [int(x) for x in heads])
                        for r, i in enumerate(idxs):
                            sp = span_ranges.get(str(i))
                            if sp is None:
                                continue
                            # sp is either contiguous [a,b] or {"ids":[...]} (explicit sub-spans)
                            if isinstance(sp, dict):
                                col = torch.tensor([int(x) for x in sp["ids"]],
                                                   dtype=torch.long, device=A.device)
                                if col.numel() == 0:
                                    continue
                            else:
                                col = torch.arange(int(sp[0]), int(sp[1]), device=A.device)
                            for h in hmask:
                                row = A[r, h]                      # [Tq, Tkv]
                                old = row.sum(-1, keepdim=True)
                                row[:, col] = 0.0
                                new = row.sum(-1, keepdim=True).clamp_min(1e-9)
                                A[r, h] = row * (old / new)        # preserve rowsum (sink unchanged)
                        ctx = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, Tq, -1)
                        return (am.o_proj(ctx),) + tuple(output[1:])
                    return am, vhook, ohook
                for L in tgt_layers:
                    am, vhook, ohook = make_mask(L)
                    handles.append(am.v_proj.register_forward_hook(vhook))
                    handles.append(am.register_forward_hook(ohook))
                try:
                    with torch.no_grad():
                        out = self.model(input_ids, attention_mask=attn, use_cache=False)
                    rec["patched"].update(cand_rows(out.logits.float(), idxs, positions))
                finally:
                    for h in handles:
                        h.remove()
                torch.cuda.empty_cache()

            elif mode == "ablate":
                targets = [(int(L), kind) for L, kind in spec["targets"]]
                # base sub-block outputs (whole tensors) to freeze into the steered run
                store = {}
                handles = []
                for L, kind in targets:
                    mod = self.layers[L].self_attn if kind == "attn" else self.layers[L].mlp

                    def cap_hook(module, args, output, _k=(L, kind)):
                        store[_k] = (output[0] if isinstance(output, tuple) else output).detach()
                    handles.append(mod.register_forward_hook(cap_hook))
                try:
                    with torch.no_grad():
                        self.model(input_ids, attention_mask=attn, use_cache=False)
                finally:
                    for h in handles:
                        h.remove()
                # steered run with target sub-blocks frozen to base output
                handles = steer_handles(attn)
                for L, kind in targets:
                    mod = self.layers[L].self_attn if kind == "attn" else self.layers[L].mlp

                    def froze(module, args, output, _k=(L, kind)):
                        base_out = store[_k]
                        if isinstance(output, tuple):
                            return (base_out,) + tuple(output[1:])
                        return base_out
                    handles.append(mod.register_forward_hook(froze))
                try:
                    with torch.no_grad():
                        out = self.model(input_ids, attention_mask=attn, use_cache=False)
                    rec["patched"].update(cand_rows(out.logits.float(), idxs, positions))
                finally:
                    for h in handles:
                        h.remove()
                del store
                torch.cuda.empty_cache()

            elif mode == "expert":
                drop = {(int(L), int(e)) for L, e in spec.get("drop", [])}
                add = {(int(L), int(e)) for L, e in spec.get("add", [])}
                drop_layers = {L for L, _ in drop} | {L for L, _ in add}
                handles = steer_handles(attn)
                for L in drop_layers:
                    router = self.layers[L].mlp.router

                    def rhook(module, args, output, _L=L):
                        scores, idx = output[0], output[1]   # [N,32], [N,topk]
                        scores = scores.clone()
                        for (LL, e) in drop:
                            if LL == _L:
                                scores[:, e] = 0.0
                        for (LL, e) in add:
                            if LL == _L:
                                # force expert e to receive the max current gate (approx force-in)
                                scores[:, e] = scores.max(dim=-1, keepdim=True).values.squeeze(-1)
                        ssum = scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                        scores = scores / ssum
                        return (scores, idx)
                    handles.append(router.register_forward_hook(rhook))
                try:
                    with torch.no_grad():
                        out = self.model(input_ids, attention_mask=attn, use_cache=False)
                    rec["patched"].update(cand_rows(out.logits.float(), idxs, positions))
                finally:
                    for h in handles:
                        h.remove()
                torch.cuda.empty_cache()

        results = []
        for i, poss in enumerate(positions):
            for p in poss:
                results.append({"seq_idx": i, "pos": int(p),
                                "base": rec["base"][(i, p)], "steer": rec["steer"][(i, p)],
                                "patched": rec["patched"].get((i, p))})
        elapsed = time.monotonic() - t_start
        self.compute_seconds += elapsed
        return {"results": results, "candidate_ids": cand, "mode": mode,
                "passthrough_err": passthrough_err, "elapsed": elapsed}


# ---------------------------------------------------------------------------
# Local orchestration helpers (caching + cost tracking)
# ---------------------------------------------------------------------------
CACHE_UUID = "gpt-oss-infer-v1-2026-05"
# Bumped only when the STEERING-hook code changes (invalidates steered generations only, not
# unsteered ones, since we add it to the cache key only when steering is present).
STEER_HOOK_VERSION = 2


def _cache_and_cost():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cost_tracker import CostTracker
    from file_cache import FileCache
    cache = FileCache(os.path.join(os.path.dirname(os.path.abspath(__file__)), "file_cache_dir"))
    tracker = CostTracker(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "total_cost.jsonl")),
                          run_description="gpt-oss-infer")
    return cache, tracker


def generate(obj, request: dict, cache, tracker, *, sample_idx: int = 0, assert_cached: bool = False,
             weights_id: str = "base") -> dict:
    """Cached + cost-tracked wrapper around ``GptOss.generate``. ``obj`` is an instantiated
    remote class handle (created inside ``app.run()``). ``weights_id`` namespaces fine-tuned-model
    generations in the cache (added to the key ONLY when != "base", so base cache stays valid)."""
    cache_key = {
        "fn": "generate",
        "model": MODEL_NAME,
        "request": request,
        "gpu": GPU_TYPE,
        "image_id": image.object_id,
        "sample_idx": sample_idx,
        "this_call_uuid": CACHE_UUID,
        **({"steer_hook_version": STEER_HOOK_VERSION} if request.get("steering") else {}),
        **({"weights_id": weights_id} if weights_id != "base" else {}),
    }

    def run():
        with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
            res = obj.generate.remote(request)
            t.elapsed = res["elapsed"]
            return res

    res, _ = cache.get_or_compute_set(cache_key, run, assert_cached=assert_cached)
    return res


def capture_activations(obj, request: dict, cache, tracker, *, sample_idx: int = 0, assert_cached: bool = False) -> dict:
    cache_key = {
        "fn": "capture",
        "model": MODEL_NAME,
        "request": request,
        "gpu": GPU_TYPE,
        "image_id": image.object_id,
        "sample_idx": sample_idx,
        "this_call_uuid": CACHE_UUID,
    }

    def run():
        with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
            res = obj.capture.remote(request)
            t.elapsed = res["elapsed"]
            return res

    res, _ = cache.get_or_compute_set(cache_key, run, assert_cached=assert_cached)
    return res


def generate_many(obj, prompts: list[list[int]], gen_params: dict, cache, tracker, *,
                  steering=None, batch_size: int = 10, sample_idx: int = 0,
                  assert_cached: bool = False, weights_id: str = "base") -> list[dict]:
    """Generate for many prompts with PER-PROMPT caching but BATCHED compute.

    Each prompt's result is cached individually (keyed by the single prompt + gen params +
    steering), so cache hits are independent of batch composition and re-runs over different
    task subsets still hit the cache. Cache misses are computed in batches of ``batch_size``
    for throughput. (Greedy decoding is not bit-identical across batch sizes due to numerical
    non-determinism, but each cached result is a valid generation for its prompt.)
    """
    gp = {
        "max_new_tokens": int(gen_params.get("max_new_tokens", 512)),
        "temperature": float(gen_params.get("temperature", 0.0)),
        "top_p": float(gen_params.get("top_p", 1.0)),
        "seed": int(gen_params.get("seed", 0)),
    }
    results: list = [None] * len(prompts)
    keys = []
    miss_idx = []
    for i, p in enumerate(prompts):
        key = {"fn": "generate_one", "model": MODEL_NAME, "prompt_token_ids": p,
               **gp, "steering": steering, "gpu": GPU_TYPE, "image_id": image.object_id,
               "sample_idx": sample_idx, "this_call_uuid": CACHE_UUID,
               **({"steer_hook_version": STEER_HOOK_VERSION} if steering else {}),
               **({"weights_id": weights_id} if weights_id != "base" else {})}
        keys.append(key)
        cached = cache.get(key)
        if cached is not None:
            results[i] = cached
        else:
            miss_idx.append(i)
    if miss_idx and assert_cached:
        raise RuntimeError(f"assert_cached=True but {len(miss_idx)} prompts not cached")

    for b in range(0, len(miss_idx), batch_size):
        idxs = miss_idx[b:b + batch_size]
        req = {"prompt_token_ids": [prompts[i] for i in idxs], **gp}
        if steering:
            req["steering"] = steering
        with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
            res = obj.generate.remote(req)
            t.elapsed = res["elapsed"]
        for j, i in enumerate(idxs):
            g = res["generations"][j]
            val = {"token_ids": g["token_ids"], "n_tokens": g["n_tokens"],
                   "batch_tokens_per_sec": res["tokens_per_sec"], "batch_size": len(idxs),
                   "hook_fire_count": res["hook_fire_count"]}
            cache.get_or_set(keys[i], val)
            results[i] = val
    return results


# Bumped only when the capture_spans hook/pooling code changes (invalidates captured directions).
CAPTURE_SPANS_VERSION = 1


def capture_span_residuals(obj, sequences: list[list[int]], spans: list, capture_layers: list[int],
                           cache, tracker, *, batch_size: int = 16, token_budget: int = 9000,
                           sample_idx: int = 0, assert_cached: bool = False,
                           weights_id: str = "base", steering=None) -> list[dict]:
    """Cached + cost-tracked wrapper around ``GptOss.capture_spans`` with PER-SEQUENCE caching but
    BATCHED compute. Each sequence's per-layer mean-analysis residual is cached individually (keyed
    by the single sequence + span + capture_layers), so cache hits are independent of batch
    composition. Returns a list aligned with ``sequences``; each entry is
    ``{str(layer): {"mean": [...H], "mean_token_norm": float, "span_len": int, "has_nan": bool}}``.
    Right-padded forward passes are deterministic + padding-invariant, so length-sorting the misses
    for throughput does not change cached values."""
    cl = list(capture_layers)
    steer_sig = None
    if steering:
        steer_sig = _vec_hash([float(x) for inj in steering for x in inj["vector"]] +
                              [float(inj["layer"]) for inj in steering])
    results: list = [None] * len(sequences)
    keys = []
    miss = []
    for i, (s, sp) in enumerate(zip(sequences, spans)):
        key = {"fn": "capture_spans_one", "model": MODEL_NAME, "sequence": s,
               "span": [int(sp[0]), int(sp[1])], "capture_layers": cl, "gpu": GPU_TYPE,
               "image_id": image.object_id, "sample_idx": sample_idx,
               "this_call_uuid": CACHE_UUID, "capture_version": CAPTURE_SPANS_VERSION,
               **({"weights_id": weights_id} if weights_id != "base" else {}),
               **({"steer_sig": steer_sig, "steer_hook_version": STEER_HOOK_VERSION} if steering else {})}
        keys.append(key)
        cached = cache.get(key)
        if cached is not None:
            results[i] = cached
        else:
            miss.append(i)
    if miss and assert_cached:
        raise RuntimeError(f"assert_cached=True but {len(miss)} sequences not cached")
    miss.sort(key=lambda i: len(sequences[i]))  # length-sort for padding efficiency (safe)
    # Token-budget batching: cap (batch_max_len * batch_size) so long sequences use SMALL batches
    # (eager attention is O(seq^2) and OOMs at fixed large bs on the longest traces).
    batches = []
    cur = []
    cur_max = 0
    for i in miss:
        L = len(sequences[i])
        nmax = max(cur_max, L)
        if cur and (nmax * (len(cur) + 1) > token_budget or len(cur) >= batch_size):
            batches.append(cur)
            cur, cur_max = [], 0
            nmax = L
        cur.append(i)
        cur_max = nmax
    if cur:
        batches.append(cur)
    for idxs in batches:
        req = {"sequences": [sequences[i] for i in idxs], "spans": [spans[i] for i in idxs],
               "capture_layers": cl}
        if steering:
            req["steering"] = steering
        with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
            res = obj.capture_spans.remote(req)
            t.elapsed = res["elapsed"]
        for j, i in enumerate(idxs):
            val = {str(L): res["layers"][str(L)][j] for L in cl}
            cache.get_or_set(keys[i], val)
            results[i] = val
    return results


# Bumped only when the train_steer training code changes (invalidates trained-vector artifacts).
TRAIN_STEER_VERSION = 1


def train_steering_vectors(obj, sequences, comp_starts, config: dict, data_hash: str, cache, tracker,
                           *, assert_cached: bool = False) -> dict:
    """Cached + cost-tracked wrapper around ``GptOss.train_steer``. The (expensive) gradient-training
    run is cached on (config + data_hash + image_id) so re-runs are free. ``data_hash`` must be a
    deterministic hash of (sequences, comp_starts, max_length) — the actual training data. Returns
    the train_steer dict (trained vectors + loss curve)."""
    cache_key = {
        "fn": "train_steer", "model": MODEL_NAME, "gpu": GPU_TYPE,
        "image_id": image.object_id, "this_call_uuid": CACHE_UUID,
        "train_steer_version": TRAIN_STEER_VERSION,
        "config": config, "data_hash": data_hash,
    }

    def run():
        request = {"sequences": sequences, "comp_starts": comp_starts, **config}
        with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
            res = obj.train_steer.remote(request)
            t.elapsed = res["elapsed"]
            return res

    res, _ = cache.get_or_compute_set(cache_key, run, assert_cached=assert_cached)
    return res


def model_info(obj, sample_prompt_token_ids: list[int], cache, tracker, *, assert_cached: bool = False) -> dict:
    cache_key = {
        "fn": "info",
        "model": MODEL_NAME,
        "sample_prompt": sample_prompt_token_ids,
        "gpu": GPU_TYPE,
        "image_id": image.object_id,
        "this_call_uuid": CACHE_UUID,
    }

    def run():
        with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
            res = obj.info.remote(sample_prompt_token_ids)
            t.elapsed = res.get("load_seconds", 0) + 2.0  # load dominates this one-off call
            return res

    res, _ = cache.get_or_compute_set(cache_key, run, assert_cached=assert_cached)
    return res


# Bumped only when lens_project / induced_shift bodies change (invalidates cached projections).
LENS_VERSION = 2


def _vec_hash(v) -> str:
    import hashlib
    import numpy as _np
    return hashlib.sha256(_np.asarray(v, dtype=_np.float32).tobytes()).hexdigest()[:16]


def lens_project(obj, vecs: list, names: list, full_names: list, cache, tracker, *,
                 topk: int = 100, assert_cached: bool = False) -> list:
    """Cached wrapper around ``GptOss.lens_project`` — PER-VECTOR cache (keyed by vec hash + topk +
    whether the full arrays were requested). Returns a list aligned with ``vecs``; each entry is the
    per-vec dict from the remote method (top/bottom tokens + optional full arrays). ``names`` is for
    logging only; ``full_names`` = subset of names for which to also fetch the full [V] arrays."""
    import numpy as _np
    full_set = set(full_names)
    results = [None] * len(vecs)
    miss = []
    keys = []
    for i, (v, nm) in enumerate(zip(vecs, names)):
        want_full = nm in full_set
        key = {"fn": "lens_project", "model": MODEL_NAME, "gpu": GPU_TYPE,
               "image_id": image.object_id, "this_call_uuid": CACHE_UUID,
               "lens_version": LENS_VERSION, "vec_hash": _vec_hash(v), "topk": topk,
               "full": want_full}
        keys.append(key)
        cached = cache.get(key)
        if cached is not None:
            results[i] = cached
        else:
            miss.append(i)
    if miss and assert_cached:
        raise RuntimeError(f"assert_cached=True but {len(miss)} lens vecs not cached")
    # meta (vocab size / tied-unembedding / norm-weight) is cached separately so it survives
    # fully-cached re-runs (the remote is not called when everything hits).
    meta_key = {"fn": "lens_meta", "model": MODEL_NAME, "image_id": image.object_id,
                "this_call_uuid": CACHE_UUID, "lens_version": LENS_VERSION}
    if miss:
        req = {"vecs": [_np.asarray(vecs[i], dtype=_np.float32).tolist() for i in miss],
               "topk": topk,
               "full_idx": [j for j, i in enumerate(miss) if names[i] in full_set]}
        with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
            res = obj.lens_project.remote(req)
            t.elapsed = res["elapsed"] + 30.0  # load dominates this one-off matvec call
        cache.get_or_set(meta_key, {k: res[k] for k in res if k != "vecs"})
        for j, i in enumerate(miss):
            cache.get_or_set(keys[i], res["vecs"][j])
            results[i] = res["vecs"][j]
    lens_project.last_meta = cache.get(meta_key, {})
    return results


def induced_shift(obj, sequences, positions, steering, candidate_ids, cache, tracker, *,
                  topk: int = 200, assert_cached: bool = False) -> list:
    """Cached wrapper around ``GptOss.induced_shift``. Both arms (base+steered) are computed in one
    remote call; cached on (sequences + positions + steering + candidate_ids hash + version)."""
    import numpy as _np
    sig = _vec_hash([float(x) for inj in steering for x in inj["vector"]] +
                    [float(inj["layer"]) for inj in steering])
    cand_hash = _vec_hash([float(c) for c in candidate_ids])
    key = {"fn": "induced_shift", "model": MODEL_NAME, "gpu": GPU_TYPE,
           "image_id": image.object_id, "this_call_uuid": CACHE_UUID,
           "lens_version": LENS_VERSION, "steer_hook_version": STEER_HOOK_VERSION,
           "sequences": sequences, "positions": positions, "steer_sig": sig,
           "cand_hash": cand_hash, "topk": topk}
    cached = cache.get(key)
    if cached is not None:
        return cached
    if assert_cached:
        raise RuntimeError("assert_cached=True but induced_shift not cached")
    req = {"sequences": sequences, "positions": positions, "steering": steering,
           "candidate_ids": list(candidate_ids), "topk": topk}
    with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
        res = obj.induced_shift.remote(req)
        t.elapsed = res["elapsed"]
    val = res["results"]
    cache.get_or_set(key, val)
    return val


# Bumped only when capture_internals / capture_attn / patch_attn bodies change.
# v2: capture_attn / patch_attn(mask_instr) / gen_ablate accept explicit token-index
# sub-spans ({"ids":[...]}); gen_ablate gains mask_spec (behavioral attention sub-span knockout).
MECH_VERSION = 2


def _steer_sig(steering):
    if not steering:
        return "none"
    return _vec_hash([float(x) for inj in steering for x in inj["vector"]] +
                     [float(inj["layer"]) for inj in steering])


def capture_internals(obj, sequences, positions, steering, candidate_ids, cache, tracker, *,
                      capture_layers=None, micro_batch: int = 2, assert_cached: bool = False,
                      weights_id: str = "base") -> dict:
    """Cached wrapper around ``GptOss.capture_internals`` (resid-lens trajectory + sub-block DLA +
    MoE router top-k, base+steered in one call). Caches the whole call on (sequences + positions +
    steering + candidates + capture_layers + version). ``weights_id`` namespaces FT-model captures
    (added to the key only when != "base"). Returns the full remote dict."""
    cl = [int(L) for L in (capture_layers or list(range(N_LAYERS)))]
    key = {"fn": "capture_internals", "model": MODEL_NAME, "gpu": GPU_TYPE,
           "image_id": image.object_id, "this_call_uuid": CACHE_UUID,
           "mech_version": MECH_VERSION, "steer_hook_version": STEER_HOOK_VERSION,
           "sequences": sequences, "positions": positions, "steer_sig": _steer_sig(steering),
           "cand_hash": _vec_hash([float(c) for c in candidate_ids]), "capture_layers": cl,
           **({"weights_id": weights_id} if weights_id != "base" else {})}
    cached = cache.get(key)
    if cached is not None:
        return cached
    if assert_cached:
        raise RuntimeError("assert_cached=True but capture_internals not cached")
    req = {"sequences": sequences, "positions": positions, "steering": steering,
           "candidate_ids": list(candidate_ids), "capture_layers": cl, "micro_batch": micro_batch}
    with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
        res = obj.capture_internals.remote(req)
        t.elapsed = res["elapsed"]
    cache.get_or_set(key, res)
    return res


def capture_attn(obj, sequences, positions, steering, span_ranges, cache, tracker, *,
                 capture_layers=None, micro_batch: int = 2, assert_cached: bool = False,
                 weights_id: str = "base") -> dict:
    """Cached wrapper around ``GptOss.capture_attn`` (per-head attention mass onto named key spans +
    the attention sink, base+steered). ``weights_id`` namespaces FT-model captures."""
    cl = [int(L) for L in (capture_layers or list(range(N_LAYERS)))]

    def _norm_span(v):
        # contiguous [a,b] or {"ids":[...]} explicit index list (sub-spans)
        if isinstance(v, dict):
            return {"ids": [int(x) for x in v["ids"]]}
        return [int(v[0]), int(v[1])]
    sr = {str(i): {k: _norm_span(v) for k, v in d.items()} for i, d in span_ranges.items()}
    key = {"fn": "capture_attn", "model": MODEL_NAME, "gpu": GPU_TYPE,
           "image_id": image.object_id, "this_call_uuid": CACHE_UUID,
           "mech_version": MECH_VERSION, "steer_hook_version": STEER_HOOK_VERSION,
           "sequences": sequences, "positions": positions, "steer_sig": _steer_sig(steering),
           "span_ranges": sr, "capture_layers": cl,
           **({"weights_id": weights_id} if weights_id != "base" else {})}
    cached = cache.get(key)
    if cached is not None:
        return cached
    if assert_cached:
        raise RuntimeError("assert_cached=True but capture_attn not cached")
    req = {"sequences": sequences, "positions": positions, "steering": steering,
           "span_ranges": sr, "capture_layers": cl, "micro_batch": micro_batch}
    with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
        res = obj.capture_attn.remote(req)
        t.elapsed = res["elapsed"]
    cache.get_or_set(key, res)
    return res


def patch_attn(obj, sequences, positions, steering, candidate_ids, patch_spec, cache, tracker, *,
               micro_batch: int = 2, assert_cached: bool = False, weights_id: str = "base") -> dict:
    """Cached wrapper around ``GptOss.patch_attn`` (QK-pattern vs OV-value patching + sub-block
    ablation). ``patch_spec`` = {mode, layers, heads, ...} (see remote method). ``weights_id``
    namespaces FT-model patches."""
    key = {"fn": "patch_attn", "model": MODEL_NAME, "gpu": GPU_TYPE,
           "image_id": image.object_id, "this_call_uuid": CACHE_UUID,
           "mech_version": MECH_VERSION, "steer_hook_version": STEER_HOOK_VERSION,
           "sequences": sequences, "positions": positions, "steer_sig": _steer_sig(steering),
           "cand_hash": _vec_hash([float(c) for c in candidate_ids]),
           "patch_spec": patch_spec,
           **({"weights_id": weights_id} if weights_id != "base" else {})}
    cached = cache.get(key)
    if cached is not None:
        return cached
    if assert_cached:
        raise RuntimeError("assert_cached=True but patch_attn not cached")
    req = {"sequences": sequences, "positions": positions, "steering": steering,
           "candidate_ids": list(candidate_ids), "patch_spec": patch_spec, "micro_batch": micro_batch}
    with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
        res = obj.patch_attn.remote(req)
        t.elapsed = res["elapsed"]
    cache.get_or_set(key, res)
    return res


def attn_row(obj, sequences, positions, steering, lh_pairs, cache, tracker, *,
             assert_cached: bool = False) -> list:
    """Cached wrapper around ``GptOss.attn_row`` (full attention rows for chosen layer/head)."""
    key = {"fn": "attn_row", "model": MODEL_NAME, "gpu": GPU_TYPE, "image_id": image.object_id,
           "this_call_uuid": CACHE_UUID, "mech_version": MECH_VERSION,
           "steer_hook_version": STEER_HOOK_VERSION, "sequences": sequences, "positions": positions,
           "steer_sig": _steer_sig(steering), "lh_pairs": [[int(L), int(h)] for L, h in lh_pairs]}
    cached = cache.get(key)
    if cached is not None:
        return cached
    if assert_cached:
        raise RuntimeError("assert_cached=True but attn_row not cached")
    req = {"sequences": sequences, "positions": positions, "steering": steering,
           "lh_pairs": [[int(L), int(h)] for L, h in lh_pairs]}
    with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
        res = obj.attn_row.remote(req)
        t.elapsed = res["elapsed"]
    cache.get_or_set(key, res["results"])
    return res["results"]


def gen_ablate(obj, sequences, steering, ablate_targets, cache, tracker, *, n_new: int = 64,
               with_steer: bool = True, mask_spec=None, assert_cached: bool = False) -> list:
    """Cached wrapper around ``GptOss.gen_ablate`` (greedy generation with sub-block ablation and/or
    the ``mask_spec`` attention-sub-span knockout)."""
    def _norm_mask(ms):
        if not ms:
            return None
        sr = {}
        for k, v in ms.get("span_ranges", {}).items():
            sr[str(k)] = ({"ids": [int(x) for x in v["ids"]]} if isinstance(v, dict)
                          else [int(v[0]), int(v[1])])
        return {"layers": [int(L) for L in ms["layers"]], "heads": ms.get("heads", "all"),
                "span_ranges": sr}
    ms_norm = _norm_mask(mask_spec)
    key = {"fn": "gen_ablate", "model": MODEL_NAME, "gpu": GPU_TYPE,
           "image_id": image.object_id, "this_call_uuid": CACHE_UUID,
           "mech_version": MECH_VERSION, "steer_hook_version": STEER_HOOK_VERSION,
           "sequences": sequences, "steer_sig": _steer_sig(steering if with_steer else None),
           "ablate_targets": [[int(L), k] for L, k in ablate_targets], "n_new": n_new,
           "with_steer": with_steer,
           **({"mask_spec": ms_norm} if ms_norm else {})}
    cached = cache.get(key)
    if cached is not None:
        return cached
    if assert_cached:
        raise RuntimeError("assert_cached=True but gen_ablate not cached")
    req = {"sequences": sequences, "steering": steering, "ablate_targets": ablate_targets,
           "n_new": n_new, "with_steer": with_steer,
           **({"mask_spec": ms_norm} if ms_norm else {})}
    with tracker.track_modal_gpu(gpu=GPU_TYPE, is_sandbox=False) as t:
        res = obj.gen_ablate.remote(req)
        t.elapsed = res["elapsed"]
    val = res["generations"]
    cache.get_or_set(key, val)
    return val


if __name__ == "__main__":
    # Smoke test: load model, print diagnostics, run one generation.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import harmony_utils as H

    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-cached", action="store_true")
    args = parser.parse_args()

    cache, tracker = _cache_and_cost()
    prompt = H.render_prompt_tokens("What is 2+2? Think step by step, then answer.", reasoning_effort="medium")

    with modal.enable_output(), app.run():
        obj = GptOss()
        info = model_info(obj, prompt, cache, tracker, assert_cached=args.assert_cached)
        print("=== MODEL INFO ===")
        import json
        print(json.dumps({k: v for k, v in info.items() if k != "activation_stats"}, indent=2))
        print("activation_stats (subset):")
        for L, s in list(info["activation_stats"].items()):
            print(f"  layer {L}: {s}")

        req = {"prompt_token_ids": [prompt], "max_new_tokens": 300, "temperature": 0.0}
        res = generate(obj, req, cache, tracker, assert_cached=args.assert_cached)
        print("\n=== GENERATION ===")
        print("tokens/sec:", round(res["tokens_per_sec"], 1), "gen_tokens:", res["total_gen_tokens"],
              "hook_fires:", res["hook_fire_count"])
        parsed = H.parse_channels(res["generations"][0]["token_ids"])
        print("ANALYSIS:", repr(parsed.analysis[:400]))
        print("FINAL:", repr(parsed.final[:200]))
        print("malformed:", parsed.malformed, "has_final:", parsed.has_final)

    print(f"\nModal cost this run: ${tracker.run_cost:.4f}")
