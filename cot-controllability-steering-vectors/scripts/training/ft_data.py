"""Shared training-data prep for the FT pipeline.

Loads an SFT set (compliant / control / plain), and builds Tinker training Datums from EXACT
pre-rendered Harmony tokens (harmony_utils.render_training_example) with a COMPLETION-ONLY loss
mask. Rendering alignment is guaranteed by construction (verified byte-for-byte against the
persisted prompt_token_sha256 by verify_render_alignment.py) — Tinker's gpt_oss renderer (which
injects a 'Current date' line we omit) is bypassed entirely.

The 5 rows with an un-adjudicated gates.meta==null (all from the single long task reasonif_gpqa_212)
are dropped by default (trivial; per INSTRUCTIONS.md).
"""
from __future__ import annotations

import hashlib
import json
import random

import harmony_utils as H

FILES = {
    "compliant": "data/sft_edited_reasoning_full.jsonl",
    "control": "data/sft_raw_trace_control_full.jsonl",
    "plain": "data/sft_plain_full.jsonl",
}


def load_rows(which: str, *, drop_meta_null: bool = True) -> list[dict]:
    path = FILES[which]
    rows = [json.loads(l) for l in open(path)]
    if drop_meta_null:
        # Only the COMPLIANT set has a `meta` gate; the 5 reasonif_gpqa_212 rows have an
        # un-adjudicated meta==null (per INSTRUCTIONS.md). The CONTROL set has NO `meta` key at all
        # (its targets are intentionally non-complying), so we must NOT drop those rows.
        kept = [r for r in rows
                if not ("meta" in r.get("gates", {}) and r["gates"]["meta"] is None)]
        n_drop = len(rows) - len(kept)
        if n_drop:
            print(f"[ft_data] dropped {n_drop} rows with gates.meta==null from {which}")
        rows = kept
    return rows


def control_pairs() -> set:
    """(task_id, instruction_id) pairs present in the raw-trace CONTROL set (the prompts shared by
    BOTH arms). Used to train a MATCHED compliant arm on exactly the shared prompts so the only
    difference vs the control is the target's compliance."""
    return {(r["task_id"], r["instruction_id"])
            for r in (json.loads(l) for l in open(FILES["control"]))}


def load_rows_mixed(which: str, *, plain_frac: float = 0.0, plain_n: int = 0, seed: int = 0,
                    drop_meta_null: bool = True, match_control: bool = False) -> list[dict]:
    """Load the primary SFT set (compliant/control) and optionally MIX IN a deterministic
    number of `plain` (no-instruction natural-trace) examples.

    The plain count is `plain_n` if given, else round(plain_frac * len(primary)). Using an ABSOLUTE
    plain_n lets the compliant and control arms share the IDENTICAL plain set even though their
    primary sizes differ slightly (matched-prompt control). The plain pool is sorted by example_id
    then shuffled with `seed` and the first N taken, so mixes are NESTED + reproducible.
    """
    rows = load_rows(which, drop_meta_null=drop_meta_null)
    if match_control and which == "compliant":
        pairs = control_pairs()
        before = len(rows)
        rows = [r for r in rows if (r["task_id"], r["instruction_id"]) in pairs]
        print(f"[ft_data] match_control: restricted compliant {before} -> {len(rows)} rows "
              f"(the {len(pairs)} prompts shared with the control set)")
    n_plain = plain_n if plain_n else (round(plain_frac * len(rows)) if plain_frac else 0)
    if n_plain > 0:
        plain = load_rows("plain", drop_meta_null=False)
        plain_sorted = sorted(plain, key=lambda r: r["example_id"])
        random.Random(seed).shuffle(plain_sorted)
        if n_plain > len(plain_sorted):
            print(f"[ft_data] WARNING plain_frac={plain_frac} wants {n_plain} plain but only "
                  f"{len(plain_sorted)} available; using all of them "
                  f"(actual frac {len(plain_sorted)/len(rows):.3f})")
        added = plain_sorted[:n_plain]
        print(f"[ft_data] mixing {len(added)} plain rows into {len(rows)} '{which}' rows "
              f"(plain_frac={plain_frac}, actual {len(added)/len(rows):.3f})")
        rows = rows + added
    return rows


def render_full_tokens(row: dict) -> tuple[list[int], list[int]]:
    """(prompt_tokens, completion_tokens) for one row via the locked rendering convention."""
    return H.render_training_example(row["prompt_user_content"], row["target_analysis"],
                                     row["target_final"])


def build_datum(row: dict, max_length: int):
    """Build a tinker.Datum (lazy import of tinker) with weights=0 on prompt, 1 on completion."""
    import torch
    import tinker
    from tinker_cookbook.supervised.common import datum_from_model_input_weights

    prompt_toks, comp_toks = render_full_tokens(row)
    full = prompt_toks + comp_toks
    weights = torch.tensor([0.0] * len(prompt_toks) + [1.0] * len(comp_toks), dtype=torch.float32)
    mi = tinker.ModelInput.from_ints(full)
    datum = datum_from_model_input_weights(mi, weights, max_length=max_length, reduction="mean")
    return datum, len(prompt_toks), len(comp_toks)


def build_datums(rows: list[dict], max_length: int):
    """Return (datums, stats). Skips rows whose full sequence exceeds max_length (logged)."""
    datums, kept, skipped = [], [], 0
    n_prompt_tok = n_comp_tok = 0
    for r in rows:
        prompt_toks, comp_toks = render_full_tokens(r)
        if len(prompt_toks) + len(comp_toks) > max_length:
            skipped += 1
            continue
        d, npr, nco = build_datum(r, max_length)
        datums.append(d)
        kept.append(r)
        n_prompt_tok += npr
        n_comp_tok += nco
    stats = {"n_examples": len(datums), "n_skipped_too_long": skipped,
             "total_prompt_tokens": n_prompt_tok, "total_completion_tokens": n_comp_tok}
    return datums, kept, stats


def data_hash(rows: list[dict]) -> str:
    """Deterministic content hash of the training set (example_ids + prompt/target shas)."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x["example_id"]):
        h.update(r["example_id"].encode())
        h.update(r["prompt_token_sha256"].encode())
        h.update(hashlib.sha256(r["target_analysis"].encode()).hexdigest().encode())
        h.update(hashlib.sha256(r["target_final"].encode()).hexdigest().encode())
    return h.hexdigest()[:16]
