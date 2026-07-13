"""Orchestrate the CPU-merge of a trained Tinker LoRA adapter into an MXFP4 HF checkpoint on the
Modal volume. Reads results/ft_train_<tag>.json for the adapter dir +
content hash, uploads the adapter to the volume, merges (cached on adapter_hash), and records the
merged model path in results/ft_merge_<tag>.json. The merged model loads into the HF eval/steering
harness (gpt_oss_infer.py) via --ft-tag.

Usage: python run_ft_merge.py --tag c32 [--assert-cached]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpt_oss_infer as G  # noqa: E402  (for _cache_and_cost)
from ft_merge_modal import (merge_app, merge_image, merge_to_volume,  # noqa: E402,F401
                            merge_to_volume_bf16, hf_cache_vol)

RESULTS = Path("results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="c32")
    p.add_argument("--bf16", action="store_true",
                   help="UNQUANTIZED merge: serve plain bf16 (no MXFP4 round-trip on the LoRA delta)")
    p.add_argument("--force-remerge", action="store_true",
                   help="ignore cached merge_info + clobber any partial dir, re-merge cleanly")
    p.add_argument("--assert-cached", action="store_true")
    args = p.parse_args()

    train_rec = json.loads((RESULTS / f"ft_train_{args.tag}.json").read_text())
    adapter_dir = train_rec["adapter_dir"]
    adapter_hash = train_rec["adapter_hash"]
    # batch_upload paths are VOLUME-ROOT-relative (NOT /cache-prefixed); the volume mounts at /cache.
    adapter_vol_relpath = f"ft_adapters/{args.tag}"
    adapter_vol_path = f"/cache/ft_adapters/{args.tag}"
    kind = "bf16" if args.bf16 else "mxfp4"
    suffix = "_bf16" if args.bf16 else ""
    merged_path = f"/cache/merged_ft_{args.tag}{suffix}"
    out_json = RESULTS / f"ft_merge_{args.tag}{suffix}.json"

    cache, tracker = G._cache_and_cost()
    # bf16 merge cache versioned separately (v1 produced a broken save with $-mangled expert keys).
    merge_key = {"fn": "ft_merge_to_volume", "base": "openai/gpt-oss-20b", "kind": kind,
                 "adapter_hash": adapter_hash, "merged_path": merged_path,
                 "this_call_uuid": "ft-merge-v1",
                 **({"bf16_merge_version": 2} if args.bf16 else {})}

    cached = None if args.force_remerge else cache.get(merge_key)
    if cached is not None:
        merge_info = cached
        print(f"[cache] merge already done ({kind}): {merge_info.get('merged_files')}")
    else:
        if args.assert_cached:
            raise RuntimeError("merge not cached")
        # Upload adapter to the volume (922MB; too big to bake into the image).
        if not Path(adapter_dir).exists():
            raise FileNotFoundError(f"adapter dir {adapter_dir} missing; re-run run_ft_train.py")
        print(f"Uploading adapter {adapter_dir} -> volume {adapter_vol_path} ...")
        with hf_cache_vol.batch_upload(force=True) as batch:
            batch.put_directory(adapter_dir, adapter_vol_relpath)
        mem = 262144 if args.bf16 else 131072
        print(f"Upload done; merging on CPU ({kind})...")
        with modal.enable_output(), merge_app.run():
            with tracker.track_modal(cpu=8.0, memory_mib=mem, is_sandbox=False) as t:
                if args.bf16:
                    merge_info = merge_to_volume_bf16.remote(adapter_vol_path, merged_path)
                else:
                    merge_info = merge_to_volume.remote(adapter_vol_path, merged_path,
                                                        clobber=args.force_remerge)
                t.elapsed = merge_info.get("merge_seconds") or None
        # don't poison the cache with a partial/failed merge
        if merge_info.get("merged_files") and not args.force_remerge:
            cache.get_or_set(merge_key, merge_info)
        elif args.force_remerge:
            # overwrite is not allowed by FileCache; the corrected info is on the volume regardless.
            pass

    rec = {"tag": args.tag, "kind": kind, "adapter_hash": adapter_hash, "merged_path": merged_path,
           "merged_files": merge_info.get("merged_files"),
           "merge_seconds": merge_info.get("merge_seconds"),
           "sampler_tinker_path": train_rec.get("sampler_tinker_path")}
    out_json.write_text(json.dumps(rec, indent=2))
    print(f"\nMerged model on volume ({kind}): {merged_path}")
    print(f"files: {merge_info.get('merged_files')}")
    print(f"Wrote {out_json}")
    print(f"Modal cost this run: ${tracker.run_cost:.4f}")


if __name__ == "__main__":
    main()
