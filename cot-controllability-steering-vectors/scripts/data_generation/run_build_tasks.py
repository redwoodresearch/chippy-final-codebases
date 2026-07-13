"""Build the unified, deterministic task pool for the CoT-controllability project.

Loads a diverse set of reasoning sources, normalizes them to one schema, deduplicates by
question stem (within and across sources), assigns deterministic seeded train/val/held-out
splits stratified by source (+ level/category/subsource), and writes ``data/tasks_all.jsonl``.

Sources (see writeups/desiderata_tasks.md for rationale + gold-answer provenance):
  gsm8k         number     openai/gsm8k (test)                 gold = text after '####'
  math          math_expr  EleutherAI/hendrycks_math (test)    gold = last \\boxed{} in solution
  mmlu_pro      mcq_letter TIGER-Lab/MMLU-Pro (test)           gold = answer letter
  openbookqa    mcq_letter allenai/openbookqa (main test)      gold = answerKey
  arc_challenge mcq_letter allenai/ai2_arc ARC-Challenge (test) gold = answerKey (relabel->A..)
  reasonif      number/mcq ykwon-hf/reasonIF (format stripped) gold = answer (per subsource)

Run:  python run_build_tasks.py        (deterministic; safe to re-run, overwrites the pool)
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault(
    "HF_TOKEN", open(os.path.expanduser("~/.cache/huggingface/token")).read().strip())
from datasets import load_dataset  # noqa: E402

import answer_scoring as A  # noqa: E402  (for last_boxed_content reuse)

SEED = 0
OUT = Path("data/tasks_all.jsonl")

# Per-source caps (before dedup). None = take all available.
CAPS = {
    "gsm8k": 520,
    "math": 520,
    "mmlu_pro": 520,
    "arc_challenge": 520,
    "openbookqa": None,   # 500 test items, take all
    "reasonif": None,     # 300 items, take all
}
SPLIT_FRACS = {"train": 0.55, "val": 0.15, "heldout": 0.30}
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]


def rng(tag: str) -> random.Random:
    """A fresh deterministic RNG per stage (so adding a source doesn't reshuffle others)."""
    return random.Random(f"{SEED}:{tag}")


def render_mcq(stem: str, option_texts: list[str]) -> tuple[str, int]:
    letters = [chr(ord("A") + k) for k in range(len(option_texts))]
    opt_str = "\n".join(f"{l}. {o}" for l, o in zip(letters, option_texts))
    return f"{stem.strip()}\n\nOptions:\n{opt_str}", len(option_texts)


def stem_key(question: str, mcq_inline: bool = False) -> str:
    """Normalized question-stem signature for dedup (alphanumeric, lowercase).

    For MCQ we key on the stem BEFORE the options so the same underlying question rendered
    differently (e.g. ReasonIF's inline options vs our rendered options) dedups correctly.
    """
    q = question
    # strip a known ReasonIF MCQ wrapper prefix
    q = re.sub(r"^read the following multiple-choice question[^\n]*\n", "", q,
               flags=re.IGNORECASE)
    # cut at the options block
    q = re.split(r"\n\s*options:\s*\n", q, flags=re.IGNORECASE)[0]
    if mcq_inline:
        # ReasonIF inline options begin at the first "\nA." / "\nA)" line
        q = re.split(r"\n\s*[A-J][.)]\s", q)[0]
    return re.sub(r"[^a-z0-9]", "", q.lower())


# ---------------------------------------------------------------------------
# Loaders (each returns a list of partial task dicts; ids/splits added later).
# ---------------------------------------------------------------------------
def load_gsm8k() -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i, ex in enumerate(ds):
        gold_raw = ex["answer"]
        ans = gold_raw.split("####")[-1].strip().replace(",", "")
        out.append({"source": "gsm8k", "uid": f"gsm8k_{i}", "question": ex["question"].strip(),
                    "answer": ans, "answer_type": "number", "n_options": None,
                    "category": "grade_school_math", "level": None, "subsource": None,
                    "gold_raw": gold_raw.split("####")[-1].strip(),
                    "_stemkey": stem_key(ex["question"])})
    return out


def load_math() -> list[dict]:
    # gather all test items with a level, then level-balanced sample to the cap.
    by_level = defaultdict(list)
    for cfg in MATH_CONFIGS:
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="test")
        for j, ex in enumerate(ds):
            boxed = A.last_boxed_content(ex["solution"])
            if boxed is None or boxed.strip() == "":
                continue
            level = ex.get("level", "Level ?")
            by_level[level].append({
                "source": "math", "uid": f"math_{cfg}_{j}", "question": ex["problem"].strip(),
                "answer": boxed.strip(), "answer_type": "math_expr", "n_options": None,
                "category": ex.get("type", cfg), "level": level, "subsource": None,
                "gold_raw": boxed.strip(), "_stemkey": stem_key(ex["problem"])})
    cap = CAPS["math"]
    levels = sorted([lv for lv in by_level if lv.startswith("Level") and lv != "Level ?"])
    per = cap // len(levels) if cap else None
    out = []
    for lv in levels:
        items = by_level[lv]
        rng(f"math:{lv}").shuffle(items)
        out.extend(items[:per] if per else items)
    return out


def load_mmlu_pro() -> list[dict]:
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    idxs = list(range(len(ds)))
    rng("mmlu_pro").shuffle(idxs)
    out = []
    for i in idxs:
        ex = ds[i]
        if not ex["options"] or not ex["answer"]:
            continue
        q, n = render_mcq(ex["question"], list(ex["options"]))
        out.append({"source": "mmlu_pro", "uid": f"mmlu_pro_{ex['question_id']}", "question": q,
                    "answer": ex["answer"].strip().upper(), "answer_type": "mcq_letter",
                    "n_options": n, "category": ex.get("category", ""), "level": None,
                    "subsource": None, "gold_raw": ex["answer"],
                    "_stemkey": stem_key(ex["question"])})
    return out


def load_openbookqa() -> list[dict]:
    ds = load_dataset("allenai/openbookqa", "main", split="test")
    out = []
    for ex in ds:
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        key = ex["answerKey"].strip()
        if key not in labels:
            continue
        gold_letter = chr(ord("A") + labels.index(key))
        q, n = render_mcq(ex["question_stem"], list(texts))
        out.append({"source": "openbookqa", "uid": f"openbookqa_{ex['id']}", "question": q,
                    "answer": gold_letter, "answer_type": "mcq_letter", "n_options": n,
                    "category": "elementary_science", "level": None, "subsource": None,
                    "gold_raw": key, "_stemkey": stem_key(ex["question_stem"])})
    return out


def load_arc_challenge() -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    idxs = list(range(len(ds)))
    rng("arc_challenge").shuffle(idxs)
    out = []
    for i in idxs:
        ex = ds[i]
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        key = ex["answerKey"].strip()
        if key not in labels:
            continue
        gold_letter = chr(ord("A") + labels.index(key))  # relabel numeric/letter labels -> A..
        q, n = render_mcq(ex["question"], list(texts))
        out.append({"source": "arc_challenge", "uid": f"arc_{ex['id']}", "question": q,
                    "answer": gold_letter, "answer_type": "mcq_letter", "n_options": n,
                    "category": "science_reasoning", "level": None, "subsource": None,
                    "gold_raw": key, "_stemkey": stem_key(ex["question"])})
    return out


def load_reasonif() -> list[dict]:
    """ReasonIF underlying questions with the bundled reasoning-format instructions STRIPPED.

    Numeric subsources (aime/amc/gsm8k) -> answer_type=number; MCQ subsources (arc/gpqa, whose
    options are already inline in the question) -> answer_type=mcq_letter.
    """
    ds = load_dataset("ykwon-hf/reasonIF", split="train")
    # keep one entry per distinct underlying question (constraints duplicate questions)
    seen = set()
    out = []
    for i, ex in enumerate(ds):
        q = ex["question"].strip()
        if q in seen:
            continue
        seen.add(q)
        src = ex["source"]
        ans = ex["answer"].strip()
        if src in ("arc", "gpqa"):
            atype = "mcq_letter"
            # count inline options A. .. to set n_options
            n = len(re.findall(r"(?m)^\s*[A-J][.)]\s", q)) or len(re.findall(r"(?<![A-Za-z])[A-J][.)]\s", q))
            n = max(2, n) if n else 4
            ans = ans.upper()
            mcq_inline = True
        else:
            atype = "number"
            ans = str(int(ans)) if re.fullmatch(r"-?\d+", ans) else ans  # '081'->'81'
            n = None
            mcq_inline = False
        out.append({"source": "reasonif", "uid": f"reasonif_{src}_{i}", "question": q,
                    "answer": ans, "answer_type": atype, "n_options": n,
                    "category": f"reasonif_{src}", "level": None, "subsource": src,
                    "gold_raw": ex["answer"], "_stemkey": stem_key(q, mcq_inline=mcq_inline)})
    return out


LOADERS = [load_gsm8k, load_math, load_mmlu_pro, load_openbookqa, load_arc_challenge,
           load_reasonif]


def assign_splits(tasks: list[dict]) -> None:
    """Deterministic, stratified (by source[:level/category/subsource]) split, in place."""
    strata = defaultdict(list)
    for t in tasks:
        if t["source"] == "math":
            s = f"math:{t['level']}"
        elif t["source"] == "mmlu_pro":
            s = f"mmlu_pro:{t['category']}"
        elif t["source"] == "reasonif":
            s = f"reasonif:{t['subsource']}"
        else:
            s = t["source"]
        strata[s].append(t)
    for s, items in strata.items():
        items_sorted = sorted(items, key=lambda t: t["task_id"])  # stable base order
        rng(f"split:{s}").shuffle(items_sorted)
        n = len(items_sorted)
        n_tr = round(n * SPLIT_FRACS["train"])
        n_va = round(n * SPLIT_FRACS["val"])
        for k, t in enumerate(items_sorted):
            t["split"] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "heldout")


def main():
    all_tasks = []
    for loader in LOADERS:
        src_tasks = loader()
        src = src_tasks[0]["source"] if src_tasks else loader.__name__
        cap = CAPS.get(src)
        if cap is not None and len(src_tasks) > cap:
            r = rng(f"cap:{src}")
            r.shuffle(src_tasks)
            src_tasks = src_tasks[:cap]
        print(f"[load] {src:14s}: {len(src_tasks)} items")
        all_tasks.extend(src_tasks)

    # Global dedup by question-stem signature (keep first occurrence in load order).
    seen = set()
    deduped = []
    n_dup = Counter()
    for t in all_tasks:
        k = t["_stemkey"]
        if k in seen or k == "":
            n_dup[t["source"]] += 1
            continue
        seen.add(k)
        deduped.append(t)
    if sum(n_dup.values()):
        print(f"[dedup] dropped {sum(n_dup.values())} duplicates by stem: {dict(n_dup)}")

    # Stable, source-prefixed task_id, then splits.
    for t in deduped:
        t["task_id"] = t.pop("uid")
    assign_splits(deduped)

    # Drop the internal stem key; order deterministically.
    deduped.sort(key=lambda t: (t["source"], t["task_id"]))
    for t in deduped:
        t.pop("_stemkey", None)

    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for t in deduped:
            f.write(json.dumps(t) + "\n")

    # ---- summary ----
    print(f"\nWrote {len(deduped)} tasks to {OUT}")
    by_src = Counter(t["source"] for t in deduped)
    by_type = Counter(t["answer_type"] for t in deduped)
    print("\nBy source:", dict(by_src))
    print("By answer_type:", dict(by_type))
    print("\nSplit sizes per source:")
    grid = defaultdict(lambda: Counter())
    for t in deduped:
        grid[t["source"]][t["split"]] += 1
    hdr = f"  {'source':14s} {'train':>7s} {'val':>6s} {'heldout':>8s} {'total':>7s}"
    print(hdr)
    for src in sorted(grid):
        c = grid[src]
        print(f"  {src:14s} {c['train']:7d} {c['val']:6d} {c['heldout']:8d} "
              f"{sum(c.values()):7d}")
    tot = Counter(t["split"] for t in deduped)
    print(f"  {'TOTAL':14s} {tot['train']:7d} {tot['val']:6d} {tot['heldout']:8d} "
          f"{len(deduped):7d}")
    # MATH level distribution
    math_levels = Counter(t["level"] for t in deduped if t["source"] == "math")
    print("\nMATH level distribution:", dict(sorted(math_levels.items())))
    rif = Counter(t["subsource"] for t in deduped if t["source"] == "reasonif")
    print("ReasonIF subsource distribution:", dict(rif))


if __name__ == "__main__":
    main()
