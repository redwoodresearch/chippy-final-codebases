"""Robust, reusable answer extraction + accuracy scoring on the Harmony ``final`` channel.

Reference module (for understanding the accuracy metric used in the evals) — it is NOT on the
figure-reproduction path. Numeric/math scoring needs the optional ``math_verify`` package
(``pip install math_verify``); multiple-choice scoring needs no extra dependency. (The research
run also had an LLM math-equivalence judge fallback; it is omitted here as it depends on the
project's Anthropic wrapper.)

Accuracy is scored on the
model's **final** answer (the parsed Harmony ``final`` channel) — never the whole output and
never the ``analysis`` channel. Three answer types are handled:

  * ``mcq_letter`` — multiple-choice. Hardened letter extraction (prefer ``\\boxed{}`` /
    "answer is X" / "(X)" / a final standalone letter); the bare ``\\b[A-J]\\b`` last resort is
    deliberately avoided because it grabs stray capitals (false positives). Letter matching is
    **case-insensitive** (then upper-cased): later instruction-perturbed/FT/steered models can emit a
    *lowercase* final letter (e.g. ``a`` under an "all lowercase reasoning" instruction), which a
    case-sensitive extractor would miss — a measurement artifact, not a real accuracy drop
    (validated by inspection; base uppercase finals are unaffected).
  * ``number`` — a plain numeric answer (GSM8K, ReasonIF numeric). Boxed-first extraction +
    tolerant numeric / ``math_verify`` comparison.
  * ``math_expr`` — free-response competition-math answers (MATH). **No naive last-number
    extraction.** Extract the model's ``\\boxed{}`` (else the final text) and check equivalence
    with the LaTeX-aware ``math_verify`` library (fractions, ``\\frac``, ``\\sqrt``, ``\\text``,
    ``\\left``/``\\right``, ``%``, intervals, tuples). For programmatic *not-equal* cases an
    optional **cached, cost-tracked LLM judge** (``math_judge_equiv``) rescues false negatives.

Design notes:
  * ``math_verify.parse`` only anchors correctly when the expression is wrapped in a LaTeX
    delimiter, so we always wrap gold/pred in ``\\boxed{...}`` before parsing (verified: bare
    ``\\left(3,\\frac{\\pi}{2}\\right)`` mis-parses, the wrapped form is correct).
  * The programmatic checker is the **primary** accuracy metric. The LLM judge is a clearly
    secondary, conservative rescue for disputed MATH cases (reported separately, with agreement).
  * Everything degrades gracefully: empty / malformed / missing ``final`` → ``None`` or incorrect,
    never a crash.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Answer-format instructions (STABLE project-wide convention — documented choice).
# Appended to every task prompt. NO reasoning/CoT-format instruction here (those come from the instruction suite).
# math/number -> \boxed{} (in-distribution for math, robust extraction).
# mcq          -> "just the letter" (forcing \boxed{C} on MCQ is out-of-distribution and caused
#                 format failures in LLM review; the extractor still accepts a boxed letter).
# ---------------------------------------------------------------------------
ANSWER_FORMAT_INSTRUCTIONS = {
    "number": "Put your final answer inside \\boxed{} (for example, \\boxed{42}).",
    "math_expr": "Put your final answer inside \\boxed{} (for example, \\boxed{42}).",
    "mcq_letter": "Give your final answer as just the letter of the correct option "
                  "(for example, C).",
}


def build_answer_instruction(answer_type: str) -> str:
    """The stable answer-format instruction for an answer type (raises on unknown type)."""
    if answer_type not in ANSWER_FORMAT_INSTRUCTIONS:
        raise ValueError(f"unknown answer_type {answer_type!r}")
    return ANSWER_FORMAT_INSTRUCTIONS[answer_type]


# ---------------------------------------------------------------------------
# \boxed{} extraction (balanced braces; handles nesting like \boxed{\frac{1}{2}}).
# ---------------------------------------------------------------------------
def last_boxed_content(text: str) -> Optional[str]:
    """Return the content of the LAST ``\\boxed{...}`` / ``\\fbox{...}`` (brace-balanced), else None."""
    if not text:
        return None
    best = None
    for marker in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(marker, start)
            if idx < 0:
                break
            # find the first "{" after the marker
            brace = text.find("{", idx)
            if brace < 0:
                start = idx + len(marker)
                continue
            depth = 0
            i = brace
            content_start = brace + 1
            end = None
            while i < len(text):
                c = text[i]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
                i += 1
            if end is not None:
                best = text[content_start:end]
                start = end + 1
            else:
                start = idx + len(marker)
    return best


# ---------------------------------------------------------------------------
# MCQ letter extraction (hardened against stray capitals).
# ---------------------------------------------------------------------------
def extract_mcq_letter(final_text: str, n_options: int = 10) -> Optional[str]:
    """Extract the chosen MCQ letter from the final channel.

    Priority (each returns the LAST match): boxed letter -> explicit "answer is X" ->
    standalone-line letter -> "(X)" -> "X." / "X)" near end. The bare ``\\b[A-J]\\b`` fallback is
    intentionally NOT used (it grabs stray capitals like the article 'A'); we return None instead,
    which scores as incorrect rather than a false positive.
    """
    if not final_text or not final_text.strip():
        return None
    n_options = max(2, min(int(n_options or 10), 26))
    letters = "".join(chr(ord("A") + k) for k in range(n_options))
    LC = f"[{letters}]"

    def _letter_from(s: str) -> Optional[str]:
        m = re.findall(rf"(?<![A-Za-z])({LC})(?![A-Za-z])", s, flags=re.IGNORECASE)
        return m[0].upper() if m else None

    # 1) boxed content -> letter inside (handles \boxed{C}, \boxed{\text{C}}, \boxed{C. ...}).
    boxed = last_boxed_content(final_text)
    if boxed is not None:
        lt = _letter_from(boxed)
        if lt:
            return lt

    # 2) explicit "answer (is|:|=) X" possibly parenthesised / bracketed.
    pats_all = [
        rf"(?:correct\s+)?(?:final\s+)?answer\s*(?:is|:|=|\s)\s*\**\(?\[?\s*({LC})\s*\)?\]?\b",
        rf"option\s*\(?\s*({LC})\s*\)?\b",
    ]
    for pat in pats_all:
        m = re.findall(pat, final_text, flags=re.IGNORECASE)
        if m:
            return m[-1].upper()

    # 3) a line that is essentially just the letter (optionally wrapped/punctuated).
    standalone = None
    for line in final_text.splitlines():
        s = line.strip().strip("*").strip()
        m = re.fullmatch(rf"\(?\[?\s*({LC})\s*\)?\]?[.:)]?", s, flags=re.IGNORECASE)
        if m:
            standalone = m.group(1).upper()
    if standalone:
        return standalone

    # 4) "(X)" anywhere (last).
    m = re.findall(rf"\(\s*({LC})\s*\)", final_text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()

    # 5) "X." / "X)" / "X:" token (last) — still requires a delimiter, not a bare letter.
    m = re.findall(rf"(?<![A-Za-z])({LC})\s*[.):]", final_text, flags=re.IGNORECASE)
    if m:
        return m[-1].upper()
    return None


# ---------------------------------------------------------------------------
# Numeric extraction.
# ---------------------------------------------------------------------------
def _to_float(s: str) -> Optional[float]:
    s = s.strip().replace("$", "").replace(",", "").replace("\\%", "").replace("%", "")
    s = s.replace("\\!", "").replace("\\,", "").strip()
    m = re.fullmatch(r"-?\d+\s*/\s*-?\d+", s)
    if m:
        try:
            a, b = s.split("/")
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_number(final_text: str) -> Optional[float]:
    """Extract the model's numeric answer: prefer ``\\boxed{...}``; else the last number in text."""
    if not final_text:
        return None
    boxed = last_boxed_content(final_text)
    if boxed is not None:
        v = _to_float(boxed)
        if v is not None:
            return v
        nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", boxed)
        if nums:
            v = _to_float(nums[-1])
            if v is not None:
                return v
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", final_text)
    if not nums:
        return None
    return _to_float(nums[-1])


# ---------------------------------------------------------------------------
# math_verify equivalence (LaTeX-aware). gold/pred wrapped in \boxed for correct anchoring.
# ---------------------------------------------------------------------------
_MV = None


def _math_verify():
    global _MV
    if _MV is None:
        from math_verify import parse, verify  # lazy import
        _MV = (parse, verify)
    return _MV


def _mv_parse_boxed(expr: str):
    parse, _ = _math_verify()
    return parse("\\boxed{" + expr + "}")


def math_equiv(gold: str, pred_expr: str) -> Optional[bool]:
    """Programmatic mathematical equivalence of two answer strings via ``math_verify``.

    ``gold`` is the raw gold answer; ``pred_expr`` is the model's extracted answer expression
    (boxed content, or the raw final text). Returns True/False, or None on a hard parse error.
    """
    if pred_expr is None or pred_expr.strip() == "":
        return False
    parse, verify = _math_verify()
    try:
        g = _mv_parse_boxed(gold)
        # Prefer the model's boxed content; else let parse extract from the full text.
        boxed = last_boxed_content(pred_expr)
        if boxed is not None:
            p = _mv_parse_boxed(boxed)
        else:
            p = parse(pred_expr)
            if not p:  # nothing extracted -> try the wrapped form
                p = _mv_parse_boxed(pred_expr.strip().splitlines()[-1] if pred_expr.strip() else "")
        if not g or not p:
            return False
        return bool(verify(g, p))
    except Exception:  # noqa: BLE001 - any checker failure -> not-confirmed-equal
        return None


# ---------------------------------------------------------------------------
# Top-level scoring.
# ---------------------------------------------------------------------------
def score_answer(final_text: str, answer: str, answer_type: str,
                 n_options: Optional[int] = None) -> dict:
    """Score one answer. Returns a rich dict (programmatic only; LLM judge applied separately).

    Keys: ``correct`` (Optional[bool]), ``extracted`` (str|None), ``method``, ``has_box`` (bool),
    ``empty_final`` (bool).
    """
    out = {"correct": None, "extracted": None, "method": answer_type,
           "has_box": False, "empty_final": False}
    if final_text is None or final_text.strip() == "":
        out["empty_final"] = True
        out["correct"] = None  # caller decides empty-final policy (default: incorrect)
        return out
    out["has_box"] = last_boxed_content(final_text) is not None

    if answer_type == "mcq_letter":
        pred = extract_mcq_letter(final_text, n_options or 10)
        out["extracted"] = pred
        if pred is None:
            out["correct"] = False
        else:
            out["correct"] = (pred == str(answer).strip().upper())
        return out

    if answer_type == "number":
        pred = extract_number(final_text)
        out["extracted"] = None if pred is None else (
            str(int(pred)) if float(pred).is_integer() else str(pred))
        gold = _to_float(str(answer))
        if pred is not None and gold is not None:
            out["correct"] = abs(pred - gold) < 1e-4 or (
                gold != 0 and abs(pred - gold) / abs(gold) < 1e-6)
            if not out["correct"]:
                # fall back to symbolic check (handles e.g. fraction/format mismatches)
                eq = math_equiv(str(answer), final_text)
                out["correct"] = bool(eq)
                if eq:
                    out["method"] = "number+math_verify"
        else:
            eq = math_equiv(str(answer), final_text)
            out["correct"] = bool(eq)
            out["method"] = "number+math_verify"
        return out

    if answer_type == "math_expr":
        boxed = last_boxed_content(final_text)
        out["extracted"] = boxed if boxed is not None else final_text.strip()[-120:]
        eq = math_equiv(str(answer), final_text)
        out["correct"] = bool(eq)
        out["mv_result"] = eq  # None if checker errored
        out["method"] = "math_verify"
        return out

    raise ValueError(f"unknown answer_type {answer_type!r}")


def score_accuracy(final_text: str, answer: str, answer_type: str,
                   n_options: Optional[int] = None) -> Optional[bool]:
    """Thin backward-compatible wrapper returning Optional[bool] (None = empty final)."""
    return score_answer(final_text, answer, answer_type, n_options)["correct"]
