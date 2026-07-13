"""Harmony rendering + channel-parsing utilities for gpt-oss-20b.

This is the single most important primitive in the project: all CoT-compliance scoring
happens on the parsed ``analysis`` channel. We use the official ``openai_harmony`` library
as the *canonical* renderer AND parser so that the prompt we send to the model and the
channels we score are guaranteed to be consistent.

Key facts about the Harmony format (verified against the library):
  * Special tokens: <|start|>=200006, <|end|>=200007, <|message|>=200008,
    <|channel|>=200005, <|return|>=200002, <|call|>=200012, <|constrain|>=200003.
  * The completion-prompt ends with ``<|start|>assistant``; the model then emits one or
    more messages of the form ``<|channel|>{analysis|commentary|final}<|message|>{text}<|end|>``
    (the final message ends with ``<|return|>`` instead of ``<|end|>``).
  * There are THREE channels: ``analysis`` (CoT/reasoning), ``commentary`` (tool/preamble),
    ``final`` (the answer). The model can emit MULTIPLE ``analysis`` messages.
  * Assistant-turn stop tokens are [<|call|>=200012, <|return|>=200002].

Malformed handling (verified empirically):
  * ``parse_messages_from_completion_tokens`` raises ``HarmonyError`` when channel markers
    are broken (e.g. a missing ``<|message|>`` marker, or no markers at all -- the kind of
    corruption steering can cause). We catch this and flag the output as ``malformed``.
  * A *truncated* generation (hit max_new_tokens mid-message) parses fine but yields no
    ``final`` channel; we track that separately via ``has_final`` / ``truncated`` rather than
    calling it malformed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    HarmonyError,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    load_harmony_encoding,
)

# Special-token ids (constants; verified above).
TOK_START = 200006
TOK_END = 200007
TOK_MESSAGE = 200008
TOK_CHANNEL = 200005
TOK_RETURN = 200002
TOK_CALL = 200012
TOK_CONSTRAIN = 200003
# Assistant-turn stop tokens used as eos for generation.
ASSISTANT_STOP_TOKENS = [TOK_CALL, TOK_RETURN]

_REASONING_MAP = {
    "low": ReasoningEffort.LOW,
    "medium": ReasoningEffort.MEDIUM,
    "high": ReasoningEffort.HIGH,
}

# Cache the encoding (loading is cheap but not free).
_ENC = None


def get_encoding():
    global _ENC
    if _ENC is None:
        _ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _ENC


def render_prompt_tokens(
    user_content: str,
    *,
    reasoning_effort: str = "medium",
    developer_instructions: Optional[str] = None,
    system_extra: Optional[str] = None,
    include_date: bool = False,
) -> list[int]:
    """Render a single-user-turn conversation to Harmony completion-prompt token ids.

    The reasoning instruction (for the thin-slice CoT-control baseline) is expected to be
    embedded in ``user_content`` and/or ``developer_instructions`` by the caller.

    Args:
        user_content: the full user message (task + any in-prompt reasoning instruction).
        reasoning_effort: "low" | "medium" | "high" (set in the Harmony system message).
        developer_instructions: optional developer-message instructions (None => no dev msg).
        system_extra: optional extra text appended to the system message content.
        include_date: if True, set the conversation start date (matches HF chat template,
            which always injects "Current date"). Off by default for determinism.
    """
    enc = get_encoding()
    if reasoning_effort not in _REASONING_MAP:
        raise ValueError(f"reasoning_effort must be one of {list(_REASONING_MAP)}")
    sys_c = SystemContent.new().with_reasoning_effort(_REASONING_MAP[reasoning_effort])
    if include_date:
        sys_c = sys_c.with_conversation_start_date(date.today().isoformat())
    msgs = [Message.from_role_and_content(Role.SYSTEM, sys_c)]
    if developer_instructions is not None:
        dev_c = DeveloperContent.new().with_instructions(developer_instructions)
        msgs.append(Message.from_role_and_content(Role.DEVELOPER, dev_c))
    msgs.append(Message.from_role_and_content(Role.USER, user_content))
    convo = Conversation.from_messages(msgs)
    return enc.render_conversation_for_completion(convo, Role.ASSISTANT)


def encode_text(text: str) -> list[int]:
    """Encode plain text to token ids, treating any special-token-looking substrings (e.g. a
    literal ``<|end|>`` that happens to appear in a reasoning trace) as ordinary text rather than
    special tokens (``disallowed_special=()``). Round-trips: ``decode_tokens(encode_text(t)) == t``.
    """
    return get_encoding().encode(text, allowed_special=set(), disallowed_special=())


def render_assistant_completion(analysis: str, final: str) -> list[int]:
    """Render an assistant turn (an ``analysis`` CoT message followed by a ``final`` answer message)
    to the **completion** token ids the model would emit *after* a completion prompt ending in
    ``<|start|>assistant`` (see ``render_prompt_tokens``). The structure mirrors a real gpt-oss
    generation exactly::

        <|channel|>analysis<|message|>{analysis}<|end|>\
        <|start|>assistant<|channel|>final<|message|>{final}<|return|>

    ``parse_channels(render_assistant_completion(a, f))`` recovers ``analysis==a`` / ``final==f``
    with ``malformed=False`` / ``truncated=False`` (verified). This is the canonical way to build an
    SFT training target's assistant turn (the Harmony *training* renderer drops prior-turn
    ``analysis`` reasoning, so it CANNOT be used to materialize a target whose CoT must be kept).
    """
    enc = get_encoding()
    toks: list[int] = []
    toks.append(TOK_CHANNEL)
    toks += enc.encode("analysis", allowed_special=set())
    toks.append(TOK_MESSAGE)
    toks += encode_text(analysis)
    toks.append(TOK_END)
    toks.append(TOK_START)
    toks += enc.encode("assistant", allowed_special=set())
    toks.append(TOK_CHANNEL)
    toks += enc.encode("final", allowed_special=set())
    toks.append(TOK_MESSAGE)
    toks += encode_text(final)
    toks.append(TOK_RETURN)
    return toks


def render_training_example(
    user_content: str,
    analysis: str,
    final: str,
    *,
    reasoning_effort: str = "medium",
    developer_instructions: Optional[str] = None,
    include_date: bool = False,
) -> tuple[list[int], list[int]]:
    """Render a full SFT example to ``(prompt_token_ids, completion_token_ids)``.

    The prompt is rendered by ``render_prompt_tokens`` (system + optional developer + user, ending
    in ``<|start|>assistant``); the completion is the assistant turn (``analysis`` + ``final``) from
    ``render_assistant_completion``. ``prompt + completion`` is the full token sequence the
    FT pipeline would train on (mask the prompt, learn the completion). ``include_date`` MUST match
    the eval-time rendering convention (we use ``include_date=False`` project-wide; see write-up).
    """
    prompt = render_prompt_tokens(
        user_content,
        reasoning_effort=reasoning_effort,
        developer_instructions=developer_instructions,
        include_date=include_date,
    )
    completion = render_assistant_completion(analysis, final)
    return prompt, completion


def decode_tokens(token_ids: list[int]) -> str:
    """Decode token ids to text *including* special-token markers (for inspection).

    Robust to invalid UTF-8 (e.g. a generation truncated at max_new_tokens mid-multibyte
    character): falls back to decoding the longest valid token prefix rather than raising.
    """
    enc = get_encoding()
    ids = list(token_ids)
    try:
        return enc.decode_utf8(ids)
    except Exception:  # noqa: BLE001 - invalid utf-8 -> decode the longest valid prefix
        lo, hi, best = 0, len(ids), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                best = enc.decode_utf8(ids[:mid])
                lo = mid + 1
            except Exception:  # noqa: BLE001
                hi = mid - 1
        return best


@dataclass
class ParsedCompletion:
    """Structured result of parsing an assistant completion's channels."""

    analysis: str  # concatenated analysis text (the CoT we score)
    commentary: str  # concatenated commentary text (tracked, never merged into analysis)
    final: str  # concatenated final text (the answer)
    analysis_segments: list[str] = field(default_factory=list)
    commentary_segments: list[str] = field(default_factory=list)
    final_segments: list[str] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # raw per-message dicts
    malformed: bool = False
    malformed_reason: Optional[str] = None
    has_analysis: bool = False
    has_final: bool = False
    has_commentary: bool = False
    num_analysis_segments: int = 0
    unexpected_channels: list[str] = field(default_factory=list)
    truncated: bool = False  # generation stopped without a turn-ending stop token
    raw_text: str = ""

    def to_dict(self) -> dict:
        return {
            "analysis": self.analysis,
            "commentary": self.commentary,
            "final": self.final,
            "analysis_segments": self.analysis_segments,
            "commentary_segments": self.commentary_segments,
            "final_segments": self.final_segments,
            "messages": self.messages,
            "malformed": self.malformed,
            "malformed_reason": self.malformed_reason,
            "has_analysis": self.has_analysis,
            "has_final": self.has_final,
            "has_commentary": self.has_commentary,
            "num_analysis_segments": self.num_analysis_segments,
            "unexpected_channels": self.unexpected_channels,
            "truncated": self.truncated,
            "raw_text": self.raw_text,
        }


def _extract_text(content) -> str:
    """Pull plain text out of a Harmony message ``content`` field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, dict):
            parts.append(item.get("text", ""))
        else:
            parts.append(str(item))
    return "".join(parts)


def parse_channels(completion_token_ids: list[int]) -> ParsedCompletion:
    """Parse the Harmony channels from an assistant completion (token ids).

    ``completion_token_ids`` must be exactly the tokens the model generated *after* the
    completion prompt (which ends with ``<|start|>assistant``), optionally including the
    trailing stop token (<|return|> / <|call|>).
    """
    enc = get_encoding()
    token_ids = list(completion_token_ids)
    raw_text = decode_tokens(token_ids)  # robust to invalid utf-8 (see decode_tokens)
    truncated = len(token_ids) == 0 or token_ids[-1] not in (TOK_RETURN, TOK_CALL)

    try:
        msgs = enc.parse_messages_from_completion_tokens(token_ids, Role.ASSISTANT)
    except HarmonyError as e:
        # Broken channel markers -> malformed (this is what steering corruption looks like).
        return ParsedCompletion(
            analysis="",
            commentary="",
            final="",
            malformed=True,
            malformed_reason=f"HarmonyError: {e}",
            truncated=truncated,
            raw_text=raw_text,
        )
    except Exception as e:  # noqa: BLE001 - be conservative; any parse failure => malformed
        return ParsedCompletion(
            analysis="",
            commentary="",
            final="",
            malformed=True,
            malformed_reason=f"{type(e).__name__}: {e}",
            truncated=truncated,
            raw_text=raw_text,
        )

    analysis_segments: list[str] = []
    commentary_segments: list[str] = []
    final_segments: list[str] = []
    unexpected: list[str] = []
    msg_dicts: list[dict] = []

    for m in msgs:
        d = m.to_dict()
        channel = d.get("channel")
        text = _extract_text(d.get("content"))
        msg_dicts.append({"role": str(d.get("role")), "channel": channel, "text": text})
        if channel == "analysis":
            analysis_segments.append(text)
        elif channel == "commentary":
            commentary_segments.append(text)
        elif channel == "final":
            final_segments.append(text)
        else:
            # No channel (or an unexpected one). Track explicitly; never silently drop.
            unexpected.append(str(channel))

    return ParsedCompletion(
        analysis="\n".join(analysis_segments),
        commentary="\n".join(commentary_segments),
        final="\n".join(final_segments),
        analysis_segments=analysis_segments,
        commentary_segments=commentary_segments,
        final_segments=final_segments,
        messages=msg_dicts,
        malformed=False,
        malformed_reason=None,
        has_analysis=len(analysis_segments) > 0,
        has_final=len(final_segments) > 0,
        has_commentary=len(commentary_segments) > 0,
        num_analysis_segments=len(analysis_segments),
        unexpected_channels=unexpected,
        truncated=truncated,
        raw_text=raw_text,
    )
