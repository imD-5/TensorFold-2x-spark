"""Top-level tool-call blocks of a reply: DeepSeek-V4.1 DSML calls blocks and <tool_call> blocks.

A call counts only when its block is complete and well formed at the top level. Text in an open block is not a call.
Text in a closed block, for example a nested <tool_call>, is argument text. An unsure block is not a call.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, NamedTuple

DSML = "\uff5cDSML\uff5c"
DSML_OPEN, DSML_CLOSE = f"<{DSML} calls>", f"</{DSML} calls>"
_INVOKE_RE = re.compile(rf'<{re.escape(DSML)} invoke name="([^"]+)">')
_PARAM_RE = re.compile(rf'<{re.escape(DSML)} parameter name="([^"]+)"(?: string="(true|false)")?>')
_SPACE_RE = re.compile(r"\s*")


def _pair(open_tag: str, close_tag: str, flags: int = 0) -> tuple[re.Pattern, re.Pattern]:
    return re.compile(re.escape(open_tag), flags), re.compile(re.escape(close_tag), flags)


# the opening and closing tags that a same-kind block nests by: the start of an invoke or parameter tag is enough
_CALLS_TAGS = _pair(DSML_OPEN, DSML_CLOSE)
_INVOKE_TAGS = _pair(f"<{DSML} invoke ", f"</{DSML} invoke>")
_PARAM_TAGS = _pair(f"<{DSML} parameter ", f"</{DSML} parameter>")

CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
# the exact tags, and the tags in any case (the call parser finds them in any case)
_TAG_TAGS = _pair(CALL_OPEN, CALL_CLOSE)
_TAG_TAGS_ANY = _pair(CALL_OPEN, CALL_CLOSE, re.IGNORECASE)


OPEN_TAG, OPEN_DSML, ILL_FORMED = "open_tool_call_tag", "open_dsml_block", "ill_formed_dsml_block"
_suppressed: dict[str, int] = {}
_suppressed_lock = threading.Lock()


def count_suppressed(reasons: set[str]) -> None:
    """Count one reply for each reason: a block that does not count kept a later tool call as text."""

    with _suppressed_lock:
        for reason in reasons:
            _suppressed[reason] = _suppressed.get(reason, 0) + 1


def suppressed_counts() -> dict[str, int]:
    """The replies counted so far, by reason (the /metrics family tool_calls_suppressed_total)."""

    with _suppressed_lock:
        return dict(_suppressed)


def reset_suppressed() -> None:
    with _suppressed_lock:
        _suppressed.clear()


class Block(NamedTuple):
    kind: str       # "tag" for a <tool_call> block, "dsml" for a DSML calls block
    start: int
    end: int
    inner: str      # the text between the opening and the closing tag


def _close_of(text: str, pos: int, tags: tuple[re.Pattern, re.Pattern]) -> re.Match | None:
    """The closing tag that ends the block whose opening tag ends at ``pos``, or None when the block is open.

    A block of the same kind in the text is skipped, so its closing tag does not end this one. A reply cut inside a
    nested block therefore leaves the outer block open."""

    open_re, close_re = tags
    depth = 1
    opened, closed = open_re.search(text, pos), close_re.search(text, pos)
    while closed is not None:
        if opened is not None and opened.start() < closed.start():
            depth += 1
            opened = open_re.search(text, opened.end())
            continue
        depth -= 1
        if depth == 0:
            return closed
        closed = close_re.search(text, closed.end())
    return None


def dsml_block_end(text: str, start: int) -> int:
    """The index after the closing tag of the DSML calls block that opens at ``start``, or -1 when it is open."""

    closed = _close_of(text, start + len(DSML_OPEN), _CALLS_TAGS)
    return closed.end() if closed else -1


def _named_in_prose(text: str, start: int, end: int) -> bool:
    """An opening tag written inline in prose (in backticks or quotes, or before a space or punctuation), not a call."""

    before = text[start - 1:start]
    after = text[end:end + 1]
    return bool(before) and before in "`'\"" or bool(after) and after in " `'\".,;:!?)"


def top_level_blocks(text: str, *, any_case: bool = True) -> tuple[list[Block], int]:
    """(the closed blocks at the top level, in order; where the first open block begins, or -1).

    A block inside a closed block is part of it, and a block of the same kind inside a block does not close it. The
    first block that never closes is open: it holds the rest of the text, and nothing in it is a block."""

    tag_tags = _TAG_TAGS_ANY if any_case else _TAG_TAGS
    blocks: list[Block] = []
    pos = 0
    tag = tag_tags[0].search(text)
    dsml = text.find(DSML_OPEN)
    while True:
        if tag is not None and tag.start() < pos:
            tag = tag_tags[0].search(text, pos)
        if 0 <= dsml < pos:
            dsml = text.find(DSML_OPEN, pos)
        if tag is None and dsml < 0:
            return blocks, -1
        if tag is not None and (dsml < 0 or tag.start() < dsml):
            closer = _close_of(text, tag.end(), tag_tags)
            if (dsml >= 0 and (closer is None or closer.start() > dsml)
                    and _named_in_prose(text, tag.start(), tag.end()) and dsml_block_end(text, dsml) >= 0):
                # prose naming the tag ("the `<tool_call>` tag"), not a call: it does not hold the complete DSML
                # calls block after it, nor reach into that block's arguments for a closing tag
                pos = tag.end()
                continue
            if closer is None:
                return blocks, tag.start()
            blocks.append(Block("tag", tag.start(), closer.end(), text[tag.end():closer.start()]))
            pos = closer.end()
        else:
            end = dsml_block_end(text, dsml)
            if end < 0:
                return blocks, dsml
            blocks.append(Block("dsml", dsml, end, text[dsml + len(DSML_OPEN):end - len(DSML_CLOSE)]))
            pos = end


def _parameters(body: str) -> dict[str, Any] | None:
    """The arguments of an invoke body, or None when the body is more than parameters and white space."""

    args: dict[str, Any] = {}
    at = _SPACE_RE.match(body, 0).end()
    while at < len(body):
        head = _PARAM_RE.match(body, at)
        close = _close_of(body, head.end(), _PARAM_TAGS) if head else None
        if close is None:
            return None
        key, is_string, value = head.group(1), head.group(2), body[head.end():close.start()]
        args[key] = value
        if is_string == "false":
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
        at = _SPACE_RE.match(body, close.end()).end()
    return args


def parse_calls_block(inner: str) -> list[tuple[str, dict[str, Any]]] | None:
    """Every invoke of a DSML calls block as (name, arguments), or None when the block is not well formed.

    The block holds invokes and white space only, an invoke holds parameters and white space only, and it holds one
    invoke at least. A value is the text up to its closing tag: it can hold any markup."""

    calls: list[tuple[str, dict[str, Any]]] = []
    at = _SPACE_RE.match(inner, 0).end()
    while at < len(inner):
        head = _INVOKE_RE.match(inner, at)
        close = _close_of(inner, head.end(), _INVOKE_TAGS) if head else None
        args = _parameters(inner[head.end():close.start()]) if close else None
        if args is None:
            return None
        calls.append((head.group(1), args))
        at = _SPACE_RE.match(inner, close.end()).end()
    return calls or None
