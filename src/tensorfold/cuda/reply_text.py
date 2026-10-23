"""The CUDA server's reply text: streamed decoding, think blocks and tool-call parsing (the Mac lane server's rules)."""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Iterator
from typing import Any

from tensorfold.dsml_calls import (CALL_CLOSE as _CALL_CLOSE, CALL_OPEN as _CALL_OPEN, DSML_OPEN as _DSML_OPEN,
                                   ILL_FORMED, OPEN_DSML, OPEN_TAG, count_suppressed, parse_calls_block,
                                   top_level_blocks)
from tensorfold.server.stopping import StopPolicy
from tensorfold.server.tools import parse_glm_tool_call_block
from tensorfold.tool_parameters import decode_parameter, parameter_schemas

_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)

# DeepSeek-V4.x DSML tool calls (the format DeepSeek's own encoding.py writes and its chat template asks for):
#   <｜DSML｜ calls>\n<｜DSML｜ invoke name="f">\n<｜DSML｜ parameter name="k" string="true">v</｜DSML｜ parameter>\n
#   </｜DSML｜ invoke>\n</｜DSML｜ calls>   (string="false": the value is JSON)
# ``tensorfold.dsml_calls`` finds the blocks: a call is a complete, well formed block at the top level of the reply.


def _segments(text: str, suppressed: list[tuple[str, int]]) -> Iterator[str | tuple[str, str]]:
    """The reply as text (a plain string) and call blocks ((its markup, its JSON payload)), in order.

    A DSML invoke becomes the payload of a ``<tool_call>`` block. Only blocks at the top level are calls: an open
    ``<tool_call>`` block, a block that is not well formed and everything inside a block stay text or argument text.
    An open DSML calls block (the reply ended inside it) is removed with everything after it: it is not a call, and
    the stream hides it too. A block that keeps a later call as text adds (reason, character offset) to
    ``suppressed``."""

    blocks, open_at = top_level_blocks(text)
    dsml_open = open_at >= 0 and text.startswith(_DSML_OPEN, open_at)
    if open_at >= 0:
        opener = _DSML_OPEN if dsml_open else _CALL_OPEN
        if top_level_blocks(text[open_at + len(opener):])[0]:        # a complete block that the open one holds
            suppressed.append((OPEN_DSML if dsml_open else OPEN_TAG, open_at))
    cursor = 0
    for block in blocks:
        yield text[cursor:block.start]
        cursor = block.end
        if block.kind == "tag":
            yield text[block.start:block.end], block.inner.strip()
            continue
        invokes = parse_calls_block(block.inner)
        if invokes is None:
            suppressed.append((ILL_FORMED, block.start))
            yield text[block.start:block.end]
            continue
        for name, args in invokes:
            payload = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
            yield _CALL_OPEN + payload + _CALL_CLOSE, payload
    yield text[cursor:open_at] if dsml_open else text[cursor:]


def _partial_tag(text: str, tag: str, *, any_case: bool = False) -> int:
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if (text[-k:].lower() == tag[:k].lower()) if any_case else text.endswith(tag[:k]):
            return k
    return 0


class StreamDecoder:
    """Decode a shared token window to preserve leading spaces and byte boundaries, deferring incomplete characters."""

    def __init__(self, tok, skip: tuple[int, ...] = ()):
        self.tok, self.skip = tok, frozenset(skip)
        self.ids: list[int] = []
        self.text = ""
        self.prefix = 0             # window start
        self.read = 0               # tokens already reflected in ``text``

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def add(self, new: list[int]) -> str:
        self.ids.extend(t for t in new if t not in self.skip)
        before = self._decode(self.ids[self.prefix:self.read])
        after = self._decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
        return self.text

    def final(self) -> str:
        """Everything, including a trailing partial character (as decoding it all at once gives)."""

        before = self._decode(self.ids[self.prefix:self.read])
        return self.text + self._decode(self.ids[self.prefix:])[len(before):]



class StopStrings:
    """A request's ``stop`` strings, decided token by token, so a reply cuts at the same token however its tokens arrive."""

    def __init__(self, strings: tuple[str, ...], tok, skip: tuple[int, ...] = ()):
        self.strings, self.tok, self.skip = strings, tok, frozenset(skip)
        # a new match lies in the last (its UTF-8 length) tokens, plus eight as the Mac's ``StopPolicy`` keeps
        self.tail = max((len(s.encode()) for s in strings), default=0) + 8

    def hit(self, tokens: list[int]) -> bool:
        """Whether the text through the newest token holds a stop string (earlier tokens were checked already)."""

        ids = [t for t in tokens[-self.tail:] if t not in self.skip]
        text = self.tok.decode(ids, skip_special_tokens=False)
        return any(stop in text for stop in self.strings)

    def visible(self, text: str, *, partial: bool = False) -> str:
        """The text before the first match; ``partial`` also holds back an end that may begin one (the Mac's rule)."""

        return StopPolicy.visible(self, text, partial=partial)


def drop_unoffered_calls() -> bool:
    """TF_DROP_UNOFFERED_TOOL_CALLS=1: a call to a tool that the request did not offer is removed from the content
    and logged, instead of kept as text. A model can call a tool from an earlier turn or another harness; a client
    then shows its markup as text."""

    return os.environ.get("TF_DROP_UNOFFERED_TOOL_CALLS", "0") == "1"


def keep_unoffered_call(text: str, name: Any) -> str:
    """The content kept for a parsed call to a tool the request did not offer: its text, or nothing with the flag."""

    if not drop_unoffered_calls():
        return text
    print(f"[tensorfold] dropped a call to a tool the request did not offer: {str(name)[:80]!r}", flush=True)
    return ""


def answer_without_tools(answer: str, *, finished: bool) -> str:
    """The answer of a request that offers no tools: as it is, or with the flag, without its tool-call blocks (each
    call is unoffered). While streaming, a block that has begun waits until it ends, as with tools. The finished
    answer logs that blocks were dropped."""

    if not drop_unoffered_calls():
        return answer
    # the parser finds tags in any case, and keeps a block that is not well formed as text
    shown = hide_tool_calls(answer, finished=finished, any_case=True, keep_ill_formed=True)
    if finished and shown != answer:
        print("[tensorfold] dropped tool-call markup from a request that offered no tools", flush=True)
    return shown


def hide_tool_calls(text: str, *, finished: bool, any_case: bool = False, keep_ill_formed: bool = False) -> str:
    """The text without its top-level tool-call blocks. A block that is still open hides the rest of the text, and
    while streaming, so does a tail that may begin an opening tag. The parser finds the same blocks.

    ``keep_ill_formed``: a closed DSML block that is not well formed is not a call, so it stays in the text, as the
    parser keeps it in the content."""

    blocks, open_at = top_level_blocks(text, any_case=any_case)
    end = len(text) if open_at < 0 else open_at
    out: list[str] = []
    pos = 0
    for block in blocks:
        if block.start >= end:
            break
        if keep_ill_formed and block.kind == "dsml" and parse_calls_block(block.inner) is None:
            continue
        out.append(text[pos:block.start])
        pos = block.end
    tail = text[pos:end]
    if open_at >= 0:
        return "".join(out) + tail
    if not finished:
        tail = tail[: len(tail) - _partial_tag(tail, _CALL_OPEN, any_case=any_case)]
    shown = "".join(out) + tail
    return shown if finished else shown[: len(shown) - _partial_tag(shown, _DSML_OPEN)]


def _finite_json(value: Any) -> bool:
    """True when the value is valid strict JSON (json.loads accepts NaN and Infinity, the call path does not)."""

    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError):
        return False
    return True


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def parse_tool_calls(text: str, tools: list[dict[str, Any]], *, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
    """DeepSeek DSML calls, and around them Qwen ``<function=name><parameter=k>v</parameter></function>``, GLM
    ``name<arg_key>..`` or JSON calls."""

    if not tools:
        return text, None
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    schemas = parameter_schemas(tools)
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    suppressed: list[tuple[str, int]] = []
    for segment in _segments(text, suppressed):
        if isinstance(segment, str):
            residue.append(segment)
            continue
        markup, block = segment
        if max_calls is not None and len(calls) >= max_calls:
            continue
        name, args = None, {}
        parsed = False                                  # the whole payload parsed, not only its name
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                # a call needs a name and object arguments that are strict JSON
                parsed = isinstance(name, str) and isinstance(args, dict) and _finite_json(args)
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                if max_calls is not None and _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip():
                    continue
                name = m.group(1).strip()
                # typed parameters (array, object, number...) decode per the tool's schema, as the Mac server does
                props = schemas.get(name.lower(), {})
                args = {key: decode_parameter(value, props.get(key, {}))
                        for key, value in ((p.group(1).strip(), p.group(2))
                                           for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2)))}
                parsed = not _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip()   # no text outside the parameters
            else:
                glm = parse_glm_tool_call_block(block, tools, complete=max_calls is not None)
                if glm is not None:
                    name, args = glm
                    # no text outside the arguments, as the complete parser requires
                    parsed = parse_glm_tool_call_block(block, tools, complete=True) is not None
        if not name or str(name).lower() not in known:
            if max_calls is None:
                residue.append(keep_unoffered_call(markup, name) if name and parsed else markup)
            elif name and parsed and drop_unoffered_calls():
                keep_unoffered_call(markup, name)    # single-call mode removes the block anyway: log it
            continue
        if max_calls is not None:
            try:
                if not isinstance(args, dict):
                    continue
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError):
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    if suppressed:
        reason, offset = min(suppressed, key=lambda found: found[1])
        count_suppressed({reason for reason, _ in suppressed})
        print(f"[tensorfold] WARNING: a tool call was kept as text: {reason} at character {offset} of the reply "
              f"({len(suppressed)} such block{'s' if len(suppressed) > 1 else ''})", flush=True)
    return "".join(residue).strip(), calls or None


# -- chat template -------------------------------------------------------------------------
