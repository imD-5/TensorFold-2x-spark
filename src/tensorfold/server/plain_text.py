"""Request text stays plain text: a special-token spelling that a client writes in a message is not a special token.

The chat template makes one string, and the tokenizer turns each added-token spelling in it into that token's id. A
client that writes ``</think>`` or ``<｜User｜>`` in a message would then forge a thinking end or a turn.
``PlainText.shield`` swaps every spelling in the request's own strings for a marker before the template runs, and
``PlainText.encode`` tokenizes the spellings that remain (the template's own) as tokens and the other text, with the
markers turned back into spellings, without the added vocabulary. This is deepseek-ai/deepseek-recipe's rule."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


def _trie_pattern(words: Iterable[str]) -> re.Pattern[str]:
    """A pattern for the longest of ``words`` at a position (a prefix tree keeps a scan of a long text fast)."""

    root: dict[str, Any] = {}
    for word in words:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node[""] = True

    def build(node: dict[str, Any]) -> str:
        parts = [re.escape(char) + build(child) for char, child in sorted(node.items()) if char]
        if not parts:
            return ""
        body = "(?:" + "|".join(parts) + ")"
        return body + "?" if "" in node else body

    return re.compile(build(root))


@dataclass(slots=True)
class Shielded:
    """A request's strings after ``PlainText.shield``: ``table`` maps each marker to the spelling it replaced."""

    messages: Any
    tools: Any
    extra: Any
    table: dict[str, str] = field(default_factory=dict)
    restore: re.Pattern[str] | None = None

    def spelled(self, text: str) -> str:
        """``text`` with each marker turned back into its spelling (an error message, say)."""

        if not self.table or self.restore is None:
            return text
        return self.restore.sub(lambda match: self.table[match.group(0)], text)


class PlainText:
    """Keeps the added-token spellings in a request's strings as plain text, for one tokenizer."""

    def __init__(self, tok: Any, *, assistant_keep: Iterable[str] = ()) -> None:
        from tokenizers import Tokenizer

        self.tok = tok
        decoder = tok.get_added_tokens_decoder()
        self.token_ids = frozenset(int(i) for i in decoder)
        self.spellings = frozenset(token.content for token in decoder.values() if token.content)
        self.pattern = _trie_pattern(self.spellings)
        self.assistant_keep = frozenset(assistant_keep)       # left as written in an assistant's content and reasoning
        plain = json.loads(tok.to_str())
        normalizer = plain.get("normalizer") or {}
        # a normalizer lets the tokenizer find a spelling in other characters (full-width ones, say) than the trie does
        self.normalizes = bool(normalizer) and normalizer.get("normalizers", [None]) != []
        plain["added_tokens"] = []
        self.plain = Tokenizer.from_str(json.dumps(plain))

    @staticmethod
    def supports(tok: Any) -> bool:
        names = ("get_added_tokens_decoder", "to_str", "encode_batch")
        return all(callable(getattr(tok, name, None)) for name in names)

    def _strings(self, value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from self._strings(key)
                yield from self._strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from self._strings(item)

    def _walk(self, value: Any, keep: frozenset[str], mark: Callable[[str], str]) -> Any:
        if isinstance(value, str):
            return self.pattern.sub(lambda match: match.group(0) if match.group(0) in keep else mark(match.group(0)),
                                    value)
        if isinstance(value, dict):
            return {self._walk(key, keep, mark): self._walk(item, keep, mark) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._walk(item, keep, mark) for item in value]
        return value

    @staticmethod
    def _joined(content: Any) -> Any:
        """Text parts as one string, as ``normalize_messages`` joins them: a spelling can span two parts."""

        if isinstance(content, list) and content and all(
                isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                for part in content):
            return "".join(part["text"] for part in content)
        return content

    def _messages(self, messages: Any, mark: Callable[[str], str], content_keep: frozenset[str]) -> Any:
        if not isinstance(messages, list):
            return self._walk(messages, frozenset(), mark)
        out = []
        for message in messages:
            if not isinstance(message, dict):
                out.append(self._walk(message, frozenset(), mark))
                continue
            if "content" in message:
                message = {**message, "content": self._joined(message["content"])}
            assistant = message.get("role") == "assistant"
            keep = {"content": content_keep | (self.assistant_keep if assistant else frozenset()),
                    "reasoning_content": self.assistant_keep if assistant else frozenset()}
            out.append({self._walk(key, frozenset(), mark): self._walk(item, keep.get(key, frozenset()), mark)
                        for key, item in message.items()})
        return out

    def shield(self, messages: Any, tools: Any = None, extra: Any = None, *,
               content_keep: Iterable[str] = ()) -> Shielded:
        """The inputs with each added-token spelling replaced by a marker; the same objects when none is there.
        ``content_keep``: spellings that the server wrote into this request's message content itself."""

        keep = frozenset(content_keep)
        while True:
            prefix = f"@TF{secrets.token_hex(8)}:"
            table: dict[str, str] = {}
            index: dict[str, str] = {}

            def mark(spelling: str) -> str:
                if spelling not in index:
                    index[spelling] = f"{prefix}{len(index)}@"
                    table[index[spelling]] = spelling
                return index[spelling]

            shielded = Shielded(self._messages(messages, mark, keep), self._walk(tools, frozenset(), mark),
                                self._walk(extra, frozenset(), mark), table)
            if not table:
                return Shielded(messages, tools, extra)
            if not any(prefix in text for text in self._strings([messages, tools, extra])):    # a marker is unique
                shielded.restore = re.compile(re.escape(prefix) + r"\d+@")
                return shielded

    def encode(self, text: str, shielded: Shielded) -> list[int]:
        """The ids of rendered ``text``: its spellings as tokens, and the rest (markers restored) without them."""

        encoding = self.tok.encode_batch([text], add_special_tokens=False)[0]
        marked = bool(shielded.table) and shielded.restore is not None and shielded.restore.search(text) is not None
        if not marked and not self.normalizes:
            return list(encoding.ids)
        ids, offsets = encoding.ids, encoding.offsets
        found = [(start, end, token) for token, (start, end) in zip(ids, offsets) if token in self.token_ids]
        # a token that the text does not spell as written was found through the normalizer: it is a forgery
        cuts = [cut for cut in found if text[cut[0]:cut[1]] in self.spellings]
        if not marked and len(cuts) == len(found):
            return list(encoding.ids)
        regions, at = [], 0
        for start, end, _ in cuts:
            regions.append(shielded.spelled(text[at:start]))
            at = end
        regions.append(shielded.spelled(text[at:]))
        pieces = self.plain.encode_batch(regions, add_special_tokens=False)
        out: list[int] = []
        for piece, (_, _, token) in zip(pieces, cuts):
            out.extend(piece.ids)
            out.append(int(token))
        out.extend(pieces[-1].ids)
        return out
