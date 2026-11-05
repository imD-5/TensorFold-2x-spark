"""Token ids of re-sent chat prompts: each piece between added tokens is encoded once, then reused.

Use pieces only for a deterministic model whose pre-tokenizer does not depend on the original input position.
A chat client re-sends its whole conversation every turn; only new pieces need the tokenizer.
"""

from __future__ import annotations

import json
import re
import threading
from array import array
from collections import OrderedDict
from typing import Any

BUDGET = 1 << 21        # cache units: token IDs, text characters and 64 units per entry


def _local_pre_tokenizer(state: Any) -> bool:
    """Refuse pre-tokenizers that need the original offset or whose state is unknown."""

    if not isinstance(state, dict):
        return False
    kind = state.get("type")
    if kind == "Sequence":
        children = state.get("pretokenizers")
        return isinstance(children, list) and all(_local_pre_tokenizer(child) for child in children)
    if kind == "Metaspace":
        return state.get("prepend_scheme") in ("always", "never")
    return isinstance(kind, str) and kind in {
        "BertPreTokenizer", "ByteLevel", "CharDelimiterSplit", "Digits", "FixedLength", "Punctuation", "Split",
        "UnicodeScripts", "Whitespace", "WhitespaceSplit",
    }


def _has_local_pre_tokenizer(tok: Any) -> bool:
    try:
        pre = tok.pre_tokenizer
        return pre is None or _local_pre_tokenizer(json.loads(pre.__getstate__()))
    except Exception:  # noqa: BLE001  custom state can refuse serialization; use the whole-input path then
        return False


def _deterministic_model(tok: Any) -> bool:
    """Read the model options without serializing its vocabulary."""

    try:
        from tokenizers.models import BPE, Unigram, WordLevel, WordPiece

        model = tok.model
        if type(model) is BPE:
            return model.dropout is None or model.dropout == 0.0
        return type(model) in (Unigram, WordLevel, WordPiece)
    except Exception:  # noqa: BLE001  unknown model metadata must not enable the optional cache
        return False


class PromptPieces:
    """``ids(text)`` equals ``tok.encode(text, add_special_tokens=False).ids``."""

    def __init__(self, tok: Any, cut: re.Pattern[str], budget: int = BUDGET) -> None:
        self._tok, self._cut, self._budget = tok, cut, budget
        self._kept: OrderedDict[str, array] = OrderedDict()      # least recently used first
        self._held = 0
        self._used = 0
        self._lock = threading.Lock()

    @classmethod
    def of(cls, tok: Any, budget: int = BUDGET) -> PromptPieces | None:
        """Pieces for ``tok``, or None where a cut could fall where the tokenizer does not split."""

        if not all(callable(getattr(tok, name, None)) for name in ("encode_batch", "get_added_tokens_decoder")):
            return None
        if getattr(tok, "padding", None) is not None or getattr(tok, "truncation", None) is not None:
            return None                              # these options apply to the whole input, not each piece
        if getattr(tok, "encode_special_tokens", False):        # special tokens are encoded as text, not split out
            return None
        if not _deterministic_model(tok) or not _has_local_pre_tokenizer(tok):
            return None
        first = [token for token in tok.get_added_tokens_decoder().values() if not token.normalized]
        # rstrip takes the spaces after a token into its match, and single_word skips a match inside a word
        if not first or any(token.rstrip or token.single_word for token in first):
            return None
        spellings = sorted({token.content for token in first if token.content}, key=len, reverse=True)
        return cls(tok, re.compile("|".join(map(re.escape, spellings))), budget)     # the longest at the leftmost

    def ids(self, text: str) -> list[int]:
        pieces, at = [], 0
        for match in self._cut.finditer(text):
            pieces.append(text[at:match.end()])
            at = match.end()
        if at < len(text):
            pieces.append(text[at:])
        with self._lock:
            found = {piece: self._kept[piece] for piece in pieces if piece in self._kept}
            for piece in found:
                self._kept.move_to_end(piece)
        missing = [piece for piece in dict.fromkeys(pieces) if piece not in found]
        if missing:                       # encode_batch releases the GIL while it encodes
            encoded = self._tok.encode_batch(missing, add_special_tokens=False)
            new = {piece: array("I", encoding.ids) for piece, encoding in zip(missing, encoded)}
            found.update(new)
            self._keep(new)
        ids: list[int] = []
        for piece in pieces:
            ids.extend(found[piece])
        return ids

    def _keep(self, new: dict[str, array]) -> None:
        with self._lock:
            for piece, ids in new.items():
                cost = len(ids) + len(piece) + 64
                if piece not in self._kept and cost <= self._budget:
                    self._kept[piece] = ids
                    self._held += len(ids)
                    self._used += cost
            while self._used > self._budget:
                piece, ids = self._kept.popitem(last=False)
                self._held -= len(ids)
                self._used -= len(ids) + len(piece) + 64
