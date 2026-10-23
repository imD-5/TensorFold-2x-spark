"""Token ids of re-sent chat prompts: each piece between added tokens is encoded once, then reused.

A tokenizer splits its input at the added tokens it matches before normalization and encodes each part on its own, so
a piece that ends in one encodes alone to the ids it gets inside the whole text. A chat client re-sends its whole
conversation every turn; only the pieces it has not sent before need the tokenizer.
"""

from __future__ import annotations

import re
import threading
from array import array
from collections import OrderedDict
from typing import Any

BUDGET = 1 << 21        # tokens kept across all pieces (4 bytes each, plus the pieces' text)


class PromptPieces:
    """``ids(text)`` equals ``tok.encode(text, add_special_tokens=False).ids``."""

    def __init__(self, tok: Any, cut: re.Pattern[str], budget: int = BUDGET) -> None:
        self._tok, self._cut, self._budget = tok, cut, budget
        self._kept: OrderedDict[str, array] = OrderedDict()      # least recently used first
        self._held = 0
        self._lock = threading.Lock()

    @classmethod
    def of(cls, tok: Any, budget: int = BUDGET) -> PromptPieces | None:
        """Pieces for ``tok``, or None where a cut could fall where the tokenizer does not split."""

        if not all(callable(getattr(tok, name, None)) for name in ("encode_batch", "get_added_tokens_decoder")):
            return None
        if getattr(tok, "encode_special_tokens", False):        # special tokens are encoded as text, not split out
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
                if piece not in self._kept and len(ids) <= self._budget:
                    self._kept[piece] = ids
                    self._held += len(ids)
            while self._held > self._budget:
                self._held -= len(self._kept.popitem(last=False)[1])
