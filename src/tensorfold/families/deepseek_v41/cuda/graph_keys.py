"""The keys of the decode graphs: context buckets, and the row sizes a round is captured for. No torch here."""

from __future__ import annotations

import os

BUCKET_MIN = int(os.environ.get("TF_DS_BUCKET_MIN") or 1024)

# A captured graph holds about 12 MiB of system memory that torch allocator does not count (on a GB10 it is the same
# memory as the host). A round graph is one per (rows, context bucket): a graph for every row count 1 .. 16 at every
# bucket up to a 1M window is 192 graphs, 2.4 GiB. A round of R rows therefore runs in the graph of the next size up
# (the engine most rows is always a size), its extra rows copies of its last row: a copy computes and writes what its
# original does, so the other rows of the round keep their bits. A padded row costs a little decode time, and the row
# counts that rounds really have (about 1 to 6 at one stream, 11 to 16 at four) are too many for a graph each, so:
# - contexts up to FINE_CONTEXT (where most rounds run): ROUND_ROWS, the sizes that cover the common counts closely;
# - longer contexts (few rounds, a graph is the same memory): ROUND_ROWS_LONG, a short ladder.
# The sizes are in rows (TF_DS_ROUND_ROWS, TF_DS_ROUND_ROWS_LONG; 0: a graph for every row count). The warm-up captures
# every graph, so no round captures while serving.
def _sizes(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(v) for v in (os.environ.get(name) or default).split(",") if v.strip())


ROUND_ROWS = _sizes("TF_DS_ROUND_ROWS", "1,2,3,4,6,8,12,16")
ROUND_ROWS_LONG = _sizes("TF_DS_ROUND_ROWS_LONG", "1,2,4,8,16")
FINE_CONTEXT = int(os.environ.get("TF_DS_FINE_CONTEXT") or 32768)

# A capture fails if another thread makes a call CUDA forbids during a "global" capture (an event record, a pinned
# host allocation). "thread_local" checks the capturing thread's own calls only (TF_DS_CAPTURE_MODE=global: as before).
CAPTURE_MODE = os.environ.get("TF_DS_CAPTURE_MODE") or "thread_local"


def bucket_for(n_tokens: int, cap: int) -> int:
    b = BUCKET_MIN
    while b < n_tokens:
        b *= 2
    return min(b, max(cap, BUCKET_MIN))


def context_buckets(limit: int, cap: int) -> list[int]:
    """Every context bucket a window that ends at or before ``limit`` tokens can use, largest first."""

    out, b = set(), BUCKET_MIN
    while b < limit:
        out.add(bucket_for(b, cap))
        b *= 2
    out.add(bucket_for(limit, cap))
    return sorted(out, reverse=True)


def row_sizes(max_rows: int, sizes: tuple[int, ...] = ROUND_ROWS) -> list[int]:
    """The row counts a round graph is captured for: ``sizes`` below ``max_rows``, and ``max_rows``; with a 0 in
    ``sizes`` (TF_DS_ROUND_ROWS=0): every count 1 .. ``max_rows``."""

    if not sizes or 0 in sizes:
        return list(range(1, max_rows + 1))
    return sorted({v for v in sizes if 0 < v < max_rows} | {max_rows})


def padded_rows(rows: int, sizes: list[int]) -> int:
    """The smallest captured size that holds ``rows``."""

    return next(v for v in sizes if v >= rows)


def sizes_for(bucket: int, fine: list[int], long: list[int] | None = None, fine_context: int = FINE_CONTEXT) -> list[int]:
    """The row sizes captured at a context bucket: ``fine`` up to ``fine_context``, ``long`` above it."""

    return fine if long is None or bucket <= fine_context else long


def round_keys(limit: int, cap: int, fine: list[int], long: list[int] | None = None,
               fine_context: int = FINE_CONTEXT) -> list[tuple[int, int]]:
    """Every (rows, bucket) a concurrent round can use, in warm-up order: the largest bucket first, so that its
    capture sizes the shared graph pool, and the most rows first within a bucket."""

    return [(rows, b) for b in context_buckets(limit, cap)
            for rows in sorted(sizes_for(b, fine, long, fine_context), reverse=True)]
