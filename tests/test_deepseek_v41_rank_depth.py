"""The "rank" depth policy (families/deepseek_v41/cuda/multi.py ``_rank_k``): the drafts a round verifies across all
streams are the allocation with the most expected tokens a millisecond under the cost table, each stream's a
prefix, within the round's row cap; one stream gets the k of the "conf" policy; a constrained stream's drafts its
grammar cuts take no row, nor a draft past the reply's last token; its drafter graphs run no more steps than the configured drafts."""

from __future__ import annotations

import itertools
import math
import random
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import multi


def decoder(max_rows: int = 16, depth: int = 5) -> multi.MultiDecoder:
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.max_rows, dec.depth_most = max_rows, depth
    return dec


def value(confs, ks, live):
    tokens = float(live)
    for cf, k in zip(confs, ks):
        surv = 1.0
        for j in range(k):
            surv *= 1.0 / (1.0 + math.exp(-cf[j]))
            tokens += surv
    rows = live + sum(ks)
    return tokens / (multi.ROUND_MS[min(len(multi.ROUND_MS), rows) - 1] + multi.DRAFT_MS)


def test_rank_is_the_best_allocation():
    rng = random.Random(5)
    dec = decoder()
    for _ in range(200):
        n = rng.randint(2, 4)
        confs = [[rng.uniform(-3.0, 6.0) for _ in range(5)] for _ in range(n)]
        want = [rng.randint(0, 5) for _ in range(n)]
        ks = dec._rank_k(n, confs, want)
        assert all(0 <= k <= w for k, w in zip(ks, want)) and n + sum(ks) <= dec.max_rows
        best = max(value(confs, c, n) for c in itertools.product(*[range(w + 1) for w in want])
                   if n + sum(c) <= dec.max_rows)
        assert value(confs, ks, n) == pytest.approx(best, rel=1e-12)


def test_one_stream_takes_the_conf_k():
    rng = random.Random(9)
    dec = decoder()
    for _ in range(200):
        cf = [rng.uniform(-3.0, 6.0) for _ in range(5)]
        assert dec._rank_k(1, [cf], [5]) == [dec._choose_k(cf, 5, 1)]


def test_sure_streams_fill_the_row_cap():
    dec = decoder()
    ks = dec._rank_k(4, [[9.0] * 5] * 4, [5] * 4)
    assert sum(ks) == dec.max_rows - 4 and max(ks) - min(ks) <= 1
    ks = dec._rank_k(4, [[9.0] * 5, [-4.0] * 5, [-4.0] * 5, [-4.0] * 5], [5] * 4)
    assert ks[0] == 5 and ks[1:] == [0, 0, 0]


class Prefix:                                      # a grammar that keeps the drafts up to the first one not in ``ok``
    def __init__(self, ok, fail=False):
        self.ok, self.fail = set(ok), fail

    def window(self, tokens, parents):
        if self.fail:
            raise RuntimeError("grammar failed")
        n = 1
        while n < len(tokens) and tokens[n] in self.ok:
            n += 1
        return SimpleNamespace(tokens=list(tokens[:n]))


def stream(sid, constraint=None, left=100):
    return SimpleNamespace(sid=sid, pending=0, constraint=constraint, count=left, out=[])


def test_cut_drafts_take_no_row():
    dec = decoder(max_rows=8)
    drafting = [stream(0, Prefix(ok={7})), stream(1), stream(2, Prefix(ok=set(), fail=True))]
    rows = [[7, 9, 9, 9, 9], [5, 5, 5, 5, 5], [3, 3, 3, 3, 3]]
    pre = {}
    caps = dec._grammar_caps(drafting, rows, {0: 5, 1: 5, 2: 4}, pre)
    assert caps == [1, 5, 4]                       # the failed grammar keeps its cap: the round's chain ends it
    assert pre[0][0] == [7] and 1 not in pre and 2 not in pre
    ks = dec._rank_k(3, [[9.0] * 5] * 3, caps)
    assert ks[0] <= 1 and 3 + sum(ks) == dec.max_rows and ks[1] >= 2


def test_no_row_for_a_draft_past_the_last_token():
    dec = decoder(max_rows=8)
    drafting = [stream(0, left=3), stream(1, left=1), stream(2)]
    caps = dec._grammar_caps(drafting, [[1] * 5] * 3, {0: 3, 1: 1, 2: 5}, {})
    assert caps == [2, 0, 5]                       # 2 kept drafts and the round's token end a 3-token reply


def test_rank_drafter_steps_stop_at_the_configured_drafts(monkeypatch):
    import torch

    from tensorfold.families.deepseek_v41.cuda import dspark

    made = []

    class Graph:
        def __init__(self, drafter, sc, dpool, streams, steps=None):
            made.append(steps)
            self.tokens, self.q0, self.slots = (torch.zeros(streams, dtype=torch.long) for _ in range(3))

        def run(self, tokens, q0, slots, samplings=None):
            return [[] for _ in tokens]

    monkeypatch.setattr(dspark, "BatchDraftGraph", Graph)
    monkeypatch.setattr(multi, "DEPTH_POLICY", "rank")
    dec = decoder(depth=3)
    dec.drafters, dec.slots, dec.pool, dec.dpool = {}, [0] * 4, None, None
    dec.e = SimpleNamespace(drafter=object(), runner=None)
    dec.m = SimpleNamespace(pool_view=lambda *a: None)
    dec.extents = SimpleNamespace(total=1)
    for n in range(1, 5):
        dec._drafts([0] * n, [1] * n, list(range(n)))
    assert made == [3, 3, 3, 3]
