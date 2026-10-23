"""Coupled DSpark drafts (families/deepseek_v41/cuda/markov.py, ``TF_DS_DRAFT_COUPLED``): a sampled stream's draft is
the token its own keyed draw (engine/exact_sampling.py ``choose_rows``) gives on the drafter's scores, so a drafter
whose scores equal the target's logits drafts exactly the sampled tokens; a greedy stream's draft stays the argmax. The
fused steps (candidates in ``_score`` / ``_finish``, the draw in ``_pick``, one gather a step) on one rank and on two
ranks with the gather driven by hand need CUDA; the choice of the drafter graph (no coupled steps when no stream
samples) runs on the CPU."""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.engine.exact_sampling import Sampling, choose_rows
from tensorfold.families.deepseek_v41.cuda import dspark, markov


def graph_stub(N: int):
    g = object.__new__(dspark.BatchDraftGraph)
    g.N, g.steps, g.coupled, g.graph = N, 2, True, None
    g.inputs = torch.zeros((8, N), dtype=torch.long)
    g.ran = []
    g.packs = {False: torch.zeros((N, 4)), True: torch.ones((N, 4))}
    g._body = lambda coupled=False: g.ran.append(coupled)
    return g


def test_a_round_with_no_sampled_stream_runs_the_greedy_graph():
    one_copy = dspark._rounds.ONE_COPY
    dspark._rounds.ONE_COPY = False                    # no pinned memory on the CPU
    try:
        g = graph_stub(3)
        g.run([1, 2, 3], [10, 20, 30], [0, 1, 2], [None, Sampling(5, 0.0, 20), None])
        g.run([1, 2, 3], [10, 20, 30], [0, 1, 2])
        assert g.ran == [False, False]
        assert g.inputs[3:6].eq(0).all()                  # seeds, top_k, divisors: 0 (greedy)
        g.run([1, 2, 3], [10, 20, 30], [0, 1, 2], [Sampling(4, 1.0, 0), Sampling(4, 1.0, 33), None])
        assert g.ran[-1] is False                         # top_k 0 or above CAND: the candidates miss the rule
        g.run([1, 2, 3], [10, 20, 30], [0, 1, 2], [None, Sampling(2**63 + 7, 0.7, 32), Sampling(9, 1.0, 1, 1.0, 0.1)])
        assert g.ran[-1] is True
        assert g.inputs[3].tolist() == [0, 7 - 2**63, 9]   # a seed's 64 bits as int64
        assert g.inputs[4].tolist() == [0, 32, 1]
        div, top_p, min_log = (struct.unpack("<3d", struct.pack("<3q", *g.inputs[i].tolist())) for i in (5, 6, 7))
        assert div == (0.0, 0.7 * markov.COUPLED_SCALE, 1.0 * markov.COUPLED_SCALE)
        assert top_p == (1.0, 0.95, 1.0)                  # 1: off (Sampling's default top_p is 0.95)
        assert min_log == (-math.inf, -math.inf, math.log(0.1))
        g.coupled = False                                 # a loop that cannot couple: greedy drafts in any case
        g.run([1, 2, 3], [10, 20, 30], [0, 1, 2], [Sampling(1, 1.0, 20)] * 3)
        assert g.ran[-1] is False
    finally:
        dspark._rounds.ONE_COPY = one_copy


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
V, RANK, BLOCK = 5120, 64, 5


def weights():
    g = torch.Generator().manual_seed(11)
    head = (torch.randn((V, RANK), generator=g) * 0.25).half().cuda()
    emb = torch.randn((V, RANK), generator=g).bfloat16().cuda()
    return head, emb


def markov_of(head, emb, world: int, rank: int, tokens):
    w = SimpleNamespace(world=world, rank=rank, dspark=SimpleNamespace(markov_head=head, markov_embed=emb))
    model = SimpleNamespace(w=w, cfg=SimpleNamespace(vocab=V), comm=None)
    return markov.Markov(model, tokens=tokens, rows=len(tokens), split=world > 1)


def drive(mks, lg, tok0, samp, steps):
    """The Markov steps of ``mks`` (one Markov a rank) on the full logits lg [N, BLOCK, V]; two ranks: each rank's
    half, the gather by hand. Every rank's drafts must be the same."""

    N = lg.shape[0]
    outs = []
    for _ in mks:
        o = torch.zeros((N, steps + 1), dtype=torch.long, device="cuda")
        o[:, 0] = tok0
        outs.append(o)
    if len(mks) == 1:
        mks[0].steps(lg.reshape(1, N * BLOCK, V), outs[0], BLOCK, steps, samp)
    else:
        h = V // 2
        halves = [lg[..., j * h:(j + 1) * h].reshape(N * BLOCK, h).contiguous() for j in range(2)]
        for i in range(steps):
            sends = [mk.local_best(halves[j], outs[j], BLOCK, i, samp=samp).clone() for j, mk in enumerate(mks)]
            g = torch.stack(sends)
            for j, mk in enumerate(mks):
                mk.pick(g, outs[j], i, samp)
        assert torch.equal(outs[0], outs[1])
    torch.cuda.synchronize()
    return outs[0].cpu()


def check(out, lg, full, world, seeds, temps, top_k, q0, top_p=1.0, min_p=0.0):
    """Each step's draft from its own previous draft: the argmax (greedy row) or choose_rows's draw on the step's
    scores (logit + the token's bias row, fp32) over the candidates the ranks send (with top_k <= CAND: the whole
    vocabulary)."""

    lg = lg.cpu()
    cache = full.cache.cpu()
    slot = full.slot.cpu()
    ids_all = np.arange(V, dtype=np.int64)
    for r in range(out.shape[0]):
        for i in range(out.shape[1] - 1):
            score = (lg[r, i] + cache[slot[out[r, i]]].float()).numpy()
            if temps[r] == 0:
                assert out[r, i + 1] == int(np.argmax(score)), (r, i)
                continue
            s = Sampling(seeds[r], temps[r] * markov.COUPLED_SCALE, top_k, top_p, min_p)
            if 0 < top_k <= markov.CAND:
                vals, ids = score[None], ids_all[None]
            else:                                         # the ranks' CAND best each, then all of them
                h = V // world
                keep = []
                for j in range(world):
                    part = np.lexsort((ids_all[j * h:(j + 1) * h], -score[j * h:(j + 1) * h]))[:markov.CAND]
                    keep.append(part + j * h)
                ids = np.concatenate(keep)[None]
                vals = score[ids]
            want = choose_rows(vals, ids, [int(q0[r]) + 1 + i], s)[0]
            assert out[r, i + 1] == want, (r, i, top_k)


@cuda
@pytest.mark.parametrize("rule", [(5, 1.0, 0.0), (20, 1.0, 0.0), (0, 1.0, 0.0), (20, 0.95, 0.0), (20, 0.7, 0.1),
                                  (0, 0.95, 0.05)])
def test_coupled_drafts_are_the_sampled_tokens(rule):
    top_k, top_p, min_p = rule
    head, emb = weights()
    cached = list(range(0, V, 2))                         # half the tokens cached: both bias paths
    full = markov_of(head, emb, 1, 0, list(range(V)))     # every token's row: the reference bias bits
    torch.manual_seed(3)
    N = 4
    lg = (torch.randn((N, BLOCK, V), device="cuda") * 3.0).float()
    lg[:, :, 100:104] = 9.0                               # near the top: candidates across programs and ranks
    lg[:, :, 3000:3002] = 9.0
    seeds = [11, 22, 2**61 + 5, 7]
    temps = [1.0, 0.0, 0.7, 1.3]
    q0 = [100, 2000, 5, 2**20 + 3]
    samp = (torch.tensor(seeds, dtype=torch.int64, device="cuda"),
            torch.tensor([t * markov.COUPLED_SCALE for t in temps], dtype=torch.float64, device="cuda"),
            torch.full((N,), top_k, dtype=torch.int64, device="cuda"),
            torch.tensor(q0, dtype=torch.int64, device="cuda"),
            torch.full((N,), top_p, dtype=torch.float64, device="cuda"),
            torch.full((N,), math.log(min_p) if min_p > 0 else -math.inf, dtype=torch.float64, device="cuda"))
    tok0 = torch.tensor([1, 2, 3, 4], device="cuda")
    for world in (1, 2):
        mks = [markov_of(head, emb, world, rk, cached) for rk in range(world)]
        out = drive(mks, lg, tok0, samp, BLOCK)
        check(out, lg, full, world, seeds, temps, top_k, q0, top_p, min_p)
        greedy = drive(mks, lg, tok0, None, BLOCK)       # no coupled steps: every row the argmax chain
        assert torch.equal(out[1], greedy[1])
        check(greedy, lg, full, world, seeds, [0.0] * N, top_k, q0)
