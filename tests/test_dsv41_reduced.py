"""Comm.reduced (TF_DS_RS): a prompt chunk's partials as bf16(p0 + p1) from a reduce-scatter and a bf16 all-gather,
the same bits as the y the mHC post kernels form from the fp32 gather; the gather itself where it does not apply."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import model as M  # noqa: E402


class TwoRanks:
    """One rank's view of a two-rank NCCL whose peer holds ``peer`` (the same collective on both ranks)."""

    def __init__(self, rank: int, mine: torch.Tensor, peer: torch.Tensor):
        self.rank, self.world, self.calls = rank, 2, []
        self.p = (mine, peer) if rank == 0 else (peer, mine)

    def reduce_scatter(self, send, recv):
        self.calls.append("reduce_scatter")
        full = (self.p[1] + self.p[0]).reshape(-1)          # the peer's add order: the same bits
        recv.copy_(full.view(2, -1)[self.rank])

    def all_gather(self, send, recv):
        self.calls.append(("all_gather", send.dtype))
        if send.dtype == torch.bfloat16:                  # the halves of bf16(p0 + p1), rank order
            halves = (self.p[0] + self.p[1]).to(torch.bfloat16).reshape(-1).view(2, -1)
            recv.copy_(torch.cat([halves[0], halves[1]]))
        else:
            recv.copy_(torch.cat([self.p[0].reshape(-1), self.p[1].reshape(-1)]))


def comm(rank, mine, peer, rdma_bytes=0):
    c = M.Comm.__new__(M.Comm)
    c.nccl, c.world, c.rdma_bytes, c._buf = TwoRanks(rank, mine, peer), 2, rdma_bytes, {}
    return c


def parts(n=8, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((n, d), generator=g) * 3, torch.randn((n, d), generator=g) * 1e-3)


@pytest.fixture
def rs(monkeypatch):
    monkeypatch.setattr(M, "RS", True)


@pytest.mark.parametrize("rank", [0, 1])
def test_bits_are_the_posts_y(rs, rank):
    p0, p1 = parts()
    mine, peer = (p0, p1) if rank == 0 else (p1, p0)
    c = comm(rank, mine, peer)
    y = c.reduced(mine)
    assert y.shape == (1, *p0.shape) and y.dtype == torch.bfloat16
    want = (p0 + p1).to(torch.bfloat16)                   # what _hc_post / _hc_mix_pf form from the gather
    assert torch.equal(y[0].view(torch.int16), want.view(torch.int16))
    assert c.nccl.calls == ["reduce_scatter", ("all_gather", torch.bfloat16)]


def test_the_post_reads_the_same_y():
    """The kernels' y from either input: WORLD fp32 loads summed then cast, or one bf16 load cast (a no-op)."""

    p0, p1 = parts(seed=1)
    from_gather = (p0 + p1).to(torch.bfloat16).to(torch.float32)
    from_reduced = (p0 + p1).to(torch.bfloat16)[None][0].to(torch.bfloat16).to(torch.float32)
    assert torch.equal(from_gather, from_reduced)


@pytest.mark.parametrize("case", ["off", "odd rows", "rdma sized", "one rank"])
def test_falls_back_to_the_gather(monkeypatch, case):
    monkeypatch.setattr(M, "RS", case != "off")
    p0, p1 = parts(n=7 if case == "odd rows" else 8)
    c = comm(0, p0, p1, rdma_bytes=p0.numel() * 4 if case == "rdma sized" else 0)
    if case == "one rank":
        c.world = 1
        assert torch.equal(c.reduced(p0), p0[None])
        return
    g = c.reduced(p0)
    assert g.dtype == torch.float32 and g.shape == (2, *p0.shape)
    assert torch.equal(g[0], p0) and torch.equal(g[1], p1)
    assert c.nccl.calls == [("all_gather", torch.float32)]
