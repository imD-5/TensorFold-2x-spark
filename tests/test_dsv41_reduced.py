"""Comm.reduced (TF_DS_RS): two ranks get bf16(p0 + p1), the y the mHC post kernels form from the fp32 gather, and
the gather itself where the reduce-scatter does not apply."""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.deepseek_v41.cuda import model as M  # noqa: E402


class Link:
    """Two ranks' collectives on the host: each call waits for both ranks' buffers, as NCCL does."""

    def __init__(self):
        self.barrier, self.sends, self.calls = threading.Barrier(2), [None, None], []

    def swap(self, rank, send):
        self.sends[rank] = send.clone()
        self.barrier.wait()
        both = list(self.sends)
        self.barrier.wait()
        return both


class Rank:
    def __init__(self, link, rank, rdma=None):
        self.link, self.rank, self.world = link, rank, 2
        if rdma is not None:
            self.rdma = rdma

    def reduce_scatter(self, send, recv):
        a, b = self.link.swap(self.rank, send)
        self.link.calls.append(("reduce_scatter", self.rank))
        recv.copy_((a + b).view(2, -1)[self.rank])                  # ncclSum of two operands

    def all_gather(self, send, recv):
        a, b = self.link.swap(self.rank, send)
        self.link.calls.append(("all_gather", send.dtype))
        recv.copy_(torch.cat([a, b]))


class Fits:
    def fits(self, x):
        return True


def both_ranks(p0, p1, rdma=None):
    """Comm.reduced on two threads, one a rank: (rank 0's result, rank 1's result, the calls)."""

    link, out = Link(), [None, None]

    def run(rank, mine):
        c = M.Comm.__new__(M.Comm)
        c.nccl, c.world, c._buf = Rank(link, rank, rdma), 2, {}
        out[rank] = c.reduced(mine)

    ts = [threading.Thread(target=run, args=(r, p)) for r, p in ((0, p0), (1, p1))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out[0], out[1], link.calls


def parts(n=8, d=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, d), generator=g) * 3, torch.randn((n, d), generator=g) * 1e-3


def test_both_ranks_get_the_posts_y(monkeypatch):
    monkeypatch.setattr(M, "RS", True)
    p0, p1 = parts()
    y0, y1, calls = both_ranks(p0, p1)
    want = (p0 + p1).to(torch.bfloat16)[None]                    # what _hc_post forms from the gather, before the mix
    for y in (y0, y1):
        assert y.dtype == torch.bfloat16 and torch.equal(y.view(torch.int16), want.view(torch.int16))
    assert sorted(c[0] for c in calls) == ["all_gather", "all_gather", "reduce_scatter", "reduce_scatter"]
    assert all(c[1] == torch.bfloat16 for c in calls if c[0] == "all_gather")


@pytest.mark.parametrize("case", ["off", "odd rows", "rdma holds it"])
def test_falls_back_to_the_gather(monkeypatch, case):
    monkeypatch.setattr(M, "RS", case != "off")
    p0, p1 = parts(n=7 if case == "odd rows" else 8)
    y0, y1, calls = both_ranks(p0, p1, Fits() if case == "rdma holds it" else None)
    for y in (y0, y1):
        assert y.dtype == torch.float32 and torch.equal(y, torch.stack([p0, p1]))
    assert [c[0] for c in calls] == ["all_gather", "all_gather"]


def test_one_rank_is_the_gather(monkeypatch):
    monkeypatch.setattr(M, "RS", True)
    c = M.Comm.__new__(M.Comm)
    c.nccl, c.world, c._buf = None, 1, {}
    p0, _ = parts()
    assert torch.equal(c.reduced(p0), p0[None])
