"""DeepSeek-V4.1 host-memory guards: the blocked Engram sum, the cache trim and the admission gate."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.memory_gate import NoRoom  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import model, multi  # noqa: E402


class Peer:
    """Rank 0 of two: the other rank's copy is 3x this one's."""

    def __init__(self):
        self.calls = 0

    def all_gather(self, send, recv):
        self.calls += 1
        recv.view(2, -1)[0].copy_(send)
        recv.view(2, -1)[1].copy_(send * 3)


@pytest.mark.parametrize("n,block", [(1300, 512), (1024, 512), (7, 512), (2048, 0), (2049, 700)])
def test_sum_blocks_equals_the_whole_sum(n, block):
    x = torch.randn(n, 37)
    whole = model.Comm(Peer(), 2).sum(x).to(torch.bfloat16)
    peer = Peer()
    got = model.Comm(peer, 2).sum_blocks(x, block, torch.bfloat16)
    assert got.dtype == torch.bfloat16 and torch.equal(got, whole)
    assert peer.calls == (1 if block <= 0 or n <= block else -(-n // block))


def test_sum_blocks_on_one_rank_is_the_input_cast():
    x = torch.randn(10, 4)
    assert torch.equal(model.Comm(Peer(), 1).sum_blocks(x, 4, torch.bfloat16), x.to(torch.bfloat16))


def _decoder(live, room):
    return SimpleNamespace(live=lambda: live, _host_room=lambda: room, gate_log=0.0)


GIB = 2**30


def test_gate_waits_only_with_live_streams_and_a_short_host(monkeypatch, capsys):
    monkeypatch.setenv("TF_DS_MEM_FLOOR_GIB", "1.5")
    monkeypatch.setattr(multi, "ADMIT_GIB", 1.0)
    multi.MultiDecoder._gate(_decoder(2, 3 * GIB), None)                     # room: goes in
    multi.MultiDecoder._gate(_decoder(0, 1 * GIB), None)                     # nothing live: waiting would not help
    multi.MultiDecoder._gate(_decoder(2, -1), None)                          # unknown: goes in
    short = _decoder(2, 2 * GIB)
    with pytest.raises(NoRoom):
        multi.MultiDecoder._gate(short, None)
    assert "admission waits" in capsys.readouterr().out
    with pytest.raises(NoRoom):                                              # (no second line within 10 s)
        multi.MultiDecoder._gate(short, None)
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(multi, "ADMIT_GIB", 0.0)
    multi.MultiDecoder._gate(short, None)                                    # off


def test_trim_is_off_below_its_threshold(monkeypatch):
    freed, slack = [], []
    trimmer = SimpleNamespace(_slack=lambda: slack.append(1))
    monkeypatch.setattr(multi.torch.cuda, "empty_cache", lambda: freed.append(1))
    monkeypatch.setattr(multi.torch.cuda, "memory_reserved", lambda *a: 1000 * 2**20)
    monkeypatch.setattr(multi.torch.cuda, "memory_allocated", lambda *a: 800 * 2**20)
    monkeypatch.setattr(multi, "TRIM", 256 * 2**20)
    multi.MultiDecoder._trim(trimmer)
    assert freed == [] and slack == []
    monkeypatch.setattr(multi.torch.cuda, "memory_allocated", lambda *a: 700 * 2**20)
    multi.MultiDecoder._trim(trimmer)
    assert freed == [1] and slack == [1]                                     # the baseline is read after a release
    monkeypatch.setattr(multi, "TRIM", -1)
    multi.MultiDecoder._trim(trimmer)
    assert freed == [1] and slack == [1]


def _torch_gap(monkeypatch, gap_mib):
    monkeypatch.setattr(multi.torch.cuda, "memory_reserved", lambda *a: (1000 + gap_mib) * 2**20)
    monkeypatch.setattr(multi.torch.cuda, "memory_allocated", lambda *a: 1000 * 2**20)


def test_slack_keeps_the_smallest_gap_seen(monkeypatch):
    d = SimpleNamespace(slack0=None)
    for gap, want in ((600, 600), (300, 300), (400, 300)):
        _torch_gap(monkeypatch, gap)
        multi.MultiDecoder._slack(d)
        assert d.slack0 == want * 2**20


def test_host_room_counts_only_the_cached_bytes_a_trim_can_return(monkeypatch):
    monkeypatch.setattr(multi, "_mem_available", lambda: 2339 * 2**20)
    d = SimpleNamespace(slack0=300 * 2**20)
    _torch_gap(monkeypatch, 800)
    assert multi.MultiDecoder._host_room(d) == (2339 + 500) * 2**20          # 500 MiB can go back
    _torch_gap(monkeypatch, 300)
    assert multi.MultiDecoder._host_room(d) == 2339 * 2**20                  # the slack is not room
    _torch_gap(monkeypatch, 100)
    assert multi.MultiDecoder._host_room(d) == 2339 * 2**20                  # (never below MemAvailable)
    d.slack0 = None                                                          # no baseline (no warm-up, no trim)
    _torch_gap(monkeypatch, 800)
    assert multi.MultiDecoder._host_room(d) == 2339 * 2**20                  # counts none of the cache
    monkeypatch.setattr(multi, "_mem_available", lambda: -1)
    assert multi.MultiDecoder._host_room(d) == -1


def test_the_gate_holds_a_head_that_is_under_the_threshold_only_by_the_slack(monkeypatch):
    """Run A of the 1.1.0 gate: MemAvailable 2,339 MiB, floor 1.5 + 1.0 GiB, about 300 MiB of slack."""

    monkeypatch.setenv("TF_DS_MEM_FLOOR_GIB", "1.5")
    monkeypatch.setattr(multi, "ADMIT_GIB", 1.0)
    monkeypatch.setattr(multi, "_mem_available", lambda: 2339 * 2**20)
    _torch_gap(monkeypatch, 300)
    d = SimpleNamespace(live=lambda: 1, slack0=300 * 2**20, gate_log=0.0)
    d._host_room = lambda: multi.MultiDecoder._host_room(d)
    with pytest.raises(NoRoom):
        multi.MultiDecoder._gate(d, None)
    d.slack0 = 0                                                             # the old reading let it through
    multi.MultiDecoder._gate(d, None)
    d.slack0 = 300 * 2**20
    monkeypatch.setattr(multi, "_mem_available", lambda: 3300 * 2**20)      # a fresh start
    multi.MultiDecoder._gate(d, None)
