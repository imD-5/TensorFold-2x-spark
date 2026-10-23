"""DeepSeek-V4.1 --parallel faults: a stream whose own host work raises (its grammar, its samples, its reply's
callback) ends alone with the error and the lane serves on; a device fault (an out-of-memory, CUDA or NCCL error, also
one that first shows in a sample) and any other failed step break the lane, and a broken lane ends its process so
that a supervisor restarts the ranks (one rank has no watchdog: a device fault ends its process too, a host error
does not); a follower that has no live stream, or another draft depth, still takes part in rank 0's round check.
The real MultiDecoder admission, fill and round paths on CPU stand-ins (no GPU, kernels or
checkpoint)."""

from __future__ import annotations

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

VOCAB = 32
CHUNK = 2048


@pytest.fixture
def multi(monkeypatch):
    """multi.py with the CUDA model and engine modules stubbed, kept prompts off."""

    monkeypatch.setenv("TF_DS_KEEP", "0")
    monkeypatch.setenv("TF_DS_EXIT_ON_BREAK", "0")      # (a test that wants the exit turns it on and patches os._exit)
    model = ModuleType("tensorfold.families.deepseek_v41.cuda.model")
    model.RAW = 4
    engine = ModuleType("tensorfold.families.deepseek_v41.cuda.engine")
    engine.PREFILL_CHUNK, engine.REPLAY = CHUNK, False
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    monkeypatch.delitem(sys.modules, "tensorfold.families.deepseek_v41.cuda.multi", raising=False)
    import tensorfold.families.deepseek_v41.cuda.multi as m

    return m


class Watch:
    """The watchdog's face to the decoder: ``abort`` records why the lane broke."""

    def __init__(self):
        self.broken, self.busy, self.aborts = None, None, []

    def abort(self, why):
        self.aborts.append(why)
        self.broken = why


class Runner:
    """The round's forward: every window's rows of zero logits (``fail`` raises instead); records the windows."""

    def __init__(self):
        self.calls, self.fail = [], None

    def forward(self, windows):
        self.calls.append([list(w[4]) for w in windows])
        if self.fail is not None:
            raise self.fail
        return torch.zeros((sum(len(w[4]) for w in windows), VOCAB)), None


def sampler(bad_seed=None, exc=None):
    """Keyed sampling's stand-in: token 7 for every row; a stream with ``bad_seed`` raises ``exc``."""

    def sample(logits, positions, sampling):
        if sampling is not None and bad_seed is not None and sampling.seed == bad_seed:
            raise exc
        return [7] * len(positions)

    return sample


def decoder(multi, sample):
    def pool_view(p, slot, base, size):
        return SimpleNamespace(base=base, size=size, slot=slot, length=0, host=multi.HostIds())

    def prefill_steps(sc, dc, prompt, image=None, start=0, snap=None):
        sc.length = len(prompt)
        sc.host.set(0, list(prompt))
        return torch.zeros((1, VOCAB))                    # the prompt's last row
        yield                                             # (a generator: one chunk a step)

    md = object.__new__(multi.MultiDecoder)
    md.e = SimpleNamespace(rank=0, world=1, drafter=None, eos=(1,), replay_mode=False, prefill_steps=prefill_steps,
                           _sample=sample)
    md.pool, md.dpool, md.cap = None, None, 65536
    md.m = SimpleNamespace(pool_view=pool_view, comm=None)
    md.slots = [multi.Slot(i, None) for i in range(4)]
    md.extents, md.free, md.max_rows, md.depth_most, md.eos = multi.Extents(md.cap), list(range(4)), 16, 0, (1,)
    md.streams, md.filling, md.chunk_s, md.since_fill, md.next_id = {}, [], 0.0, 0.0, 0
    md.link, md.round_end, md.rounds, md.round_log = None, None, 0, []
    md.kept, md.next_kept, md.ticks, md.keep_stats, md.spill = {}, 0, 0, {}, None
    md.runner, md.watch = Runner(), Watch()
    return md


def stream(multi, seed, prompt=(5, 6, 9), count=4):
    from tensorfold.engine.exact_sampling import Sampling

    s = multi.Stream(list(prompt), count, Sampling(seed, 1.0, 0, 1.0, 0.0), draft=False, stop_eos=False)
    s.got = []
    s.emit = lambda new: s.got.extend(new)
    return s


def admitted(multi, md, *streams):
    """Admit and fill every stream (each fill gives its first token)."""

    for s in streams:
        md.admit(s)
        while s in md.filling:
            md._fill(s)
    return streams


def test_a_failed_sample_ends_only_its_stream(multi):
    md = decoder(multi, sampler())
    a, b = admitted(multi, md, stream(multi, 13), stream(multi, 14))
    md.e._sample = sampler(13, RuntimeError("Inplace update to inference tensor outside InferenceMode"))
    done = md.round()
    assert done == [a] and a.done and isinstance(a.error, RuntimeError) and a.got == [7]
    assert not b.done and b.error is None and b.got == [7, 7]          # the other stream got its token
    assert md.watch.aborts == []                                        # the lane is not broken
    md.finish(done)
    assert md.round() == [] and b.got == [7, 7, 7]                      # and it serves on


def test_a_failed_reply_callback_ends_only_its_stream(multi):
    md = decoder(multi, sampler())
    a, b = admitted(multi, md, stream(multi, 13), stream(multi, 14))

    def gone(new):
        raise BrokenPipeError("the client has gone")

    a.emit = gone
    done = md.round()
    assert done == [a] and isinstance(a.error, BrokenPipeError)
    assert b.got == [7, 7] and md.watch.aborts == []


def test_a_failed_first_token_ends_only_its_stream(multi):
    md = decoder(multi, sampler(13, ValueError("bad sampling row")))
    a, b = stream(multi, 13), stream(multi, 14)
    md.admit(a)
    assert md._fill(a) == [a] and isinstance(a.error, ValueError) and a.got == []
    admitted(multi, md, b)
    assert b.got == [7] and md.watch.aborts == []
    md.finish([a])
    assert md.round() == [] and b.got == [7, 7]


class BadMask:
    """A grammar whose draft window works but whose mask raises an error that is not a GrammarError (the 2026-10-17
    inference-tensor bug)."""

    spec, think_end, finished = None, None, False

    def window(self, tokens, parents):
        return SimpleNamespace(tokens=list(tokens), rows=[0])

    def mask(self, logits, window=None, offset=0):
        raise RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed")


class BadWindow(BadMask):
    def window(self, tokens, parents):
        raise RuntimeError("the grammar state is corrupt")


@pytest.mark.parametrize("grammar", [BadMask, BadWindow])
def test_a_failed_grammar_ends_only_its_stream(multi, grammar):
    md = decoder(multi, sampler())
    a, b = admitted(multi, md, stream(multi, 13), stream(multi, 14))
    a.constraint = grammar()
    done = md.round()
    assert done == [a] and isinstance(a.error, RuntimeError) and a.got == [7]   # no tokens from the failed round
    assert len(md.runner.calls[-1]) == 2                    # its rows still went through the forward (ranks in step)
    assert b.got == [7, 7] and md.watch.aborts == []


def test_a_failed_forward_still_breaks_the_lane(multi):
    md = decoder(multi, sampler())
    admitted(multi, md, stream(multi, 13), stream(multi, 14))
    md.runner.fail = RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md.round()
    assert len(md.watch.aborts) == 1 and "a step failed on rank 0" in md.watch.aborts[0]


class DeviceMask(BadMask):
    """A grammar whose mask hits an asynchronous CUDA error (its GPU ops are the first sync after the forward)."""

    def mask(self, logits, window=None, offset=0):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


class DeviceWindow(BadMask):
    def window(self, tokens, parents):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")


DEVICE_FAULTS = [
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("an illegal memory access was encountered"),
    RuntimeError("CUBLAS_STATUS_EXECUTION_FAILED when calling cublasGemmEx"),
    RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"),
    RuntimeError("NCCL error: unhandled cuda error"),
    RuntimeError("device-side assert triggered"),
    torch.cuda.OutOfMemoryError("Tried to allocate 2.00 GiB"),
]


@pytest.mark.parametrize("exc", DEVICE_FAULTS, ids=lambda e: type(e).__name__ + ":" + str(e)[:24])
def test_a_device_fault_in_a_sample_breaks_the_lane(multi, exc):
    md = decoder(multi, sampler())
    admitted(multi, md, stream(multi, 13), stream(multi, 14))
    md.e._sample = sampler(13, exc)
    with pytest.raises(type(exc)):
        md.round()
    assert len(md.watch.aborts) == 1 and "a step failed on rank 0" in md.watch.aborts[0]


def test_after_a_device_fault_every_live_stream_gets_the_error_from_drop(multi):
    md = decoder(multi, sampler())
    a, b = admitted(multi, md, stream(multi, 13), stream(multi, 14))
    md.e._sample = sampler(13, RuntimeError("CUDA error: an illegal memory access was encountered"))
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md.round()
    assert len(md.watch.aborts) == 1 and a.error is None and not b.done     # nobody ends alone: the lane is broken
    assert md.drop() == [a, b]                                              # and every live request gets the error


@pytest.mark.parametrize("grammar", [DeviceMask, DeviceWindow])
def test_a_device_fault_in_a_grammar_breaks_the_lane(multi, grammar):
    md = decoder(multi, sampler())
    a, _ = admitted(multi, md, stream(multi, 13), stream(multi, 14))
    a.constraint = grammar()
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md.round()
    assert len(md.watch.aborts) == 1


def test_a_device_fault_in_the_first_token_breaks_the_lane(multi):
    md = decoder(multi, sampler(13, RuntimeError("CUDA error: an illegal memory access was encountered")))
    a = stream(multi, 13)
    md.admit(a)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md._fill(a)
    assert len(md.watch.aborts) == 1
    assert md.drop() == [a]                                 # the stream is not lost: its request gets the error


def test_an_accelerator_error_breaks_the_lane(multi, monkeypatch):
    class Accelerator(RuntimeError):
        pass

    monkeypatch.setattr(torch, "AcceleratorError", Accelerator, raising=False)
    md = decoder(multi, sampler())
    admitted(multi, md, stream(multi, 13), stream(multi, 14))
    md.e._sample = sampler(13, Accelerator("a kernel failed"))
    with pytest.raises(Accelerator):
        md.round()
    assert len(md.watch.aborts) == 1


def test_a_failed_synchronize_makes_a_host_looking_fault_a_device_fault(multi, monkeypatch):
    """The fault's message names no CUDA call, but the device does not sync: the lane is poisoned."""

    def fails():
        raise RuntimeError("an earlier kernel failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", fails)
    md = decoder(multi, sampler())
    admitted(multi, md, stream(multi, 13), stream(multi, 14))
    md.e._sample = sampler(13, RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed"))
    with pytest.raises(RuntimeError, match="inference tensor"):
        md.round()
    assert len(md.watch.aborts) == 1


def test_a_host_fault_stays_with_its_stream_when_the_device_syncs(multi, monkeypatch):
    synced = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: synced.append(1))
    md = decoder(multi, sampler())
    a, b = admitted(multi, md, stream(multi, 13), stream(multi, 14))
    md.e._sample = sampler(13, RuntimeError("Inplace update to inference tensor outside InferenceMode is not allowed"))
    assert md.round() == [a] and isinstance(a.error, RuntimeError) and b.got == [7, 7]
    assert synced and md.watch.aborts == []


class Ranks:
    """The model's all-gather on two ranks: this rank's row, and rank 0's (another row, unless ``same``)."""

    def __init__(self, same=False):
        self.calls, self.same = 0, same

    def gather(self, mine):
        self.calls += 1
        return torch.stack([mine if self.same else mine + 1.0, mine])


def ranked(monkeypatch, md, rank, same=False):
    """Make ``md`` a rank of two; its digest check gathers through ``Ranks`` (on CPU its tensors take no device)."""

    real = torch.tensor

    def tensor(data, **kwargs):
        kwargs.pop("device", None)
        return real(data, **kwargs)

    monkeypatch.setattr(torch, "tensor", tensor)
    md.e.rank, md.e.world = rank, 2
    md.m.comm = Ranks(same)
    return md.m.comm


def test_a_follower_without_a_live_stream_takes_part_in_the_round_check(multi, monkeypatch):
    md = decoder(multi, sampler())
    (a,) = admitted(multi, md, stream(multi, 13))
    ranks = ranked(monkeypatch, md, 1)
    md._stream_failed(a, ValueError("bad sampling row"))        # it ended on this rank alone: nothing is live here
    with pytest.raises(multi.OutOfStep):
        md.round(told=["round", 0])
    assert ranks.calls == 1 and md.runner.calls == [] and md.watch.aborts == []    # rank 0 got its gather: no hang


@pytest.fixture
def depths(multi, monkeypatch):
    """Rank 0's draft depths by live streams (the "conf" policy's default table), up to five drafts."""

    monkeypatch.setattr(multi, "DEPTH_POLICY", "conf")
    monkeypatch.setattr(multi, "DEPTH_BY", [5, 5, 3, 3])


@pytest.mark.usefixtures("depths")
def test_a_follower_with_another_draft_depth_takes_part_in_the_round_check(multi, monkeypatch):
    md = decoder(multi, sampler())
    md.depth_most = 5
    c = admitted(multi, md, stream(multi, 13), stream(multi, 14), stream(multi, 15))[2]
    ranks = ranked(monkeypatch, md, 1)
    md._stream_failed(c, ValueError("bad sampling row"))        # rank 0 has three live streams (depth 3), this rank two
    with pytest.raises(multi.OutOfStep):
        md.round(told=["round", 3])
    assert ranks.calls == 1 and md.runner.calls == [] and md.watch.aborts == []


@pytest.mark.usefixtures("depths")
def test_a_follower_check_that_passes_still_fails_the_step(multi, monkeypatch):
    md = decoder(multi, sampler())
    md.depth_most = 5
    admitted(multi, md, stream(multi, 13), stream(multi, 14))
    ranks = ranked(monkeypatch, md, 1, same=True)
    with pytest.raises(multi.OutOfStep, match="different live streams"):
        md.round(told=["round", 3])                             # (equal shapes with other depths: no step runs on it)
    assert ranks.calls == 1 and md.runner.calls == []


class Sent(Ranks):
    """``Ranks`` that keeps every row this rank sent."""

    def __init__(self):
        super().__init__(same=True)
        self.rows = []

    def gather(self, mine):
        self.rows.append(mine.tolist())
        return super().gather(mine)


@pytest.mark.usefixtures("depths")
def test_a_follower_with_only_another_depth_sends_a_digest_rank_0_cannot_match(multi, monkeypatch):
    md = decoder(multi, sampler())
    md.depth_most = 5
    admitted(multi, md, stream(multi, 13), stream(multi, 14))  # same streams as rank 0; this rank's depth is 5
    ranked(monkeypatch, md, 1)
    md.m.comm = sent = Sent()
    md._agree("round", [md._shape(), 3])                         # what rank 0 sends for its round at depth 3
    with pytest.raises(multi.OutOfStep):
        md.round(told=["round", 3])
    assert len(sent.rows) == 2 and sent.rows[1] != sent.rows[0]   # so rank 0's check fails too: no hang


@pytest.mark.usefixtures("depths")
def test_a_follower_with_rank_0s_depth_runs_the_round_after_one_check(multi, monkeypatch):
    md = decoder(multi, sampler())
    md.depth_most = 5
    b = admitted(multi, md, stream(multi, 13), stream(multi, 14))[1]
    ranks = ranked(monkeypatch, md, 1, same=True)
    assert md.round(told=["round", 5]) == []
    assert ranks.calls == 1 and len(md.runner.calls) == 1 and b.got == [7, 7]


def test_rank_0_without_a_live_stream_sends_no_round_and_waits_for_nobody(multi, monkeypatch):
    md = decoder(multi, sampler())
    ranks = ranked(monkeypatch, md, 0)
    assert md.round() == [] and ranks.calls == 0


@pytest.mark.usefixtures("depths")
def test_a_follower_drops_its_streams_when_the_round_check_fails(multi, monkeypatch):
    md = decoder(multi, sampler())
    md.depth_most = 5
    c = admitted(multi, md, stream(multi, 13), stream(multi, 14), stream(multi, 15))[2]
    ranks = ranked(monkeypatch, md, 1)
    md._stream_failed(c, ValueError("bad sampling row"))
    ops = iter([["round", 3]])
    with pytest.raises(RuntimeError, match="step link closed"):     # (the loop runs on until the link closes)
        md.follow(SimpleNamespace(receive=lambda: next(ops, None)))
    assert ranks.calls == 1 and md.streams == {} and md.free == [0, 1, 2, 3] and md.watch.aborts == []


def watchdog(multi):
    wd = object.__new__(multi.Watchdog)
    wd.comm, wd.rank, wd.broken, wd.lock, wd.socks = SimpleNamespace(), 0, None, threading.Lock(), []
    return wd


def test_a_broken_lane_ends_its_process_after_the_grace(multi, monkeypatch):
    codes, gone = [], threading.Event()

    def exit_(code):
        codes.append(code)
        gone.set()

    monkeypatch.setattr(multi, "EXIT_ON_BREAK", True)
    monkeypatch.setattr(multi, "EXIT_GRACE", 0.05)
    monkeypatch.setattr(multi.os, "_exit", exit_)
    wd = watchdog(multi)
    wd.abort("rank 1 stopped answering")
    wd.abort("a second reason")                             # one break, one exit
    assert gone.wait(5.0) and codes == [multi.EXIT_CODE] == [75]
    assert wd.broken == "rank 1 stopped answering"


def test_exit_on_break_off_keeps_the_process(multi, monkeypatch):
    codes = []
    monkeypatch.setattr(multi, "EXIT_ON_BREAK", False)
    monkeypatch.setattr(multi, "EXIT_GRACE", 0.0)
    monkeypatch.setattr(multi.os, "_exit", codes.append)
    wd = watchdog(multi)
    wd.abort("NCCL error 3")
    threading.Event().wait(0.2)
    assert codes == [] and wd.broken == "NCCL error 3"


@pytest.fixture
def exits(multi, monkeypatch):
    """The process exit, patched (a 0.05 s grace): the exit codes, and an event that each exit sets. A timer that
    fires after the patch is gone would end pytest, so a test waits on the event or for longer than the grace."""

    codes, gone = [], threading.Event()

    def exit_(code):
        codes.append(code)
        gone.set()

    monkeypatch.setattr(multi, "EXIT_ON_BREAK", True)
    monkeypatch.setattr(multi, "EXIT_GRACE", 0.05)
    monkeypatch.setattr(multi.os, "_exit", exit_)
    return SimpleNamespace(codes=codes, gone=gone)


def single_rank(multi, sample=None):
    """A one-rank decoder: no watchdog, two live streams."""

    md = decoder(multi, sampler() if sample is None else sample)
    md.watch = None
    return md, admitted(multi, md, stream(multi, 13), stream(multi, 14))


def test_a_device_fault_of_a_forward_without_a_watchdog_ends_the_process_once(multi, exits, capsys):
    md, _ = single_rank(multi)
    md.runner.fail = RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md.round()
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md.round()                                          # a second fault: no second timer
    assert exits.gone.wait(5.0)
    threading.Event().wait(0.3)                             # (a second exit would show by now)
    assert exits.codes == [multi.EXIT_CODE] == [75]
    out = capsys.readouterr().out
    assert "a device fault poisons this lane" in out and "the process exits" in out


@pytest.mark.parametrize("exc", DEVICE_FAULTS, ids=lambda e: type(e).__name__ + ":" + str(e)[:24])
def test_a_device_fault_in_a_sample_without_a_watchdog_ends_the_process(multi, exits, exc):
    md, _ = single_rank(multi)
    md.e._sample = sampler(13, exc)
    with pytest.raises(type(exc)):
        md.round()
    assert exits.gone.wait(5.0)
    assert exits.codes == [75]


def test_a_device_fault_in_the_first_token_without_a_watchdog_ends_the_process(multi, exits):
    md = decoder(multi, sampler(13, RuntimeError("CUDA error: an illegal memory access was encountered")))
    md.watch = None
    a = stream(multi, 13)
    md.admit(a)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md._fill(a)
    assert exits.gone.wait(5.0)
    assert exits.codes == [75]


def test_a_host_error_of_a_step_without_a_watchdog_keeps_the_process(multi, exits):
    md, _ = single_rank(multi)
    md.runner.fail = RuntimeError("the forward's host code has a bug")
    with pytest.raises(RuntimeError, match="host code"):
        md.round()
    assert not exits.gone.wait(0.3) and exits.codes == []   # (no collective to pair wrongly: the lane serves on)
    md.runner.fail = None
    assert len(md.drop()) == 2                              # (the scheduler fails the live requests)
    (c,) = admitted(multi, md, stream(multi, 15))
    assert md.round() == [] and c.got == [7, 7]


def test_a_device_fault_without_a_watchdog_keeps_the_process_with_exit_on_break_off(multi, exits, monkeypatch):
    monkeypatch.setattr(multi, "EXIT_ON_BREAK", False)
    monkeypatch.setattr(multi, "EXIT_GRACE", 0.0)
    md, _ = single_rank(multi)
    md.runner.fail = RuntimeError("CUDA error: an illegal memory access was encountered")
    with pytest.raises(RuntimeError, match="illegal memory access"):
        md.round()
    assert not exits.gone.wait(0.3) and exits.codes == []


class OneToken:
    """A grammar that takes one token, then its stop token: finished after the first token it follows. The stand-in
    logits are all zero, so the mask has nothing to show here."""

    spec, think_end = SimpleNamespace(kind="regex", text="x"), None

    def __init__(self):
        self.finished = False

    def window(self, tokens, parents):
        return SimpleNamespace(tokens=list(tokens), rows=[])

    def mask(self, logits, window=None, offset=0):
        return logits

    def advance(self, tokens):
        self.finished = self.finished or bool(tokens)


def test_a_reply_ends_at_its_grammar_stop_in_a_round_with_ignore_eos(multi):
    md = decoder(multi, sampler())
    a, b = admitted(multi, md, stream(multi, 13, count=10), stream(multi, 14, count=10))
    a.constraint = OneToken()                               # stop_eos is off: only the grammar ends the reply
    done = md.round()
    assert done == [a] and a.done and a.error is None and a.got == [7, 7]
    assert not b.done and b.got == [7, 7]


def test_a_reply_ends_at_its_grammar_stop_on_its_first_token_with_ignore_eos(multi):
    md = decoder(multi, sampler())
    a = stream(multi, 13, count=10)
    a.constraint = OneToken()
    md.admit(a)
    assert md._fill(a) == [a] and a.done and a.error is None and a.got == [7]
