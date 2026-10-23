"""shutdown.leave and Scheduler.close(timeout): a busy worker does not hold the process."""

import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.cuda import shutdown
from tensorfold.cuda.scheduler import Scheduler


class Busy:
    """A decoder whose round does not return until ``release`` is set (a long prompt chunk)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()

    def live(self):
        return 0 if self.release.is_set() else 1

    def round(self):
        self.entered.set()
        self.release.wait(30)
        return []

    def finish(self, done):
        pass

    def drop(self):
        return []


def test_close_with_a_timeout_reports_a_busy_worker():
    decoder = Busy()
    scheduler = Scheduler(decoder)
    assert decoder.entered.wait(5)
    started = time.monotonic()
    assert scheduler.close(timeout=0.2) is False
    assert time.monotonic() - started < 5
    assert scheduler.thread.is_alive()
    assert scheduler.decoder is decoder                  # a worker still running keeps its decoder
    decoder.release.set()
    assert scheduler.close(timeout=5) is True
    assert not scheduler.thread.is_alive()
    assert scheduler.decoder is None


class Streaming(Busy):
    """A stream that needs two rounds; the first does not return until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.rounds = 0

    def live(self):
        return 1 if self.rounds < 2 else 0

    def round(self):
        self.rounds += 1
        if self.rounds == 1:
            self.entered.set()
            self.release.wait(30)
        return []


def test_close_while_a_stream_is_live_lets_the_worker_finish_it_and_end_without_an_error(monkeypatch):
    errors = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    decoder = Streaming()
    scheduler = Scheduler(decoder)
    assert decoder.entered.wait(5)
    assert scheduler.close(timeout=0.1) is False         # (close's mark waits in the queue behind the busy round)
    decoder.release.set()
    scheduler.thread.join(5)
    assert not scheduler.thread.is_alive()
    assert decoder.rounds == 2                           # the live stream ran to its end
    assert errors == []


def test_close_without_drain_ends_the_worker_after_its_current_round(monkeypatch):
    errors = []
    monkeypatch.setattr(threading, "excepthook", errors.append)
    decoder = Streaming()
    scheduler = Scheduler(decoder)
    assert decoder.entered.wait(5)
    assert scheduler.close(timeout=0.1, drain=False) is False
    decoder.release.set()
    scheduler.thread.join(5)
    assert not scheduler.thread.is_alive()
    assert decoder.rounds == 1                           # the stream was left, not finished
    assert errors == []


def test_leave_ends_the_process_with_the_code_when_the_worker_is_stuck(monkeypatch, capsys):
    ended = []

    def fake_exit(code):
        ended.append(code)

    monkeypatch.setattr(shutdown, "_exit", fake_exit)
    monkeypatch.setenv("TF_SHUTDOWN_GRACE_S", "0.2")
    decoder = Busy()
    scheduler = Scheduler(decoder)
    assert decoder.entered.wait(5)
    app = SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler))
    print("a log line")
    with pytest.raises(SystemExit):                      # (only a replaced _exit returns)
        shutdown.leave(app, 3)
    assert ended == [3]
    assert scheduler.thread.is_alive()                   # it did not wait for the busy worker
    decoder.release.set()
    scheduler.close(timeout=5)


def test_leave_ends_the_process_when_the_wait_is_interrupted(monkeypatch):
    ended = []
    monkeypatch.setattr(shutdown, "_exit", ended.append)

    class Interrupted:
        def close(self, timeout=None, *, drain=True):
            raise KeyboardInterrupt                      # a second Ctrl-C that got through

    with pytest.raises(SystemExit):
        shutdown.leave(SimpleNamespace(engine=SimpleNamespace(scheduler=Interrupted())), 5)
    assert ended == [5]


def test_leave_stops_an_idle_worker_and_an_app_without_one(monkeypatch):
    ended = []
    monkeypatch.setattr(shutdown, "_exit", ended.append)

    class Idle(Busy):
        def live(self):
            return 0

    decoder = Idle()
    scheduler = Scheduler(decoder)
    with pytest.raises(SystemExit):
        shutdown.leave(SimpleNamespace(engine=SimpleNamespace(scheduler=scheduler)))
    assert not scheduler.thread.is_alive()
    with pytest.raises(SystemExit):
        shutdown.leave(SimpleNamespace())
    assert ended == [0, 0]


@pytest.mark.parametrize("value, seconds", [("0.5", 0.5), ("", shutdown.GRACE_S), ("soon", shutdown.GRACE_S),
                                            ("-4", 0.0)])
def test_grace_comes_from_the_environment(monkeypatch, value, seconds):
    monkeypatch.setenv("TF_SHUTDOWN_GRACE_S", value)
    assert shutdown.grace() == seconds
