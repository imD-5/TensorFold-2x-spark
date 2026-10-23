"""SIGTERM ends the CUDA server without interpreter finalization, so a worker busy in a torch call cannot abort it.

Before this change ``serve`` returned, the interpreter finalized, CPython killed the busy daemon worker at its next GIL
re-acquire, and the C++ runtime printed "terminate called without an active exception" (earlyoom sends SIGTERM).
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

CHILD = '''
import atexit, sys, time
from types import SimpleNamespace
from tensorfold.cuda import http
from tensorfold.cuda.scheduler import Scheduler

marker, work = sys.argv[1], sys.argv[2]
atexit.register(lambda: open(marker, "w").write("finalized"))      # runs only if the interpreter finalizes
if work == "torch":
    import torch
    a = torch.randn(2500, 2500)


class Busy:
    def live(self):
        return 1

    def round(self):
        if work == "torch":
            a @ a
            print("round", flush=True)
            while True:
                a @ a                                   # a torch op that releases the GIL, over and over
        print("round", flush=True)
        time.sleep(60)
        return []

    def finish(self, done):
        pass

    def drop(self):
        return []


serve_forever = http.Server.serve_forever


def ready(self, *args, **kwargs):
    print("ready", flush=True)
    return serve_forever(self, *args, **kwargs)


http.Server.serve_forever = ready
app = SimpleNamespace(engine=SimpleNamespace(scheduler=Scheduler(Busy())), auth=None)
http.serve(app, "127.0.0.1", 0)
'''


def run_child(tmp_path, work, *, grace="0.5", second=None):
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    marker = tmp_path / "finalized"
    env = dict(os.environ, PYTHONPATH=str(SRC), TF_SHUTDOWN_GRACE_S=grace, PYTHONUNBUFFERED="1")
    child = subprocess.Popen([sys.executable, str(script), str(marker), work], env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    seen = set()
    try:
        deadline = time.monotonic() + 60
        while seen != {"ready", "round"} and time.monotonic() < deadline:
            line = child.stdout.readline()
            if not line:
                break
            seen.add(line.strip())
        assert seen == {"ready", "round"}, child.stderr.read()
        time.sleep(1.0)                                  # the worker is inside its round
        started = time.monotonic()
        child.send_signal(signal.SIGTERM)
        if second is not None:
            time.sleep(0.3)                              # the worker still holds the exit for its grace time
            child.send_signal(second)
        child.wait(timeout=30)
        took = time.monotonic() - started
    finally:
        if child.poll() is None:
            child.kill()
    return child.returncode, child.stderr.read(), marker.exists(), took


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
def test_sigterm_ends_the_server_process_without_finalizing(tmp_path):
    code, err, finalized, took = run_child(tmp_path, "sleep")
    assert code == 0, err
    assert not finalized                                 # os._exit: no atexit, no interpreter finalization
    assert took < 10


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
@pytest.mark.parametrize("second", [signal.SIGINT, signal.SIGTERM])
def test_a_second_signal_during_the_exit_does_not_cut_it_short(tmp_path, second):
    code, err, finalized, took = run_child(tmp_path, "sleep", grace="2", second=second)
    assert code == 0, err
    assert not finalized
    assert "Traceback" not in err


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
def test_sigterm_with_a_worker_in_a_torch_op_does_not_abort(tmp_path):
    pytest.importorskip("torch")
    code, err, finalized, took = run_child(tmp_path, "torch")
    assert "terminate called" not in err
    assert code == 0, err
    assert not finalized
