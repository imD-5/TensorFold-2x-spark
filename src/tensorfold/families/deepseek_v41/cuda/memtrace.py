"""TF_DS_MEMTRACE=<path prefix>: host and allocator memory every 0.2 s with the engine's phase, one CSV a rank.

On GB10 the GPU allocator takes host memory, so a rise of ``reserved`` is a fall of ``MemAvailable``. ``phase(name)``
tags the thread's work; its end writes an ``E`` row with the allocator's peaks since the last one. Off by default.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time

PATH = os.environ.get("TF_DS_MEMTRACE", "")
PERIOD = 0.2
_KEYS = ("MemAvailable", "Cached", "AnonPages", "Shmem", "Mapped", "Dirty", "Unevictable", "SwapFree")
HEADER = "kind,t,phase,dur_s," + ",".join(k.lower() + "_kib" for k in _KEYS) + \
         ",rss_kib,allocated_b,reserved_b,inactive_split_b,peak_allocated_b,peak_reserved_b,cuda_free_b"
_active: dict[int, str] = {}
_lock = threading.Lock()
_file = None
_stop = threading.Event()
_NONE = contextlib.nullcontext()


def meminfo() -> list[int]:
    """The ``_KEYS`` of /proc/meminfo in KiB (-1 where missing)."""

    got: dict[str, int] = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in _KEYS:
                got[key] = int(rest.split()[0])
    return [got.get(k, -1) for k in _KEYS]


def _rss_kib() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024
    except (OSError, ValueError, IndexError):
        return -1


def _torch_stats(reset: bool) -> list[int]:
    """allocated, reserved, inactive split bytes, the peaks of both since the last reset, the driver's free bytes."""

    try:
        import torch

        if not torch.cuda.is_available():
            return [-1] * 6
        s = torch.cuda.memory_stats()
        row = [s.get("allocated_bytes.all.current", -1), s.get("reserved_bytes.all.current", -1),
               s.get("inactive_split_bytes.all.current", -1), s.get("allocated_bytes.all.peak", -1),
               s.get("reserved_bytes.all.peak", -1), torch.cuda.mem_get_info()[0]]
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return row
    except Exception:  # noqa: BLE001  (a trace never stops the engine)
        return [-1] * 6


def _write(kind: str, name: str, dur: float, reset: bool) -> None:
    """One row. A trace never stops the engine: a failed read or write (a full or read-only disk) ends the trace."""

    global _file
    try:
        mem = meminfo()
        t = _torch_stats(reset)
        row = [kind, f"{time.time():.3f}", name, f"{dur:.3f}", *mem, _rss_kib(), *t]
        with _lock:
            if _file is not None:
                _file.write(",".join(map(str, row)) + "\n")
    except Exception as exc:  # noqa: BLE001
        with _lock:
            gone, _file = _file, None
        _stop.set()
        if gone is not None:
            print(f"[tensorfold] TF_DS_MEMTRACE stopped: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            with contextlib.suppress(Exception):
                gone.close()


def _sample() -> None:
    while not _stop.is_set():
        with _lock:
            names = "|".join(sorted(set(_active.values()))) or "idle"
        _write("S", names, 0.0, False)
        _stop.wait(PERIOD)


def start(rank: int, path: str | None = None) -> bool:
    """Open ``<prefix>.rank<N>.csv`` and start the sampler. False when TF_DS_MEMTRACE is empty."""

    global _file
    prefix = PATH if path is None else path
    if not prefix or _file is not None:
        return False
    try:
        os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
        _file = open(f"{prefix}.rank{rank}.csv", "w", buffering=1)
        _file.write(HEADER + "\n")
    except OSError as exc:
        _file = None
        print(f"[tensorfold] TF_DS_MEMTRACE is off: {exc}", flush=True)
        return False
    _stop.clear()
    threading.Thread(target=_sample, daemon=True, name="memtrace").start()
    return True


def stop() -> None:
    """End the sampler and close the file (tests; the engine lets the process end it)."""

    global _file
    _stop.set()
    with _lock:
        if _file is not None:
            _file.close()
        _file = None


@contextlib.contextmanager
def _traced(name: str):
    tid = threading.get_ident()
    with _lock:
        before = _active.get(tid)
        _active[tid] = name
    t = time.perf_counter()
    try:
        yield
    finally:
        with _lock:
            if before is None:
                _active.pop(tid, None)
            else:
                _active[tid] = before
        _write("E", name, time.perf_counter() - t, True)


def phase(name: str):
    """A context manager tagging this thread's work ``name`` in the trace (a no-op while the trace is off)."""

    return _traced(name) if _file is not None else _NONE
