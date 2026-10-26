"""memtrim.trim: off unless TF_TRIM=1, and a no-op where the libc has no malloc_trim."""

import ctypes

from tensorfold.server import memtrim


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(memtrim, "TRIM", False)
    called = []
    monkeypatch.setattr(memtrim, "_LIBC", lambda n: called.append(n))
    memtrim.trim()
    assert called == []


def test_calls_glibc_when_on(monkeypatch):
    called = []
    monkeypatch.setattr(memtrim, "TRIM", True)
    monkeypatch.setattr(memtrim, "_LIBC", lambda n: called.append(n))
    memtrim.trim()
    assert called == [0]


def test_a_libc_without_malloc_trim_is_not_an_error(monkeypatch):
    def missing(_name):
        raise OSError("no libc.so.6")

    monkeypatch.setattr(memtrim, "TRIM", True)
    monkeypatch.setattr(memtrim, "_LIBC", None)
    monkeypatch.setattr(ctypes, "CDLL", missing)
    memtrim.trim()
    assert memtrim._LIBC is False
    memtrim.trim()                                   # and the second call stays quiet
