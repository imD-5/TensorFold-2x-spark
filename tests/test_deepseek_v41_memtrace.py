"""DeepSeek-V4.1's TF_DS_MEMTRACE: off it does nothing; on it writes one CSV a rank with sample rows (S) tagged by the
thread's phase and an end row (E) for each phase."""

import time

from tensorfold.families.deepseek_v41.cuda import memtrace


def _rows(path):
    lines = path.read_text().splitlines()
    return lines[0], [line.split(",") for line in lines[1:]]


def test_off_is_a_no_op():
    assert memtrace._file is None
    with memtrace.phase("fill"):
        pass
    assert memtrace.start(0, "") is False


def test_phase_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(memtrace, "PERIOD", 0.01)
    monkeypatch.setattr(memtrace, "meminfo", lambda: [1000, 2, 3, 4, 5, 6, 7, 8])
    monkeypatch.setattr(memtrace, "_torch_stats", lambda reset: [10, 20, 30, 40, 50, 60])
    assert memtrace.start(1, str(tmp_path / "run")) is True
    try:
        assert memtrace.start(1, str(tmp_path / "other")) is False       # one trace a process
        with memtrace.phase("fill"):
            with memtrace.phase("vit"):
                time.sleep(0.1)
            time.sleep(0.05)
        time.sleep(0.05)
    finally:
        memtrace.stop()
    header, rows = _rows(tmp_path / "run.rank1.csv")
    assert header == memtrace.HEADER
    assert all(len(r) == len(header.split(",")) for r in rows)
    ends = [r for r in rows if r[0] == "E"]
    assert [r[2] for r in ends] == ["vit", "fill"]
    assert float(ends[1][3]) >= float(ends[0][3]) > 0
    assert {r[2] for r in rows if r[0] == "S"} >= {"idle", "vit", "fill"}
    assert rows[0][4] == "1000" and rows[0][-4:] == ["30", "40", "50", "60"]
    assert memtrace.phase("after") is memtrace._NONE                     # off again after stop()


def test_a_failed_write_ends_the_trace_not_the_request(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(memtrace, "PERIOD", 0.01)
    monkeypatch.setattr(memtrace, "meminfo", lambda: [1000, 2, 3, 4, 5, 6, 7, 8])
    monkeypatch.setattr(memtrace, "_torch_stats", lambda reset: [10, 20, 30, 40, 50, 60])
    assert memtrace.start(0, str(tmp_path / "run")) is True

    class Full:
        def write(self, text):
            raise OSError(28, "No space left on device")

        def close(self):
            pass

    try:
        memtrace._file = Full()
        with memtrace.phase("fill"):                         # the request's own work goes on
            pass
        assert memtrace._file is None
        assert "TF_DS_MEMTRACE stopped" in capsys.readouterr().out
    finally:
        memtrace.stop()


def test_start_on_an_unwritable_path_is_off(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert memtrace.start(0, str(blocker / "sub" / "run")) is False
    assert memtrace._file is None
    assert "is off" in capsys.readouterr().out
