#!/usr/bin/env python3
"""Replay-by-replay summary of an nsys cuda_gpu_trace CSV (``nsys stats -r cuda_gpu_trace -f csv``) of tp4_trace.py:
wall, GPU busy (the union of kernel intervals over every stream), idle gaps, kernel count, the collect kernels' time
(RDMA gather waits included) and the top kernels by time a replay (medians over replays).

    trace_stats.py trace.csv [--gap-ms 2.0]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict


def rows_of(path: str):
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith(("NOTICE", "Processing", "Generating", "\x1b"))]
    rd = csv.DictReader(lines)
    out = []
    for r in rd:
        try:
            start = int(r.get("Start (ns)") or r["Start"])
            dur = int(r.get("Duration (ns)") or r["Duration"])
        except (KeyError, ValueError, TypeError):
            continue
        name = r.get("Name") or ""
        if name.startswith("[CUDA mem"):           # copies / sets keep their own names
            pass
        out.append((start, dur, name, r.get("Strm") or r.get("Stream") or ""))
    out.sort()
    return out


def split(rows, gap_ns: int):
    reps, cur, end = [], [], None
    for r in rows:
        if cur and r[0] - end > gap_ns:
            reps.append(cur)
            cur = []
        cur.append(r)
        end = max(end or 0, r[0] + r[1]) if cur[1:] else r[0] + r[1]
    if cur:
        reps.append(cur)
    return reps


def union(rows) -> int:
    busy, s0, e0 = 0, None, None
    for s, d, _, _ in rows:
        e = s + d
        if s0 is None:
            s0, e0 = s, e
        elif s > e0:
            busy += e0 - s0
            s0, e0 = s, e
        else:
            e0 = max(e0, e)
    if s0 is not None:
        busy += e0 - s0
    return busy


def short(name: str) -> str:
    n = name.split("(")[0]
    n = n.replace("void ", "").replace("(anonymous namespace)::", "")
    return n[:70]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--gap-ms", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    rows = rows_of(a.csv)
    if not rows:
        sys.exit("no kernel rows")
    reps = [r for r in split(rows, int(a.gap_ms * 1e6)) if len(r) > 100]
    walls, busys, counts, coll, streams = [], [], [], [], defaultdict(list)
    per = defaultdict(list)
    for rep in reps:
        wall = max(s + d for s, d, _, _ in rep) - rep[0][0]
        walls.append(wall / 1e6)
        busys.append(union(rep) / 1e6)
        counts.append(len(rep))
        coll.append(sum(d for _, d, n, _ in rep if "collect_kernel" in n) / 1e6)
        acc = defaultdict(float)
        cnt = defaultdict(int)
        st = defaultdict(float)
        for s, d, n, strm in rep:
            acc[short(n)] += d / 1e6
            cnt[short(n)] += 1
            st[strm] += d / 1e6
        for k, v in acc.items():
            per[k].append((v, cnt[k]))
        for k, v in st.items():
            streams[k].append(v)
    med = statistics.median
    summary = {"replays": len(reps), "wall_ms": round(med(walls), 2), "busy_ms": round(med(busys), 2),
               "idle_ms": round(med([w - b for w, b in zip(walls, busys)]), 2), "kernels": int(med(counts)),
               "collect_ms": round(med(coll), 2),
               "streams_ms": {k: round(med(v), 2) for k, v in sorted(streams.items(), key=lambda kv: -med(kv[1]))}}
    print(json.dumps(summary), flush=True)
    top = sorted(per.items(), key=lambda kv: -med([v for v, _ in kv[1]]))[:a.top]
    for k, vs in top:
        print(f"  {med([v for v, _ in vs]):8.3f} ms  x{int(med([c for _, c in vs])):5d}  {k}", flush=True)


if __name__ == "__main__":
    main()
