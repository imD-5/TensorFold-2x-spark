"""The lane's decode round on its ranks (same command on each): the round graph (rounds.RoundRunner, as --parallel
serves) replayed for one stream's window of R rows, its median wall time by R (--sweep), the drafter's batched graph
for one stream, and with nsys a kernel timeline of the replays:

  nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda \\
      -o trace-r<rank> python3 tp4_trace.py --rank R --world 4 --master <HEAD_IP> --model M [--rows 6] [--reps 30]

The timed replays run between cudaProfilerStart / Stop, so the report holds only them (graph capture and warm-up stay
out). Rows carry random token ids (a window of random drafts). Every rank prints its line; with the RDMA gather's probes
(default) the mean staging, doorbell-to-flags and copy-out times of the gathers in the timed replays.
"""

import argparse
import json
import statistics
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=4)
    p.add_argument("--master", required=True)
    p.add_argument("--port", type=int, default=29668)
    p.add_argument("--model", required=True)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--sweep", default="", help="also time these row counts (e.g. 1,2,...,16), no profiler range")
    p.add_argument("--prompt-len", type=int, default=600)
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--warm", type=int, default=5)
    p.add_argument("--no-probe", action="store_true", help="skip the RDMA gather's latency probes")
    a = p.parse_args()
    from pathlib import Path

    from tensorfold.families.deepseek_v41.cuda import engine as E

    E.WARM = False
    eng = E.DsEngine(Path(a.model), rank=a.rank, world=a.world, master=a.master, port=a.port, drafts=5,
                     context=8192, parallel=4)
    mu, m = eng.multi, eng.model
    rgs = [x for x in (getattr(getattr(m.comm, "nccl", None), "rdma", None), getattr(m.comm, "col", None),
                       getattr(m.comm, "pair", None)) if x is not None]
    if not a.no_probe:
        for rg in rgs:
            rg.ext.configure(rg.h, True)           # before any capture: the graphs carry the probe pointer
    g = torch.Generator().manual_seed(1)
    P = a.prompt_len
    prompt = torch.randint(1000, 100000, (P,), generator=g).tolist()
    sc = m.pool_view(mu.pool, 0, 0, mu.extents.total)       # slot 0 over the whole window (as the engine's warm-up)
    eng.prefill(sc, None, prompt)
    torch.cuda.synchronize()
    host = np.zeros(P + 64, dtype=np.int32)
    host[:P] = prompt

    def window(R):
        ids = torch.randint(1000, 100000, (R,), generator=g).tolist()
        host[P:P + R] = ids
        return [(0, 0, mu.extents.total, P, ids, host[:P + R])]

    def timed(R, reps, profile=False):
        for _ in range(a.warm):                    # capture (first call) + warm replays
            mu.runner.forward(window(R))
        torch.cuda.synchronize()
        if profile:
            if not a.no_probe:
                for rg in rgs:
                    rg.ext.probes(rg.h)            # reset: only the timed replays' gathers count
            torch.cuda.cudart().cudaProfilerStart()
        times = []
        for _ in range(reps):
            w = window(R)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mu.runner.forward(w)
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - t0))
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return times

    out = {"rank": a.rank}
    times = timed(a.rows, a.reps, profile=True)
    out.update({"rows": a.rows, "reps": a.reps, "ms_median": round(statistics.median(times), 2),
                "ms_min": round(min(times), 2)})
    if not a.no_probe:
        names = ["all4", "col", "pair"][:len(rgs)] if len(rgs) == 3 else [f"g{i}" for i in range(len(rgs))]
        out["rdma_gather_us"] = {}
        for name, rg in zip(names, rgs):
            st, fl, cp, n = rg.ext.probes(rg.h)    # mean ns a gather: staging, doorbell -> every peer's flag, copy-out
            out["rdma_gather_us"][name] = {"stage": round(st / 1e3, 2), "doorbell_to_flags": round(fl / 1e3, 2),
                                           "copy_out": round(cp / 1e3, 2), "per_round": round(n / a.reps, 1)}
    # the drafter's batched graph for one stream (the lane's _drafts)
    try:
        for _ in range(a.warm):
            mu._drafts([prompt[-1]], [P], [0])
        torch.cuda.synchronize()
        dt = []
        for _ in range(a.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mu._drafts([prompt[-1]], [P], [0])
            torch.cuda.synchronize()
            dt.append(1000 * (time.perf_counter() - t0))
        out["draft_ms_median"] = round(statistics.median(dt), 2)
    except Exception as exc:                       # noqa: BLE001 (a profile without the drafter still helps)
        out["draft_error"] = f"{type(exc).__name__}: {exc}"[:200]
    if a.sweep:
        out["sweep_ms"] = {}
        for R in [int(x) for x in a.sweep.split(",") if x.strip()]:
            out["sweep_ms"][R] = round(statistics.median(timed(R, max(10, a.reps // 2))), 2)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
