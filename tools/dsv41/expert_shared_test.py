"""A prompt chunk's shared expert as one dense GEMM (TF_EXL3_SHARED_DENSE, experts.SHARED_DENSE) against the grouped
launches it replaces: the routed outputs (every slot weighted and summed) bit for bit on real layers (rank 0's half of
a TP2 split) at prompt row counts, served kernels and kernel "rows"; then the times.

  python3 tools/dsv41/expert_shared_test.py --model M [--layers 10,2] [--rows 2048,1000,256,64] [--require-faster]
      [--out F]                                                         (one GPU, lane idle)
Exits 1 when a check fails.
"""

import argparse
import json
import sys

import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="10,2")
    p.add_argument("--rows", default="2048,1000,256,64")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--require-faster", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3 import experts as X
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    rows = [int(r) for r in a.rows.split(",")]
    slots = cfg.topk + 1
    res: dict = {"checks": {}, "times_ms": {}}
    for L in (int(v) for v in a.layers.split(",")):
        lay = load_block(sh, cfg, f"layers.{L}", L, 0, 2, cfg.n_routed)
        ex = lay.experts
        E, D = ex.count - 1, cfg.dim
        res["checks"][f"L{L} shared trellis found"] = X._shared_trellis(ex) is not None
        s = X.Scratch(ex, rows=max(rows), slots=slots, prompt=True)
        g = torch.Generator(device="cuda").manual_seed(25 + L)
        for R in rows:
            x = (torch.randn((R, D), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
            routed = torch.stack([torch.randperm(E, generator=g, device="cuda")[:cfg.topk]
                                  for _ in range(R)]).to(torch.int32)
            pick = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device="cuda")], 1).contiguous()
            wts = torch.rand((R, slots), generator=g, device="cuda")
            wts[:, -1] = 1.0
            for kern in ("mma", "rows"):
                outs = {}
                for dense in (False, True):
                    X.SHARED_DENSE = dense
                    run = lambda: X.routed(x, pick, wts, ex, s, None, R, limit=cfg.swiglu_limit,
                                           act_mode=X.ACT_F32, kernel=kern, shared_slot=slots - 1)
                    outs[dense] = run().clone()
                    if R == max(rows) or R == rows[1 % len(rows)]:
                        for _ in range(3):
                            run()
                        torch.cuda.synchronize()
                        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        t0.record()
                        for _ in range(a.iters):
                            run()
                        t1.record()
                        torch.cuda.synchronize()
                        key = f"L{L} R{R} {kern}, shared {'dense' if dense else 'grouped'}"
                        res["times_ms"][key] = round(t0.elapsed_time(t1) / a.iters, 3)
                        print(json.dumps({key: res["times_ms"][key]}), flush=True)
                res["checks"][f"L{L} R{R} {kern}: dense == grouped"] = bool(torch.equal(outs[True], outs[False]))
        X.SHARED_DENSE = True
        del lay, ex, s
        torch.cuda.empty_cache()
    if a.require_faster:
        for L in (int(v) for v in a.layers.split(",")):
            R = max(rows)
            res["checks"][f"L{L} R{R} served kernels faster with the dense shared expert"] = (
                res["times_ms"][f"L{L} R{R} mma, shared dense"] < res["times_ms"][f"L{L} R{R} mma, shared grouped"])
    res["passed"] = all(res["checks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
