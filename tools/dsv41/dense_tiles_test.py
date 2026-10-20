"""Prompt chunks' dense EXL3 matmuls (cuda/exl3/prefill.py) under other tile settings than the served ones (128 rows a
program, a 32-wide K step, 8 warps, 4 stages): every dense linear of real layers (rank 0's half of a TP2 split), the
outputs compared bit for bit with the served setting at prompt-chunk row counts, then the times at 2048 rows. A setting
whose outputs equal the served ones for every linear and row count is an exact speedup candidate.

  python3 tools/dsv41/dense_tiles_test.py --model M [--layers 10,20] [--rows 2048,1000] [--out F]   (one GPU, idle)
Exits 1 when the by-shape tiles (TF_EXL3_PREFILL_TILES=1) change any output, or a row-strided input does.
"""

import argparse
import json
import sys

import torch

# (rows a program, K step, warps, stages, row blocks a raster group); None = prefill.tiles(..., by_shape=True)
SETTINGS = {"served": (128, 32, 8, 4, 8), "by_shape": None, "128/64/8/3": (128, 64, 8, 3, 8),
            "64/64/4/4": (64, 64, 4, 4, 8), "128/64/8/4": (128, 64, 8, 4, 8), "64/32/4/4": (64, 32, 4, 4, 8)}


def linears(obj, prefix=""):
    from tensorfold.cuda.exl3.linear import Exl3Linear

    seen = set()
    for name in sorted(vars(obj)):
        v = getattr(obj, name)
        if isinstance(v, Exl3Linear) and id(v) not in seen:
            seen.add(id(v))
            yield prefix + name, v
        elif isinstance(v, (list, tuple)):
            for i, u in enumerate(v):
                if isinstance(u, Exl3Linear) and id(u) not in seen:
                    seen.add(id(u))
                    yield f"{prefix}{name}[{i}]", u


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layers", default="10,20")
    p.add_argument("--rows", default="2048,1000")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.cuda.exl3 import prefill as P
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda.weights import Shards, load_block

    cfg = Cfg.read(a.model)
    sh = Shards(a.model)
    rows = [int(r) for r in a.rows.split(",")]
    ws = P.Workspace()
    res: dict = {"checks": {}, "equal": {}, "times_ms": {}, "shapes": {}}
    g = torch.Generator(device="cuda").manual_seed(11)
    orig = P.tiles

    def run(lin, x, out, st):
        if st is None:
            return P.matmul(lin, x, out, ws, by_shape=True)
        P.tiles = lambda k, n, by_shape=None: st
        try:
            return P.matmul(lin, x, out, ws, by_shape=False)
        finally:
            P.tiles = orig

    for L in (int(v) for v in a.layers.split(",")):
        lay = load_block(sh, cfg, f"layers.{L}", L, 0, 2, cfg.n_routed)
        for name, lin in linears(lay):
            key = f"L{L} {name}"
            res["shapes"][key] = [lin.k, lin.n]
            for R in rows:
                x = (torch.randn((R, lin.k), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
                base = None
                for sname, st in SETTINGS.items():
                    out = torch.empty((R, lin.n), dtype=torch.bfloat16, device="cuda")
                    try:
                        y = run(lin, x, out, st).clone()
                    except Exception as exc:                        # noqa: BLE001
                        res["equal"][f"{key} R{R} {sname}"] = f"error: {str(exc)[:120]}"
                        continue
                    if sname == "served":
                        base = y
                        continue
                    eq = bool(torch.equal(y, base))
                    res["equal"][f"{key} R{R} {sname}"] = eq
                    if sname == "by_shape":
                        res["checks"][f"{key} R{R}: by-shape tiles == served"] = eq
                    if R == max(rows):
                        for _ in range(2):
                            run(lin, x, out, st)
                        torch.cuda.synchronize()
                        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        t0.record()
                        for _ in range(a.iters):
                            run(lin, x, out, st)
                        t1.record()
                        torch.cuda.synchronize()
                        res["times_ms"][f"{key} R{R} {sname}"] = round(t0.elapsed_time(t1) / a.iters, 3)
                # a row-strided view of the same rows (wo_a's slices of the attention output): the same outputs
                wide = torch.empty((R, lin.k + 256), dtype=torch.bfloat16, device="cuda")
                wide[:, 128:128 + lin.k] = x
                xs = wide[:, 128:128 + lin.k]
                out = torch.empty((R, lin.n), dtype=torch.bfloat16, device="cuda")
                res["checks"][f"{key} R{R}: row-strided input == contiguous"] = bool(
                    torch.equal(run(lin, xs, out, SETTINGS["served"]).clone(), base))
                if R == max(rows):
                    out = torch.empty((R, lin.n), dtype=torch.bfloat16, device="cuda")
                    for _ in range(2):
                        run(lin, x, out, SETTINGS["served"])
                    torch.cuda.synchronize()
                    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    t0.record()
                    for _ in range(a.iters):
                        run(lin, x, out, SETTINGS["served"])
                    t1.record()
                    torch.cuda.synchronize()
                    res["times_ms"][f"{key} R{R} served"] = round(t0.elapsed_time(t1) / a.iters, 3)
                print(json.dumps({k: v for k, v in res["equal"].items() if k.startswith(f"{key} R{R}")}), flush=True)
        del lay
        torch.cuda.empty_cache()
    res["passed"] = all(res["checks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
