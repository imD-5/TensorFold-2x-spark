"""Prompt chunks' indexer score kernel (_index_score_rows, keys + tile maxima as model.attention_k calls it at long
contexts) at other launch settings (rows a program RB, warps, stages) against the 8 rows / 8 warps launch, bit for bit,
then the times, at attention_k's row block for each key count. The default checks the module's own setting
(kernels.IDX_RB / IDX_WARPS / IDX_TREE); --rb / --warps sweep others, --tree with the kernel's explicit head-sum tree.

  python3 tools/dsv41/index_knobs_bench.py [--n 8192,32768,131072,524288] [--rb 2 --warps 2] [--stages 1]
      [--tree] [--require-faster] [--out F]
Exits 1 when the module's setting differs from the 8 / 8 launch in any key or tile maximum, or (with
--require-faster) when it is not faster at the two largest key counts. Swept settings are reported, not checked.
"""

import argparse
import json
import sys

import torch
import triton


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", default="8192,32768,131072,524288")
    p.add_argument("--rb", default="")
    p.add_argument("--warps", default="")
    p.add_argument("--stages", default="1")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--require-faster", action="store_true")
    p.add_argument("--tree", action="store_true", help="swept settings with the kernel's explicit head-sum tree (TREE)")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    g = torch.Generator(device="cuda").manual_seed(40)
    IH, ID, BN = 32, 128, 64
    mine = (K.IDX_RB, K.IDX_WARPS)
    sweep = [(rb, nw) for rb in (int(x) for x in a.rb.split(",") if x) for nw in (int(x) for x in a.warps.split(",") if x)]
    settings = [mine] + [s for s in sweep if s != mine and s != (8, 8)]
    res: dict = {"setting": f"RB{mine[0]} w{mine[1]}" + (" tree" if K.IDX_TREE else ""), "checks": {}, "equal": {},
                 "ms": {}, "failed": {}}
    ns_list = [int(x) for x in a.stages.split(",")]
    for n in (int(x) for x in a.n.split(",")):
        rows = min(2048, max(16, (1 << 26) // (4 * n)))            # attention_k's row block at this key count
        q = K.fp4_qd_p2((torch.randn((rows, IH, ID), generator=g, device="cuda") * 0.7).to(torch.bfloat16))
        codes = torch.randint(0, 256, (n, ID // 2), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        scales = torch.randint(118, 131, (n, ID // 32), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        w = (torch.randn((rows, IH), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
        vis = (n - (rows - 1 - torch.arange(rows, device="cuda")) // 2).to(torch.int64)
        nt = triton.cdiv(n, BN)

        def run(rb, nw, ns, tree=False):
            out = torch.empty((rows, n), dtype=torch.int64, device="cuda")
            tm = torch.empty((rows, nt), dtype=torch.int64, device="cuda")
            K._index_score_rows[(triton.cdiv(rows, rb), nt)](q, codes, scales, w, vis, out, n, rows, tm, nt, IH=IH,
                                                              ID=ID, BN=BN, RB=rb, PACKED=True, KEYS=True,
                                                              HAS_TMAX=True, TREE=tree, num_warps=nw, num_stages=ns)
            return out, tm

        def timed(rb, nw, ns, tree=False):
            for _ in range(2):
                run(rb, nw, ns, tree)
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(a.iters):
                run(rb, nw, ns, tree)
            t1.record()
            torch.cuda.synchronize()
            return round(t0.elapsed_time(t1) / a.iters, 3)

        ref = run(8, 8, 3)
        res["ms"][f"n{n} rows{rows} RB8 w8 (reference)"] = timed(8, 8, 3)
        for rb, nw in settings:
            for ns in ns_list:
                tree = K.IDX_TREE if (rb, nw) == mine else a.tree
                key = f"n{n} rows{rows} RB{rb} w{nw} s{ns}" + (" tree" if tree else "")
                try:
                    o = run(rb, nw, ns, tree)
                except Exception as exc:
                    res["failed"][key] = f"{exc!s}"[:160]
                    continue
                eq = bool(torch.equal(o[0], ref[0]) and torch.equal(o[1], ref[1]))
                del o
                res["equal"][key] = eq
                res["ms"][key] = timed(rb, nw, ns, tree)
                if (rb, nw) == mine:
                    res["checks"][key + " == RB8 w8"] = eq
                print(json.dumps({key: res["ms"][key], "equal": eq,
                                  "reference ms": res["ms"][f"n{n} rows{rows} RB8 w8 (reference)"]}), flush=True)
        del q, codes, scales, w, vis, ref
        torch.cuda.empty_cache()
    if a.require_faster:
        for n in sorted(int(x) for x in a.n.split(","))[-2:]:
            ref_ms = next(v for k, v in res["ms"].items() if k.startswith(f"n{n} ") and k.endswith("(reference)"))
            mine_ms = min(v for k, v in res["ms"].items() if k.startswith(f"n{n} ") and f" RB{mine[0]} w{mine[1]} " in k)
            res["checks"][f"n{n} RB{mine[0]} w{mine[1]} faster"] = mine_ms < ref_ms
    res["passed"] = bool(res["checks"]) and all(res["checks"].values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
