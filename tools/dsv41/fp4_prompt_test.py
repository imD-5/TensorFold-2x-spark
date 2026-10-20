"""Prompt chunks' FP4 writes through the decode path's one-launch kernels: K.fp4_qd_p2 against
ops.fp4_qd(x, 32, e4m3_scale=False) for indexer queries, and K.fp4_store against model.store_rows for packed caches
(index keys: 32-element blocks, E8M0 scales; compressed latents: 16-element blocks, E4M3 scales), bit for bit at prompt
row counts, with exact ties of the E2M1 rounding, zeros, tiny and huge values mixed in; the other cache rows untouched.

  python3 tools/dsv41/fp4_prompt_test.py --model M [--out F]      (one GPU)
Exits 1 when a check fails.
"""

import argparse
import json
import sys

import torch


def values(shape, gen, dev, block):
    """bf16 values with every case of the E2M1 rounding: random, zeros, ties (|v| / scale at 0.75, 1.75, 3.5, 0.25,
    1.25, 2.5, 5.0 with the block's amax at 6 x scale), negatives, tiny and huge blocks."""

    n = 1
    for d in shape:
        n *= d
    v = torch.randn((n // block, block), generator=gen) * 3.0
    k = v.shape[0]
    ties = torch.tensor([0.75, 1.75, 3.5, 0.25, 1.25, 2.5, 5.0, 6.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 0.0, -0.0],
                        dtype=torch.float32)
    sel = torch.arange(k) % 4 == 1                           # a quarter of the blocks: exact grid points and ties
    m = int(sel.sum())
    reps = (block + ties.numel() - 1) // ties.numel()
    tb = ties.repeat(reps)[:block]
    sign = torch.where(torch.rand((m, block), generator=gen) < 0.5, -1.0, 1.0)
    scale = 2.0 ** torch.randint(-8, 6, (m, 1), generator=gen).float()
    v[sel] = tb[None, :] * sign * scale
    v[sel, 0] = 6.0 * scale[:, 0]                            # amax = 6 x scale: |v| / s lands on the listed values
    v[torch.arange(k) % 7 == 3] = 0.0                        # all-zero blocks
    v[torch.arange(k) % 11 == 5] *= 1e-30                    # tiny
    v[torch.arange(k) % 13 == 6] *= 1e4                      # huge
    return v.reshape(shape).to(torch.bfloat16).to(dev)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41 import ops
    from tensorfold.families.deepseek_v41.config import Cfg
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import model as M

    cfg = Cfg.read(a.model)
    dev = "cuda"
    gen = torch.Generator().manual_seed(28)
    res: dict = {"shapes": {"idx_heads": cfg.idx_heads, "idx_dim": cfg.idx_dim, "head_dim": cfg.head_dim},
                 "checks": {}}
    ok = res["checks"]
    for n in (2048, 1000, 17):
        iq = values((n, cfg.idx_heads, cfg.idx_dim), gen, dev, 32)
        ok[f"fp4_qd_p2 == fp4_qd, {n} rows"] = bool(torch.equal(K.fp4_qd_p2(iq),
                                                                ops.fp4_qd(iq, 32, e4m3_scale=False)))
    for d, block, e4m3, name in ((cfg.idx_dim, 32, False, "index keys"), (cfg.head_dim, 16, True, "latents")):
        for n in (512, 250, 3):
            x = values((n, d), gen, dev, block)
            total = 4 * n + 64
            rows = torch.arange(37, 37 + n, device=dev)
            fill_c = torch.randint(0, 256, (total, d // 2), generator=gen, dtype=torch.uint8).to(dev)
            fill_s = torch.randint(0, 256, (total, d // block), generator=gen, dtype=torch.uint8).to(dev)
            ca = (fill_c.clone(), fill_s.clone())
            cb = (fill_c.clone(), fill_s.clone())
            M.store_rows(ca, rows, x, block, e4m3)
            K.fp4_store(x, cb, rows, block, e4m3)
            torch.cuda.synchronize()
            ok[f"fp4_store == store_rows, {name}, {n} rows"] = bool(torch.equal(ca[0], cb[0])
                                                                    and torch.equal(ca[1], cb[1]))
    res["passed"] = all(ok.values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
