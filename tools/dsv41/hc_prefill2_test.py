"""Prompt chunks' mHC through hc_pre_pf (switch "hc_pf2": RB rows a program share each mixing weight tile, a sublayer's
post fused into the next one's mixes) against the kernels they replace (hc_post then hc_pre, one row a program): the
posted streams, the normed input, pre / post / comb and the mixes' partial sums, bit for bit, at prompt-chunk row
counts and every RB tried; then their times at 2048 rows.

  python3 tools/dsv41/hc_prefill2_test.py [--rb 4,8,16] [--require-faster] [--out F]      (one GPU, lane idle)
Exits 1 when a check fails (--require-faster: also when post + mixes at the default RB are not faster than
hc_post + hc_pre).
"""

import argparse
import json
import sys
import time

import torch


def inputs(R, D, gen, dev):
    h = (torch.randn((R, 4, D), generator=gen) * 0.7).to(torch.bfloat16).to(dev)
    fn = (torch.randn((24, 4 * D), generator=gen) * 0.02).to(dev)
    scale = (torch.rand((3,), generator=gen) + 0.5).to(dev)
    base = (torch.randn((24,), generator=gen) * 0.5).to(dev)
    pre_in = (torch.rand((R, 4), generator=gen) * 0.9 + 0.05).to(dev)
    norm = (torch.rand((D,), generator=gen) + 0.5).to(torch.bfloat16).to(dev)
    g = (torch.randn((2, R, D), generator=gen) * 0.3).to(dev)
    post = (torch.rand((R, 4), generator=gen) * 2.0).to(dev)
    comb = torch.rand((R, 4, 4), generator=gen)
    comb = (comb / comb.sum(1, keepdim=True)).to(dev)
    return h, fn, scale, base, pre_in, norm, g, post, comb


def outs(R, D, dev, nb):
    return (torch.empty((R, D), dtype=torch.bfloat16, device=dev), torch.empty((R, 4), device=dev),
            torch.full((R * nb * 32,), float("nan"), device=dev))


def sums(part, R, nb):
    """The partial sums the finish reads: the 24 mixes and the sum of squares of each (row, K block)."""

    return part.view(R, nb, 32)[:, :, :25]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rb", default="4,8,16")
    p.add_argument("--require-faster", action="store_true")
    p.add_argument("--out")
    a = p.parse_args()
    from tensorfold.families.deepseek_v41.cuda import kernels as K

    dev = "cuda"
    gen = torch.Generator().manual_seed(7)
    D, eps, hc_eps, iters = 5120, 1e-6, 1e-6, 20
    nb = K.HC_BLOCKS
    rbs = [int(v) for v in a.rb.split(",")]
    res: dict = {"checks": {}, "times_ms": {}}
    ok = res["checks"]
    for R in (2048, 1919, 128, 33, 17, 5):
        h, fn, scale, base, pre_in, norm, g, post0, comb0 = inputs(R, D, gen, dev)
        # the reference: the post (attention or MoE), then the next sublayer's mixes; and the mixes alone
        h1, post1, comb1 = h.clone(), post0.clone(), comb0.clone()
        x1, pre1, part1 = outs(R, D, dev, nb)
        K.hc_post(g, h1, post1, comb1, h1)
        K.hc_pre(h1, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x1, pre1, post1, comb1, part1)
        post3, comb3 = post0.clone(), comb0.clone()
        x3, pre3, part3 = outs(R, D, dev, nb)
        K.hc_pre(h, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x3, pre3, post3, comb3, part3)
        for rb in rbs:
            K.HC_PF_RB = rb
            h2, post2, comb2 = h.clone(), post0.clone(), comb0.clone()
            x2, pre2, part2 = outs(R, D, dev, nb)
            h_out = torch.empty_like(h2)
            src = K.hc_pre_pf(h2, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x2, pre2, post2, comb2, part2,
                              gathered=g, h_out=h_out)
            post4, comb4 = post0.clone(), comb0.clone()
            x4, pre4, part4 = outs(R, D, dev, nb)
            src4 = K.hc_pre_pf(h, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x4, pre4, post4, comb4, part4)
            torch.cuda.synchronize()
            ok[f"posted/{R}/rb{rb}/streams"] = bool(torch.equal(src, h1)) and src.data_ptr() == h_out.data_ptr()
            ok[f"posted/{R}/rb{rb}/sums"] = bool(torch.equal(sums(part2, R, nb), sums(part1, R, nb)))
            ok[f"posted/{R}/rb{rb}/x_pre_post_comb"] = bool(torch.equal(x2, x1) and torch.equal(pre2, pre1)
                                                            and torch.equal(post2, post1) and torch.equal(comb2, comb1))
            ok[f"posted/{R}/rb{rb}/input_untouched"] = bool(torch.equal(h2, h))
            ok[f"plain/{R}/rb{rb}/sums"] = bool(torch.equal(sums(part4, R, nb), sums(part3, R, nb)))
            ok[f"plain/{R}/rb{rb}/x_pre_post_comb"] = bool(src4.data_ptr() == h.data_ptr() and torch.equal(x4, x3)
                                                           and torch.equal(pre4, pre3) and torch.equal(post4, post3)
                                                           and torch.equal(comb4, comb3))

    def timed(fn_, reps=20):
        for _ in range(3):
            fn_()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn_()
            torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t0))
        return round(sorted(ts)[len(ts) // 2], 3)

    R = 2048
    h, fn, scale, base, pre_in, norm, g, post0, comb0 = inputs(R, D, gen, dev)
    x, pre, part = outs(R, D, dev, nb)
    hb = h.clone()
    hb2 = torch.empty_like(h)
    post, comb = post0.clone(), comb0.clone()

    def old_pair():
        K.hc_post(g, hb, post, comb, hb)
        K.hc_pre(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb, part)

    t = res["times_ms"]
    t["2048 rows: post + mixes, old (hc_post + hc_pre)"] = timed(old_pair)
    t["2048 rows: mixes alone, old (hc_pre)"] = timed(
        lambda: K.hc_pre(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb, part))
    t["2048 rows: hc_post alone"] = timed(lambda: K.hc_post(g, hb, post, comb, hb))
    for rb in rbs:
        K.HC_PF_RB = rb
        t[f"2048 rows: post + mixes, hc_pre_pf rb{rb}"] = timed(
            lambda: K.hc_pre_pf(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb, part,
                                gathered=g, h_out=hb2))
        t[f"2048 rows: mixes alone, hc_pre_pf rb{rb}"] = timed(
            lambda: K.hc_pre_pf(hb, fn, scale, base, pre_in, norm, eps, hc_eps, iters, x, pre, post, comb, part))
        t[f"2048 rows: post + dots alone (hc_mix_pf) rb{rb}"] = timed(
            lambda: K.hc_mix_pf(hb, fn, part, g, hb2, post, comb))
    if a.require_faster:
        rb0 = K.HC_PF_RB_DEFAULT
        ok[f"2048 rows: post + mixes faster at the default rb{rb0}"] = (
            t[f"2048 rows: post + mixes, hc_pre_pf rb{rb0}"] < t["2048 rows: post + mixes, old (hc_post + hc_pre)"])
        # (the plain mixes stay on hc_pre in the model: hc_mix_pf's plain dots are only timed here)
    res["passed"] = all(ok.values())
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
