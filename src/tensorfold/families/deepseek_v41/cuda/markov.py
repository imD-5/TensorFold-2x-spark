"""DSpark's Markov steps (the drafter's last part) as fixed-order kernels inside the drafter's graph, with the bias rows
of frequent tokens cached and the vocabulary split over the TP ranks. The drafts are 3b9818d's, bit for bit.

Step i of the drafter's Markov loop picks d_{i+1} = argmax_v (logits_i[v] + bias(d_i)[v]) with bias(t) =
fp16(markov_head @ e(t)), e(t) = fp16(markov_embed[t]) (3b9818d: a cuBLAS GEMM [N, rank] x [rank, V] whose fp16
output is widened and added, then torch.argmax). Here, a step is:

- ``_bias``: for each row whose token has no cached row, every bias element as one tensor-core dot of the head row and
  e(t) (``tl.dot`` [64, rank] x [rank, 16]: fp32 accumulation in rank order, 16 at a time, then fp16 round to nearest)
  into a staging row. The per-element arithmetic does not depend on the vocabulary range, the tile, the row count
  (1..16) or the other rows, and it is the arithmetic of the cuBLAS kernel 3b9818d runs at these shapes:
  markov_check.py finds every element of every token's row equal to cuBLAS's (N = 1, 2, 4 and the serial
  matrix-vector shape). A launch whose rows all have cached rows only reads the tokens and slots.
- ``_score``: the bias (the token's cached row, else its staging row), widened, plus the fp32 logit; each program's
  best (value, index) a row under torch.argmax's order (NaN above everything, then the larger value, ties to the lower
  index; a total order, so any reduction tree gives the same best). ``_finish``: the row's best of the programs'.
- **Cached rows** (``TF_DS_MARKOV_CACHE`` = K, default 256; 0: none): bias(t) of the first K tokens of
  ``markov_tokens.TOKENS`` (a frequency ranking of a code + English sample), made at load time by ``_bias`` itself, so
  a cached row has the computed bits. Which tokens have rows changes the time, never a draft. K x V / world fp16 a rank.
- **Vocabulary split** (``TF_DS_MARKOV_SPLIT=1``, default; TP > 1): each rank scores its vocabulary half (its half of
  the head's rows and the cache's columns) against its own half of the drafter head's logits (no logits gather), and
  the ranks' bests, (value, index) a row, go through one small gather a step; ``_pick`` takes the best in the same
  order (the lower index, so rank 0's, wins a tie): the best of the union under a total order is the best of the
  halves' bests, so the drafts are the unsplit loop's. Every rank picks from the same gathered bytes: the same drafts
  on every rank. ``TF_DS_MARKOV_SPLIT=0``: the gathered logits, every column on every rank.
- **Coupled drafts** (``TF_DS_DRAFT_COUPLED=1``, default; SPLIT or one rank): a sampled stream's draft is the token its
  own keyed draw (engine/exact_sampling.py ``choose_rows``: the top_k by (value down, id up), then the argmax of
  value / temperature + G(seed, position, id) over the candidates its top_p and min_p keep) gives on the drafter's
  scores, so a drafter that scores as the target
  does drafts the sampled tokens. The same three launches a step and the same one gather: ``_score`` also keeps each
  program's CAND best (value, id) of a sampled row, ``_finish`` the row's CAND best of the programs' (sent beside the
  greedy best), ``_pick`` merges the ranks' candidates and draws. A greedy row (divisor 0) keeps the argmax. Drafts only
  propose: a coupled pick changes how many drafts a round keeps, never a reply.

``TF_DS_MARKOV=0``: 3b9818d's loop.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from . import kernels as K

ON = os.environ.get("TF_DS_MARKOV", "1") != "0"               # 0: 3b9818d's cuBLAS loop
SPLIT = os.environ.get("TF_DS_MARKOV_SPLIT", "1") != "0"
# cached rows: 256 = 33 MB a rank (4096: 530 MB, ~0.1-0.2 ms a 1-stream round faster; the memory floor is a tracked row)
CACHE_ROWS = int(os.environ.get("TF_DS_MARKOV_CACHE") or 256)
BN = 64                 # vocabulary entries a dot (V / world is a multiple of it)
SUB = 4                 # dots a bias program, one after another
BC = 1024               # vocabulary entries a scoring program
NP = 16                 # rows a launch at most (tl.dot's smallest N); a step's rows are its streams
BIG = tl.constexpr(1 << 30)
COUPLED = os.environ.get("TF_DS_DRAFT_COUPLED", "1") != "0"
CAND = 32               # candidates a rank sends a sampled row (a stream couples only with 0 < top_k <= CAND)
SEND_W = 4 + 2 * CAND   # a coupled step's gathered row: the greedy best (value, index, 0, 0), CAND values, CAND ids
# a sampled stream's drafts divide the drafter's scores by its temperature x TF_DS_DRAFT_COUPLED_SCALE: below 1 the
# drafts keep nearer the drafter's argmax (0.5 on the pair: T equal or higher than greedy drafts on every workload;
# 1.0 lost T on prose)
COUPLED_SCALE = float(os.environ.get("TF_DS_DRAFT_COUPLED_SCALE") or 0.5)


@triton.jit
def _better(v1, i1, v2, i2):
    """(v1, i1) before (v2, i2) in torch.argmax's order: NaN above everything, then the larger value, ties (and NaN
    against NaN) to the lower index."""

    n1 = v1 != v1
    n2 = v2 != v2
    gt = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return (n1 & (~n2 | (i1 < i2))) | (~n1 & ~n2 & gt)


@triton.jit
def _best(v1, i1, v2, i2):
    t = _better(v1, i1, v2, i2)
    return tl.where(t, v1, v2), tl.where(t, i1, i2)


@triton.jit
def _key(v, i):
    """(value fp32, index int32) -> int64 keys whose order is the target's candidate order reversed: the value's
    ordered bits above (-0 as +0; NaN above everything, as torch.topk), the inverted index below (ties: the lower id
    is the larger key). A total order: sorted down, the first k are the top k."""

    bits = tl.where(v == 0.0, 0, v.to(tl.int32, bitcast=True))
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    return (ordered << 32) | (0xFFFFFFFF - i.to(tl.int64))


@triton.jit
def _unkey(key):
    o = (key >> 32).to(tl.int32)
    v = tl.where(o < 0, o ^ 0x7FFFFFFF, o).to(tl.float32, bitcast=True)
    return v, (0xFFFFFFFF - (key & 0xFFFFFFFF)).to(tl.int32)


@triton.jit
def _mix(x):
    """splitmix64's finalizer on uint64 (exact_sampling._mix)."""

    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    return x ^ (x >> 31)


@triton.jit
def _gumbel(seed, pos, ids):
    """The target's Gumbel draws -log(-log(u)) for token ``ids`` at one (seed, position): u is exact_sampling.uniform's
    double, bit for bit; the logs are libdevice's (within an ulp of numpy's)."""

    x = _mix(seed.to(tl.uint64, bitcast=True) + 0x9E3779B97F4A7C15)
    x = _mix(x ^ (pos.to(tl.uint64, bitcast=True) * 0xD1B54A32D192ED03))
    x = _mix(x ^ ids.to(tl.uint64))
    u = (x >> 11).to(tl.float64) * 1.1102230246251565e-16 + 5.551115123125783e-17     # * 2^-53 + 2^-54
    return -libdevice.log(-libdevice.log(u))


@triton.jit(do_not_specialize=["t_stride", "o_row", "n_rows", "n_cols", "seg0"])
def _bias(TOK, t_stride, SLOT, EMB, HEAD, OUT, o_row, n_rows, n_cols, seg0, BN: tl.constexpr, SUB: tl.constexpr,
          RANK: tl.constexpr, NP: tl.constexpr, PDL: tl.constexpr = False):
    """Program (t, s): for each row whose token has no slot, OUT[r, s * n_cols + c] = fp16(head[v] . fp16(emb[tok]))
    for SUB tiles of BN columns from (t * SUB) * BN (vocabulary entries v = (seg0 + s) * n_cols + c)."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    t = tl.program_id(0)
    s = tl.program_id(1)
    r = tl.arange(0, NP)
    rm = r < n_rows
    tok = tl.load(TOK + r * t_stride, mask=rm, other=0)
    slot = tl.load(SLOT + tok, mask=rm, other=0)
    miss = rm & (slot < 0)
    if tl.max(miss.to(tl.int32), axis=0) > 0:
        k = tl.arange(0, RANK)
        e = tl.load(EMB + tok[:, None].to(tl.int64) * RANK + k[None, :], mask=miss[:, None], other=0.0)
        e = e.to(tl.float16)
        for u in tl.static_range(SUB):
            c = (t * SUB + u) * BN + tl.arange(0, BN)
            cm = c < n_cols
            v = (seg0 + s) * n_cols + c
            w = tl.load(HEAD + v[:, None].to(tl.int64) * RANK + k[None, :], mask=cm[:, None], other=0.0)
            acc = tl.dot(w, tl.trans(e))                  # [BN, NP] fp32, rank order
            tl.store(OUT + r[None, :].to(tl.int64) * o_row + (s * n_cols + c)[:, None], acc.to(tl.float16),
                     mask=miss[None, :] & cm[:, None])


@triton.jit(do_not_specialize=["l_seg", "l_row", "t_stride", "c_row", "s_row", "n_rows", "n_cols", "seg0",
                               "n_part"])
def _score(LG, l_seg, l_row, TOK, t_stride, SLOT, CACHE, c_row, STAGE, s_row, PV, PI, n_rows, n_cols, seg0, n_part,
           DIV, PK, BC: tl.constexpr, CPL: tl.constexpr = False, CAND: tl.constexpr = 32, PDL: tl.constexpr = False):
    """Program (t, s): columns t * BC .. of segment s for each row: score = logit + widened bias (the token's cached
    row, else its staging row); the best (value, index) a row into PV / PI [rows, n_part]. ``CPL``: also, for each
    sampled row (DIV[r] > 0), the program's CAND best keys (``_key``, sorted down) into PK [rows, n_part, CAND]."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    t = tl.program_id(0)
    s = tl.program_id(1)
    c = t * BC + tl.arange(0, BC)
    cm = c < n_cols
    cc = s * n_cols + c
    v = (seg0 + s) * n_cols + c
    idx = tl.where(cm, v, BIG)
    p = s * tl.num_programs(0) + t
    for r in range(n_rows):
        tok = tl.load(TOK + r * t_stride)
        slot = tl.load(SLOT + tok)
        hit = slot >= 0
        bc = tl.load(CACHE + slot.to(tl.int64) * c_row + cc, mask=cm & hit, other=0.0)
        bs = tl.load(STAGE + r * s_row + cc, mask=cm & (slot < 0), other=0.0)
        b = tl.where(hit, bc, bs)
        lg = tl.load(LG + s * l_seg + r * l_row + c, mask=cm, other=0.0)
        score = tl.where(cm, lg + b.to(tl.float32), float("-inf"))
        bv, bi = tl.reduce((score, idx), 0, _best)
        tl.store(PV + r * n_part + p, bv)
        tl.store(PI + r * n_part + p, bi)
        if CPL:  # noqa: SIM102 (a constexpr test, then a runtime one)
            if tl.load(DIV + r) > 0:
                keys = tl.sort(_key(score, idx), descending=True)
                j = tl.arange(0, BC)
                tl.store(PK + (r * n_part + p) * CAND + j, keys, mask=j < CAND)


@triton.jit(do_not_specialize=["n_part", "o_stride", "send"])
def _finish(PV, PI, n_part, OUT, o_stride, SEND, send, DIV, PK, BP: tl.constexpr, CPL: tl.constexpr = False,
            CAND: tl.constexpr = 32, BK: tl.constexpr = 32, SW: tl.constexpr = 4, PDL: tl.constexpr = False):
    """Row r's best of its n_part program bests: the token into OUT[r * o_stride] (int64), or (``send``) (value,
    index, 0, 0) fp32 into SEND[r] (SW wide) for the ranks' gather. ``CPL``, a sampled row: its CAND best of the
    programs' candidates, values then ids (fp32: exact below 2^24), into SEND[r, 4:]."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    p = tl.arange(0, BP)
    pm = p < n_part
    v = tl.load(PV + r * n_part + p, mask=pm, other=float("-inf"))
    i = tl.load(PI + r * n_part + p, mask=pm, other=BIG)
    bv, bi = tl.reduce((v, i), 0, _best)
    if send != 0:
        q = tl.arange(0, 4)
        val = tl.where(q == 0, bv, tl.where(q == 1, bi.to(tl.float32), 0.0))
        tl.store(SEND + r * SW + q, val)
        if CPL:  # noqa: SIM102 (a constexpr test, then a runtime one)
            if tl.load(DIV + r) > 0:
                j = tl.arange(0, BK)
                keys = tl.load(PK + r * n_part * CAND + j, mask=j < n_part * CAND, other=-9223372036854775808)
                cv, ci = _unkey(tl.sort(keys, descending=True))
                tl.store(SEND + r * SW + 4 + j, cv, mask=j < CAND)
                tl.store(SEND + r * SW + 4 + CAND + j, ci.to(tl.float32), mask=j < CAND)
    else:
        tl.store(OUT + r * o_stride, bi.to(tl.int64))


@triton.jit(do_not_specialize=["n_rows", "o_stride", "step"])
def _pick(G, n_rows, OUT, o_stride, DIV, SEED, TOPK, Q0, TOPP, MINL, step, WORLD: tl.constexpr,
          CPL: tl.constexpr = False, CAND: tl.constexpr = 32, BW: tl.constexpr = 64, SW: tl.constexpr = 4,
          PDL: tl.constexpr = False):
    """Row r's token from the ranks' (value, index) bests [WORLD, rows, SW], in the same order. ``CPL``, a sampled
    row (DIV[r] > 0: temperature x COUPLED_SCALE): the ranks' candidates in the target's order, the first TOPK[r]
    (0: all), of them the prefix whose softmax mass before it is below TOPP[r] (1: all) and those within MINL[r] of
    the first (ln min_p; -inf: all), and of these the argmax of value / DIV[r] + G(SEED[r], Q0[r] + 1 + step, id):
    choose_rows's draw at the position the draft takes, with the drafter's scores."""

    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    r = tl.program_id(0)
    bv = tl.load(G + r * SW)
    bi = tl.load(G + r * SW + 1).to(tl.int32)
    for w in tl.static_range(1, WORLD):
        v = tl.load(G + (w * n_rows + r) * SW)
        i = tl.load(G + (w * n_rows + r) * SW + 1).to(tl.int32)
        bv, bi = _best(v, i, bv, bi)
    if CPL:
        div = tl.load(DIV + r)
        if div > 0:
            q = tl.arange(0, BW)
            qm = q < WORLD * CAND
            at = G + ((q // CAND) * n_rows + r) * SW + 4 + q % CAND
            cv = tl.load(at, mask=qm, other=float("-inf"))
            ci = tl.load(at + CAND, mask=qm, other=BIG).to(tl.int32)
            cv, ci = _unkey(tl.sort(_key(cv, ci), descending=True))
            top_k = tl.load(TOPK + r)
            keep = (q < tl.where(top_k > 0, top_k, WORLD * CAND)) & qm
            scaled = cv.to(tl.float64) / div
            first = tl.max(tl.where(keep, scaled, float("-inf")), 0)          # the first candidate's
            drawn = scaled + _gumbel(tl.load(SEED + r), tl.load(Q0 + r) + 1 + step, ci)
            top_p = tl.load(TOPP + r)
            if top_p < 1.0:                                                   # choose_rows's top_p prefix
                e = tl.where(keep, libdevice.exp(scaled - first), 0.0)
                mass = tl.cumsum(e / tl.sum(e, 0), 0)
                keep = keep & (q < tl.sum(((mass < top_p) & keep).to(tl.int32), 0) + 1)
            keep = keep & (scaled >= first + tl.load(MINL + r))              # min_p (MINL -inf: every candidate)
            drawn = tl.where(keep, drawn, float("-inf"))
            a = tl.argmax(drawn, 0)
            bi = tl.sum(tl.where(q == a, ci, 0), 0)
    tl.store(OUT + r * o_stride, bi.to(tl.int64))


class Markov:
    """The Markov loop of one drafter on one rank: the head rows it scores (all, or its vocabulary half with SPLIT),
    the cached bias rows and the slot of every token (-1: none)."""

    def __init__(self, model, tokens: list[int] | None = None, rows: int | None = None, split: bool | None = None):
        w, c = model.w, model.cfg
        dw = w.dspark
        self.m = model
        self.world, self.rank = w.world, w.rank
        self.V = c.vocab
        self.rank_dim = dw.markov_head.shape[1]
        self.split = (SPLIT if split is None else split) and self.world > 1
        self.n_cols = self.V // self.world                # a segment: the drafter head's vocabulary part
        assert self.n_cols * self.world == self.V and self.n_cols % BN == 0, (self.V, self.world)
        self.segs = 1 if self.split else self.world
        self.seg0 = self.rank if self.split else 0
        self.cols = self.segs * self.n_cols               # the columns this rank scores (cache, staging rows)
        self.head = dw.markov_head.contiguous()
        self.emb = dw.markov_embed.contiguous()
        self.b_tiles = triton.cdiv(self.n_cols, BN * SUB)
        self.s_tiles = triton.cdiv(self.n_cols, BC)
        self.n_part = self.s_tiles * self.segs
        self.bp = triton.next_power_of_2(self.n_part)
        dev = self.head.device
        self.none = torch.full((self.V,), -1, dtype=torch.int32, device=dev)
        self.dummy16 = torch.zeros((1,), dtype=torch.float16, device=dev)
        self.slot = self.none
        self.cache = self.dummy16
        self.tokens: list[int] = []
        rows = CACHE_ROWS if rows is None else int(rows)
        if rows > 0:
            if tokens is None:
                from .markov_tokens import TOKENS as tokens
            tokens = [t for t in dict.fromkeys(int(t) for t in tokens) if 0 <= t < self.V][:rows]
            self.fill(tokens)

    def bias(self, tok: torch.Tensor, t_stride: int, slot: torch.Tensor, out: torch.Tensor, o_row: int, n: int):
        """``_bias`` for n rows (tokens tok[r * t_stride]) into out [n, cols] fp16 (the rows without a slot)."""

        _bias[(self.b_tiles, self.segs)](tok, t_stride, slot, self.emb, self.head, out, o_row, n, self.n_cols,
                                         self.seg0, BN=BN, SUB=SUB, RANK=self.rank_dim, NP=NP, num_warps=4,
                                         **K._pdl())

    def fill(self, tokens: list[int]) -> None:
        """The cached rows of ``tokens``: ``_bias``'s rows (every row computed: the slot table of none)."""

        dev = self.head.device
        K_ = len(tokens)
        cache = torch.empty((max(K_, 1), self.cols), dtype=torch.float16, device=dev)
        tok = torch.tensor(tokens, dtype=torch.int64, device=dev)
        for r0 in range(0, K_, NP):
            self.bias(tok[r0:], 1, self.none, cache[r0], self.cols, min(NP, K_ - r0))
        slot = torch.full((self.V,), -1, dtype=torch.int32, device=dev)
        if K_:
            slot[tok] = torch.arange(K_, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        self.cache, self.slot, self.tokens = cache, slot, tokens

    def logits_of(self, local: torch.Tensor):
        """What the steps read of the drafter head's output ``local`` [R, n_cols] fp32 (this rank's columns): itself
        with SPLIT (no gather), else every rank's columns (one gather, [world, R, n_cols], read in place)."""

        return local if self.split else self.m.comm.gather(local)

    def _scratch(self, N: int, dev):
        sc = self.__dict__.setdefault("_sc", {}).get(N)
        if sc is None:
            sc = (torch.empty((N, self.n_part), dtype=torch.float32, device=dev),
                  torch.empty((N, self.n_part), dtype=torch.int32, device=dev),
                  torch.zeros((N, 4), dtype=torch.float32, device=dev),
                  torch.empty((N, self.cols), dtype=torch.float16, device=dev))
            self._sc[N] = sc
        return sc

    def _coupled_scratch(self, N: int, dev):
        """A coupled step's program candidates [N, n_part, CAND] int64 and its gathered row [N, SEND_W] fp32."""

        sc = self.__dict__.setdefault("_csc", {}).get(N)
        if sc is None:
            sc = (torch.empty((N, self.n_part, CAND), dtype=torch.int64, device=dev),
                  torch.zeros((N, SEND_W), dtype=torch.float32, device=dev))
            self._csc[N] = sc
        return sc

    def couples(self) -> bool:
        """Whether ``steps`` can couple: each rank scores its own vocabulary part (SPLIT, or one rank)."""

        return COUPLED and (self.split or self.world == 1)

    def local_best(self, lg: torch.Tensor, out: torch.Tensor, block: int, i: int, stage: torch.Tensor | None = None,
                   samp=None):
        """Step i on this rank's columns: into out[:, i + 1] (unsplit), else this rank's (value, index) [N, 4] fp32
        for the gather (returned). ``stage``: the staging rows [N, cols] to use (tests read them back). ``samp``
        (``steps``): also the sampled rows' candidates, [N, SEND_W] returned (unsplit too)."""

        N = out.shape[0]
        assert N <= NP and out.stride(1) == 1
        nc = self.n_cols
        pv, pi, send, st = self._scratch(N, out.device)
        st = st if stage is None else stage
        cpl = samp is not None
        if cpl:
            pk, send = self._coupled_scratch(N, out.device)
            div = samp[1]
        else:
            pk, div = pv, pv                               # not read
        tok, ts = out[:, i], out.stride(0)
        self.bias(tok, ts, self.slot, st, self.cols, N)
        base = lg.view(-1)[i * nc:]                        # stream r's step i: + s * R * nc + r * block * nc
        _score[(self.s_tiles, self.segs)](base, N * block * nc, block * nc, tok, ts, self.slot, self.cache, self.cols,
                                          st, self.cols, pv, pi, N, nc, self.seg0, self.n_part, div, pk, BC=BC,
                                          CPL=cpl, CAND=CAND, num_warps=4, **K._pdl())
        sends = self.split or cpl
        _finish[(N,)](pv, pi, self.n_part, out[:, i + 1], ts, send, int(sends), div, pk, BP=self.bp, CPL=cpl,
                      CAND=CAND, BK=triton.next_power_of_2(self.n_part * CAND), SW=send.shape[1], num_warps=4,
                      **K._pdl())
        return send if sends else None

    def pick(self, g: torch.Tensor, out: torch.Tensor, i: int, samp=None) -> None:
        """out[:, i + 1] from every rank's (value, index) [world, N, 4]; ``samp``: every rank's [world, N, SEND_W]
        rows, the sampled rows drawn from the candidates."""

        N = out.shape[0]
        if samp is None:
            _pick[(N,)](g, N, out[:, i + 1], out.stride(0), g, g, g, g, g, g, i, WORLD=g.shape[0], num_warps=1,
                        **K._pdl())
            return
        seed, div, top_k, q0, top_p, min_log = samp
        _pick[(N,)](g, N, out[:, i + 1], out.stride(0), div, seed, top_k, q0, top_p, min_log, i, WORLD=g.shape[0],
                    CPL=True, CAND=CAND, BW=triton.next_power_of_2(g.shape[0] * CAND), SW=SEND_W, num_warps=4, **K._pdl())

    def steps(self, lg: torch.Tensor, out: torch.Tensor, block: int, steps: int, samp=None) -> None:
        """out [N, steps + 1] int64 (out[:, 0] the last tokens) -> out[:, 1:] the drafts; lg: ``logits_of``'s tensor
        for N streams of ``block`` rows each (stream r's step i at row r * block + i). ``samp`` (only if ``couples``):
        (seed [N] int64, divisor [N] float64 (temperature x COUPLED_SCALE; 0: a greedy row), top_k [N] int64 (0: all
        candidates), the last token's position q0 [N] int64, top_p [N] float64 (1: off), ln min_p [N] float64 (-inf:
        off)): the sampled rows' drafts are their keyed draws, with the same launches and the one gather a step."""

        for i in range(steps):
            send = self.local_best(lg, out, block, i, samp=samp)
            if samp is not None:
                self.pick(self.m.comm.gather(send) if self.split else send[None], out, i, samp)
            elif self.split:
                self.pick(self.m.comm.gather(send), out, i)
