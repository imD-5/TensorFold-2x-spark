"""DeepSeek-V4.1 prompt indexer: the fine tile maxima (switch "prompt_tile") give the same top-k as the full search."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tensorfold.families.deepseek_v41.cuda import kernels as K  # noqa: E402

IH, ID, TOPK = 32, 128, 512


def _inputs(rows, n, packed, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((rows, IH, ID), generator=g, device="cuda").to(torch.bfloat16)
    if packed:
        codes = torch.randint(0, 256, (n, ID // 2), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        scales = torch.randint(120, 131, (n, ID // 32), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
        k = (codes, scales)
    else:
        k = torch.randn((n, ID), generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn((rows, IH), generator=g, device="cuda").to(torch.bfloat16)
    w[: rows // 4] = 0                          # all-zero scores: the order comes from the index alone (ties)
    vis = torch.randint(1, n + 1, (rows,), generator=g, device="cuda", dtype=torch.int64)
    vis[-1] = n
    return q, k, w, vis


def _tile_max(keys, tile, width):
    rows, n = keys.shape
    pad = torch.full((rows, width * tile), K.PAD_KEY, dtype=torch.int64, device=keys.device)
    pad[:, :n] = keys
    return pad.view(rows, width, tile).amax(dim=2)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("n", [4097, 5000, 12345, 32768, 40000])
def test_fine_tiles_same_selection(n, packed):
    rows = 64
    q, k, w, vis = _inputs(rows, n, packed, seed=n)
    tile = K.prompt_tile(rows, n, TOPK)
    assert tile == K.PROMPT_TILE
    keys, tm = K.index_score(q, k, w, vis, n, keys=True, tmax=True, tile=tile)
    ref_keys = K.index_score(q, k, w, vis, n, keys=True)
    assert torch.equal(keys, ref_keys)                                   # the same scores and keys
    width = K.prompt_tile_width(n, tile)
    assert tm.shape == (rows, width)
    assert torch.equal(tm, _tile_max(keys, tile, width))
    got = K.topk_select_pruned(keys, tm, TOPK, vis, tile)
    want = K.topk_select(keys, TOPK, vis)
    assert torch.equal(got, want)
    # and the reference of the fp32 path: topk_indices of the scores, masked at vis
    score = K.index_score(q, k, w, vis, n)
    top = K.topk_indices(score, TOPK)
    assert torch.equal(got, torch.where(top < vis[:, None], top, -1))


def test_tile_choice():
    assert K.prompt_tile(64, 4096, TOPK) == K.TOPK_PRUNE_BN             # fits topk_select's one launch already
    assert K.prompt_tile(64, 4100, TOPK) == K.PROMPT_TILE
    assert K.prompt_tile(K.DECODE_ROWS, 20000, TOPK) == K.TOPK_PRUNE_BN  # decode-sized blocks: the row kernel
    assert K.prompt_tile_width(4100, 8) == 1024
    assert K.prompt_tile_width(32768, 8) == 4096


@pytest.mark.parametrize("n", [4097, 9000, 16384, 33333])
def test_candidate_scores_same_selection(n):
    """The candidate layers' path: fp32 scores with -inf outside the pool's blocks, keys and 8-key tile maxima from
    score_keys, the tile search: the same as topk_indices masked at vis."""
    rows = 48
    g = torch.Generator(device="cuda").manual_seed(n)
    score = torch.randn((rows, n), generator=g, device="cuda")
    score[: rows // 4] = 0.0                                             # ties: the index decides
    blocks = torch.rand((rows, -(-n // 8)), generator=g, device="cuda") < 0.3
    keep = blocks.repeat_interleave(8, dim=-1)[:, :n]
    score.masked_fill_(~keep, float("-inf"))
    vis = torch.randint(1, n + 1, (rows,), generator=g, device="cuda", dtype=torch.int64)
    vis[-1] = n
    score.masked_fill_(torch.arange(n, device="cuda")[None] >= vis[:, None], float("-inf"))
    tile = K.prompt_tile(rows, n, TOPK)
    assert tile == K.PROMPT_TILE
    keys, tm = K.score_keys(score, tmax=True, tile=tile)
    assert torch.equal(keys, K.score_keys(score))
    width = K.prompt_tile_width(n, tile)
    assert torch.equal(tm, _tile_max(keys, tile, width))
    got = K.topk_select_pruned(keys, tm, TOPK, vis, tile)
    top = K.topk_indices(score, TOPK)
    assert torch.equal(got, torch.where(top < vis[:, None], top, -1))
