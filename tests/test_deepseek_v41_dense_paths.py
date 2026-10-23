"""Test the dense decode tiles, the dense prompt GEMM, and the separate shared-expert launches."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton", reason="The DeepSeek-V4.1 CUDA model requires Triton")

from tensorfold.families.deepseek_v41.cuda import kernels as K  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import model as M  # noqa: E402
from tensorfold.families.deepseek_v41.cuda import weights as W  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def test_moe_prefetch_reads_a_separate_shared_expert():
    """The MoE prefetch keeps the separate shared expert's weights in kernel order."""

    w1, w3 = SimpleNamespace(words=torch.zeros(4)), SimpleNamespace(words=torch.zeros(5))
    w2 = W.DenseLinear(torch.zeros((2, 3)))
    lay = SimpleNamespace(hc_ffn=(torch.zeros(1),), gate_w=torch.zeros(2), shared=(w1, w3, w2),
                          experts=SimpleNamespace(count=2, keep=[torch.zeros(1)] * 6))
    got = M._l2_tensors(None, lay, "moe")
    assert [t.data_ptr() for t in got] == [t.data_ptr() for t in
                                           (lay.hc_ffn[0], lay.gate_w, w1.words, w3.words, w2.weight)]


@cuda
@pytest.mark.parametrize("n", [5120, 16])
def test_dense_rows_tiles_do_not_depend_on_neighbor_rows(n):
    """Each row of a tile has the bits of the same row alone, for rows on both sides of a tile edge."""

    generator = torch.Generator(device="cuda").manual_seed(n)
    weight = (torch.randn((n, 1152), generator=generator, device="cuda") / 34).to(torch.bfloat16)
    inputs = torch.randn((40, 1152), generator=generator, device="cuda").to(torch.bfloat16)
    full = K.dense_rows(inputs, weight)
    for row in (0, 15, 16, 17, 31, 32, 39):
        assert torch.equal(K.dense_rows(inputs[row:row + 1], weight), full[row:row + 1])
    noisy = inputs.clone()
    noisy[1:] = noisy[1:] * 1000
    assert torch.equal(K.dense_rows(noisy, weight)[0], full[0])


@cuda
def test_dense_rows_requires_bf16():
    """Reject FP32 rows instead of rounding them to BF16."""

    weight = torch.zeros((16, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="BF16 input and weight"):
        K.dense_rows(torch.zeros((2, 128), device="cuda"), weight)


@cuda
def test_dense_prompt_gemm_makes_no_fp32_weight_copy():
    """A prompt chunk allocates its output, not an FP32 copy of the weight."""

    weight = torch.randn((5120, 4096), device="cuda").to(torch.bfloat16)
    inputs = torch.randn((600, 4096), device="cuda").to(torch.bfloat16)
    M.mm(W.DenseLinear(weight), inputs, torch.float32)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = M.mm(W.DenseLinear(weight), inputs, torch.float32)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    assert extra < weight.numel() * 4 // 2
    reference = inputs.double() @ weight.double().t()
    assert (out.double() - reference).abs().max() <= 1e-4 * reference.abs().max()


@cuda
@pytest.mark.parametrize("limit", [10.0, float("inf")])
def test_clamped_swiglu_has_the_torch_bits(limit):
    """The fused activation gives the bits of the torch expression it replaces."""

    generator = torch.Generator(device="cuda").manual_seed(3)
    g = (torch.randn((37, 1152), generator=generator, device="cuda") * 8).to(torch.bfloat16)
    u = (torch.randn((37, 1152), generator=generator, device="cuda") * 8).to(torch.bfloat16)
    reference = (torch.nn.functional.silu(g.float().clamp(max=limit))
                 * u.float().clamp(-limit, limit)).to(torch.bfloat16)
    assert torch.equal(K.clamped_swiglu(g, u, limit), reference)


@cuda
@pytest.mark.parametrize("rows", [1, 4, 128, 200])
def test_shared_apart_grouped_launches_keep_the_bits(rows, monkeypatch):
    """The grouped gate/up launch and the fused activation give the bits of separate mm calls."""

    from tensorfold.cuda.exl3.linear import Exl3Linear

    generator = torch.Generator().manual_seed(rows)

    def exl3(k: int, n: int) -> Exl3Linear:
        trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 16 * 3), generator=generator).to(torch.int16)
        suh = (torch.rand(k, generator=generator) + 0.5).half()
        svh = (torch.rand(n, generator=generator) * 0.02).half()
        return Exl3Linear.from_tensors(trellis.cuda(), suh.cuda(), svh.cuda(), "3inst")

    down = (torch.randn((512, 256), generator=generator) / 16).to(torch.bfloat16).cuda()
    lay = SimpleNamespace(shared=(exl3(512, 256), exl3(512, 256), W.DenseLinear(down)))
    model = M.Model.__new__(M.Model)
    model.cfg = SimpleNamespace(swiglu_limit=10.0)
    inputs = torch.randn((rows, 512), generator=generator).to(torch.bfloat16).cuda()
    fast = model._shared_apart(lay, inputs)
    monkeypatch.setattr(M, "KERNELS", False)
    w1, w3, w2 = lay.shared
    g, u = M.mm(w1, inputs), M.mm(w3, inputs)
    h = (torch.nn.functional.silu(g.float().clamp(max=10.0)) * u.float().clamp(-10.0, 10.0)).to(torch.bfloat16)
    monkeypatch.setattr(M, "KERNELS", True)
    assert torch.equal(fast, M.mm(w2, h, torch.float32))
