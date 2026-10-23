"""Test dense and wider EXL3 overrides for DeepSeek-V4.1 weights."""

from __future__ import annotations

import json
import math
import struct
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

W = pytest.importorskip("tensorfold.families.deepseek_v41.cuda.weights")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

ST = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int16: "I16",
      torch.float8_e4m3fn: "F8_E4M3"}


def _write(path, tensors: dict) -> None:
    """Write CPU tensors in the safetensors format used by this test."""

    header, blobs, at = {}, [], 0
    for name, tensor in tensors.items():
        blob = tensor.contiguous().view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
        header[name] = {"dtype": ST[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [at, at + len(blob)]}
        blobs.append(blob)
        at += len(blob)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def _group(k: int, n: int, bits: int, seed: int, mul1: bool = True) -> dict:
    generator = torch.Generator().manual_seed(seed)
    group = {
        "trellis": torch.randint(-32768, 32768, (k // 16, n // 16, 16 * bits), generator=generator).to(torch.int16),
        "suh": torch.rand(k, generator=generator).half(),
        "svh": torch.rand(n, generator=generator).half(),
    }
    if mul1:
        group["mul1"] = torch.zeros(1, dtype=torch.int16)
    return group


def _named(prefix: str, group: dict) -> dict:
    return {f"{prefix}.{key}": value for key, value in group.items()}


def _exl3_bits(shards, prefix: str) -> int:
    """Read the EXL3 bit width from the test checkpoint header."""

    entry = shards.entries.get(f"{prefix}.trellis")
    return entry[2]["shape"][-1] // 16 if entry is not None else 0


def test_override_replaces_all_old_group_parts(tmp_path):
    """An override removes stale EXL3 parts and codebook markers from its group."""

    base = _named("layers.10.attn.wo_b", _group(256, 128, 5, 1))
    base["layers.10.attn_norm.weight"] = torch.rand(256).to(torch.bfloat16)
    _write(tmp_path / "model-00001-of-00002.safetensors", base)
    wide = _group(256, 128, 8, 3, mul1=False)
    dense = torch.rand(128, 256).to(torch.bfloat16)
    _write(tmp_path / "override-dealign.safetensors", _named("layers.10.attn.wo_b", wide)
           | {"layers.10.ffn.shared_experts.w2.weight": dense})

    shards = W.Shards(tmp_path)
    assert _exl3_bits(shards, "layers.10.attn.wo_b") == 8
    assert not W.is_dense(shards, "layers.10.attn.wo_b")
    assert "layers.10.attn.wo_b.mul1" not in shards
    assert W._codebook(shards, "layers.10.attn.wo_b") == "3inst"
    assert torch.equal(shards.get("layers.10.attn.wo_b.trellis"), wide["trellis"])
    assert W.is_dense(shards, "layers.10.ffn.shared_experts.w2")
    assert _exl3_bits(shards, "layers.10.ffn.shared_experts.w2") == 0
    assert torch.equal(shards.get("layers.10.ffn.shared_experts.w2.weight"), dense)
    assert torch.equal(shards.get("layers.10.attn_norm.weight"), base["layers.10.attn_norm.weight"])
    shards.close()


def test_multiple_overrides_keep_neighbors_and_replace_nested_groups(tmp_path):
    """Later overrides remove old group members and retain adjacent names."""

    prefix = "layers.10.attn.wo_b"
    base = _named(prefix, _group(256, 128, 5, 1))
    neighbor = _named(prefix + "_next", _group(256, 128, 5, 2))
    base.update(neighbor)
    base[prefix + ".child.weight"] = torch.ones(1, dtype=torch.bfloat16)
    base["layers.10.attn_norm.weight"] = torch.ones(128, dtype=torch.bfloat16)
    _write(tmp_path / "zz-model.safetensors", base)
    _write(tmp_path / "override-01.safetensors",
           {prefix + ".weight": torch.ones((128, 256), dtype=torch.bfloat16)})
    wide = _named(prefix, _group(256, 128, 8, 3, mul1=False))
    _write(tmp_path / "override-02.safetensors", wide)

    shards = W.Shards(tmp_path)
    expected = set(wide) | set(neighbor) | {"layers.10.attn_norm.weight"}
    assert set(shards.entries) == expected
    assert _exl3_bits(shards, prefix) == 8
    assert W._codebook(shards, prefix) == "3inst"
    for name, tensor in wide.items():
        assert torch.equal(shards.get(name), tensor)
    for name, tensor in neighbor.items():
        assert torch.equal(shards.get(name), tensor)
    assert torch.equal(shards.get("layers.10.attn_norm.weight"), base["layers.10.attn_norm.weight"])
    shards.close()


@pytest.mark.parametrize("codebook", ["mul1", "mcg"])
def test_base_shards_keep_split_exl3_parts_and_gate_bias(tmp_path, codebook):
    """Base shards can store one EXL3 group in separate files."""

    prefix = "layers.10.attn.wo_b"
    group = _named(prefix, _group(256, 128, 5, 1))
    if codebook == "mcg":
        group[prefix + ".mcg"] = group.pop(prefix + ".mul1")
    gate = "layers.10.ffn.gate"
    first = {name: tensor for name, tensor in group.items() if not name.endswith(".trellis")}
    first[gate + ".bias"] = torch.ones(8, dtype=torch.float32)
    second = {prefix + ".trellis": group[prefix + ".trellis"],
              gate + ".weight": torch.ones((8, 256), dtype=torch.bfloat16)}
    _write(tmp_path / "model-01.safetensors", first)
    _write(tmp_path / "model-02.safetensors", second)

    shards = W.Shards(tmp_path)
    assert set(shards.entries) == set(first) | set(second)
    assert W._codebook(shards, prefix) == codebook
    for name, tensor in (first | second).items():
        assert torch.equal(shards.get(name), tensor)
    shards.close()


@pytest.mark.parametrize("source,dtype", [("model-00", torch.float8_e4m3fn),
                                         ("model-02", torch.float8_e4m3fn),
                                         ("override-fp8", torch.float8_e4m3fn),
                                         ("override-bf16", torch.bfloat16)])
def test_scaled_engram_keeps_exl3_parts(tmp_path, source, dtype):
    """A scaled source beside an EXL3 group does not select the dense path."""

    prefix = "layers.10.engram.wkv"
    group = _named(prefix, _group(256, 128, 5, 1))
    _write(tmp_path / "model-01.safetensors", group)
    _write(tmp_path / f"{source}.safetensors",
           {prefix + ".weight": torch.ones((128, 256)).to(dtype),
            prefix + ".scale": torch.ones((1, 2), dtype=torch.float32)})

    shards = W.Shards(tmp_path)
    assert not W.is_dense(shards, prefix)
    assert W._codebook(shards, prefix) == "mul1"
    trellis, suh, svh, codebook = W.exl3_parts(shards, prefix, device=None)
    assert codebook == "mul1"
    for part, value in zip(("trellis", "suh", "svh"), (trellis, suh, svh)):
        assert torch.equal(value, group[f"{prefix}.{part}"])
    assert prefix + ".weight" in shards and prefix + ".scale" in shards
    shards.close()


@pytest.mark.parametrize("dtype,shape,extra", [
    (torch.float8_e4m3fn, (128, 256), None),
    (torch.float16, (128, 256), None),
    (torch.bfloat16, (256,), None),
    (torch.bfloat16, (128, 256), "scale"),
    (torch.bfloat16, (128, 256), "trellis"),
])
def test_dense_requires_one_bf16_matrix(tmp_path, dtype, shape, extra):
    """Only a BF16 matrix without EXL3 or scale parts selects the dense path."""

    prefix = "layers.10.attn.wo_b"
    tensors = {prefix + ".weight": torch.ones(shape).to(dtype)}
    if extra is not None:
        tensors[prefix + "." + extra] = torch.ones(1, dtype=torch.int16)
    _write(tmp_path / "model.safetensors", tensors)

    shards = W.Shards(tmp_path)
    assert not W.is_dense(shards, prefix)
    shards.close()


def test_bf16_override_removes_stale_exl3_parts(tmp_path):
    """A BF16 override removes every old EXL3 part and scale."""

    prefix = "layers.10.engram.wkv"
    base = _named(prefix, _group(256, 128, 5, 1))
    base[prefix + ".weight"] = torch.ones((128, 256)).to(torch.float8_e4m3fn)
    base[prefix + ".scale"] = torch.ones((1, 2), dtype=torch.float32)
    _write(tmp_path / "zz-model.safetensors", base)
    weight = torch.full((128, 256), 2, dtype=torch.bfloat16)
    _write(tmp_path / "override-bf16.safetensors", {prefix + ".weight": weight})

    shards = W.Shards(tmp_path)
    assert set(shards.entries) == {prefix + ".weight"}
    assert W.is_dense(shards, prefix)
    assert torch.equal(shards.get(prefix + ".weight"), weight)
    shards.close()


@pytest.mark.parametrize("prefix,shape", [("layers.10.attn_norm", (256,)),
                                          ("layers.10.ffn.gate", (8, 256))])
def test_plain_weight_override_keeps_bias(tmp_path, prefix, shape):
    """A norm or gate weight override keeps the adjacent bias."""

    bias = torch.ones(shape[0], dtype=torch.float32)
    _write(tmp_path / "model.safetensors",
           {prefix + ".weight": torch.ones(shape, dtype=torch.bfloat16), prefix + ".bias": bias})
    weight = torch.full(shape, 2, dtype=torch.bfloat16)
    _write(tmp_path / "override-plain.safetensors", {prefix + ".weight": weight})

    shards = W.Shards(tmp_path)
    assert torch.equal(shards.get(prefix + ".weight"), weight)
    assert torch.equal(shards.get(prefix + ".bias"), bias)
    shards.close()


@pytest.fixture
def comm_type():
    """Import the model communicator when its CUDA module dependencies are present."""

    pytest.importorskip("triton")
    from tensorfold.families.deepseek_v41.cuda.model import Comm

    return Comm


def test_comm_sum_result_survives_gather_buffer_reuse(comm_type):
    """A sum result stays valid after a later gather overwrites its buffer."""

    gathered = torch.tensor([[[1.0, 2.0]], [[3.0, 5.0]]])
    comm = comm_type.__new__(comm_type)
    comm.world = 2
    comm.gather = lambda _: gathered
    result = comm.sum(torch.empty(0))
    assert torch.equal(result, torch.tensor([[4.0, 7.0]]))
    assert result.data_ptr() != gathered[0].data_ptr()
    gathered.fill_(-1)
    assert torch.equal(result, torch.tensor([[4.0, 7.0]]))


def test_comm_sum_uses_in_place_all_reduce_for_two_ranks(comm_type):
    """A two-rank sum uses the input buffer when the communicator supports all-reduce."""

    value = torch.tensor([[1.0, 2.0]])
    calls = []

    def all_reduce(tensor):
        calls.append(tensor.data_ptr())
        tensor.add_(torch.tensor([[3.0, 5.0]]))

    comm = comm_type.__new__(comm_type)
    comm.world = 2
    comm.nccl = SimpleNamespace(all_reduce=all_reduce)
    result = comm.sum(value)
    assert calls == [value.data_ptr()]
    assert result.data_ptr() == value.data_ptr()
    assert torch.equal(result, torch.tensor([[4.0, 7.0]]))


@pytest.mark.parametrize("fits", [True, False])
def test_comm_sum_keeps_rdma_for_payloads_that_fit(fits, comm_type):
    """Small sums use the RDMA gather. Larger sums use all-reduce."""

    value = torch.tensor([[1.0, 2.0]])
    calls = []

    def all_reduce(tensor):
        calls.append("reduce")
        tensor.add_(torch.tensor([[3.0, 5.0]]))

    def gather(tensor):
        calls.append("gather")
        return torch.stack([tensor, torch.tensor([[3.0, 5.0]])])

    comm = comm_type.__new__(comm_type)
    comm.world = 2
    comm.nccl = SimpleNamespace(all_reduce=all_reduce, rdma=SimpleNamespace(fits=lambda _: fits))
    comm.gather = gather
    result = comm.sum(value)
    assert calls == ["gather" if fits else "reduce"]
    assert torch.equal(result, torch.tensor([[4.0, 7.0]]))
    if fits:
        assert torch.equal(value, torch.tensor([[1.0, 2.0]]))
    else:
        assert result.data_ptr() == value.data_ptr()


@cuda
@pytest.mark.parametrize("rows", [1, 7, 16])
@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("fold", [False, True])
def test_dense_wo_ab_accepts_plain_and_rotated_inputs(rows, rotated, fold):
    """A dense output projection accepts both decode input paths."""

    from tensorfold.cuda.exl3.linear import Exl3Linear
    from tensorfold.families.deepseek_v41.cuda import model as M

    wo_a = []
    for seed in (21, 22):
        group = _group(512, 256, 4, seed, mul1=False)
        wo_a.append(Exl3Linear.from_tensors(group["trellis"].cuda(), group["suh"].cuda(),
                                          group["svh"].cuda(), "3inst"))
    generator = torch.Generator(device="cuda").manual_seed(23)
    dense = W.DenseLinear(torch.randn((512, 512), generator=generator, device="cuda").to(torch.bfloat16))
    lay = SimpleNamespace(wo_a=wo_a, wo_b=dense)
    inputs = torch.randn((rows, 2, 512), generator=generator, device="cuda").to(torch.bfloat16)
    reference = M.mm(dense, M.wo_a_out(lay, inputs), torch.float32)
    if rotated:
        ga = M._group(lay, "_tf_wo_a", lay.wo_a)
        xh = ga.rotate([inputs[:, group] for group in range(2)])
        result = M.wo_ab(lay, xh=xh, fold=fold)
    else:
        result = M.wo_ab(lay, inputs, fold=fold)
    assert torch.equal(result, reference)
    assert not hasattr(lay, "_tf_wo_b")


@cuda
@pytest.mark.parametrize("k,n", [(4096, 5120), (1152, 5120), (256, 16)])
def test_dense_rows_are_row_invariant(k, n):
    """Dense decode rows agree with the FP64 product and do not depend on row count."""

    from tensorfold.families.deepseek_v41.cuda import kernels as K

    generator = torch.Generator(device="cuda").manual_seed(k + n)
    weight = (torch.randn((n, k), generator=generator, device="cuda") / math.sqrt(k)).to(torch.bfloat16)
    inputs = torch.randn((128, k), generator=generator, device="cuda").to(torch.bfloat16)
    full = K.dense_rows(inputs, weight)
    reference = inputs.double() @ weight.double().t()
    assert (full.double() - reference).abs().max() <= 1e-4 * reference.abs().max() + 1e-6
    for rows in (1, 2, 7, 16, 17, 64):
        for start in (0, 5):
            assert torch.equal(K.dense_rows(inputs[start:start + rows], weight), full[start:start + rows])
    padded = torch.zeros((4, 2 * k), dtype=torch.bfloat16, device="cuda")
    padded[:, :k] = inputs[:4]
    assert torch.equal(K.dense_rows(padded[:, :k], weight), full[:4])


@cuda
@pytest.mark.parametrize("rows", [1, 16, 128, 129, 600])
@pytest.mark.parametrize("out_dtype", ["f32", "bf16"])
def test_dense_mm_paths(rows, out_dtype):
    """The decode and prompt paths accept one dense projection."""

    from tensorfold.families.deepseek_v41.cuda import model as M

    dtype = torch.float32 if out_dtype == "f32" else torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(rows)
    weight = (torch.randn((5120, 1152), generator=generator, device="cuda") / 34).to(torch.bfloat16)
    inputs = torch.randn((rows, 1152), generator=generator, device="cuda").to(torch.bfloat16)
    output = M.mm(W.DenseLinear(weight), inputs, dtype)
    reference = inputs.double() @ weight.double().t()
    tolerance = (1e-4 if dtype == torch.float32 and rows <= 128 else 1e-2) * reference.abs().max()
    assert output.dtype == dtype and output.shape == (rows, 5120)
    assert (output.double() - reference).abs().max() <= tolerance


@cuda
def test_dense_loads_this_ranks_input_columns(tmp_path):
    """A dense down projection keeps only the columns assigned to this TP rank."""

    weight = torch.randn((512, 1024)).to(torch.bfloat16)
    _write(tmp_path / "model-00001-of-00001.safetensors", {"layers.12.attn.wo_b.weight": weight})
    shards = W.Shards(tmp_path)
    for rank in (0, 1):
        linear = W.linear(shards, "layers.12.attn.wo_b", rows=(rank * 512, (rank + 1) * 512))
        assert isinstance(linear, W.DenseLinear) and (linear.n, linear.k) == (512, 512)
        assert linear.weight.is_cuda and linear.weight.is_contiguous()
        assert torch.equal(linear.weight.cpu(), weight[:, rank * 512:(rank + 1) * 512])
    shards.close()


E, D, I, TOPK = 32, 512, 256, 6


def _mats(seed: int, shared_down_k2: int):
    generator = torch.Generator().manual_seed(seed)

    def matrix(k, n, k2, scale):
        trellis = torch.randint(-32768, 32768, (k // 16, n // 16, 8 * k2), generator=generator,
                                dtype=torch.int32).to(torch.int16)
        sign = torch.randint(0, 2, (k,), generator=generator).float() * 2 - 1
        suh = (sign * (torch.rand(k, generator=generator) + 0.5) / math.sqrt(k)).half()
        svh = ((torch.rand(n, generator=generator) + 0.5) * scale).half()
        return trellis.cuda(), suh.cuda(), svh.cuda()

    gate = [matrix(D, I, 8, 1.0) for _ in range(E + 1)]
    up = [matrix(D, I, 8, 1.0) for _ in range(E + 1)]
    down = [matrix(I, D, 8, 0.25) for _ in range(E)] + [matrix(I, D, shared_down_k2, 0.25)]
    return gate, up, down


def _model():
    from tensorfold.families.deepseek_v41.cuda import model as M

    model = M.Model.__new__(M.Model)
    model.cfg = SimpleNamespace(topk=TOPK, route_scale=2.5, swiglu_limit=10.0)
    model.scratch = {}
    return model


def _layers(shared_down_k2: int, seed: int = 11):
    from tensorfold.cuda.exl3 import experts
    from tensorfold.cuda.exl3.linear import Exl3Linear

    gate, up, down = _mats(seed, shared_down_k2)
    generator = torch.Generator().manual_seed(seed + 1)
    common = {"gate_w": (torch.randn((E, D), generator=generator) / math.sqrt(D)).half().cuda(),
              "gate_b": (torch.randn(E, generator=generator) * 0.1).float().cuda(), "hc_ffn": None}
    packed = SimpleNamespace(experts=experts.prepare(gate, up, down, "3inst"), **common)
    shared = tuple(Exl3Linear.from_tensors(*matrix, "3inst") for matrix in (gate[E], up[E], down[E]))
    apart = SimpleNamespace(experts=experts.prepare(gate[:E], up[:E], down[:E], "3inst"), shared=shared, **common)
    return packed, apart, shared


def _close(left, right, relative):
    return (left.double() - right.double()).abs().max() <= relative * right.double().abs().max()


@cuda
@pytest.mark.parametrize("rows", [1, 4, 16, 200])
def test_shared_apart_matches_packed_shared_expert(rows):
    """A compatible separate shared expert gives the packed result."""

    packed, apart, _ = _layers(8)
    inputs = torch.randn((rows, D), generator=torch.Generator().manual_seed(rows)).to(torch.bfloat16).cuda()
    model = _model()
    assert _close(model.moe(apart, inputs), model.moe(packed, inputs), 2e-2)


@cuda
def test_shared_apart_accepts_a_wide_or_dense_down_projection():
    """A wider EXL3 shared down projection and its BF16 form have the same result within BF16 rounding."""

    from tensorfold.families.deepseek_v41.cuda import model as M

    _, apart, shared = _layers(16)
    inputs = torch.randn((8, D), generator=torch.Generator().manual_seed(5)).to(torch.bfloat16).cuda()
    model = _model()
    wide = model._shared_apart(apart, inputs)
    down = shared[2]
    dense_weight = M.dequant_fp32(None, down.suh, down.svh, down.k, down.n, layer=down).t().contiguous().to(
        torch.bfloat16)
    dense = SimpleNamespace(**{**vars(apart), "shared": (shared[0], shared[1], W.DenseLinear(dense_weight))})
    assert _close(model._shared_apart(dense, inputs), wide, 2e-2)
