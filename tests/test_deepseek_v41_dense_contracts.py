"""Check the shape and storage rules for the dense row kernel before launch."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton", reason="The dense row wrapper requires Triton")

from tensorfold.families.deepseek_v41.cuda import kernels as K


@pytest.mark.parametrize("input_shape,weight_shape", [
    ((128,), (16, 128)), ((1, 128), ()), ((1, 128), (128,)), ((1, 1, 128), (16, 128)),
])
def test_dense_rows_requires_matrices(input_shape, weight_shape):
    """Reject invalid ranks before shape access or a kernel launch."""

    with pytest.raises(ValueError, match="two-dimensional input and weight"):
        K.dense_rows(torch.empty(input_shape), torch.empty(weight_shape))


def test_dense_rows_rejects_input_inner_stride():
    """Reject input views whose columns are not adjacent in storage."""

    inputs = torch.empty((2, 256), dtype=torch.bfloat16)[:, ::2]
    weight = torch.empty((16, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unit input inner stride"):
        K.dense_rows(inputs, weight)


@pytest.mark.parametrize("transpose", [False, True])
def test_dense_rows_rejects_weight_stride(transpose):
    """Reject weight views with a gap between rows or columns."""

    inputs = torch.empty((2, 128), dtype=torch.bfloat16)
    weight = (torch.empty((128, 16), dtype=torch.bfloat16).t() if transpose
              else torch.empty((16, 256), dtype=torch.bfloat16)[:, :128])
    with pytest.raises(ValueError, match="contiguous weight tensor"):
        K.dense_rows(inputs, weight)


def test_dense_rows_requires_cuda():
    """Reject CPU tensors before a kernel launch."""

    with pytest.raises(ValueError, match="same CUDA device"):
        K.dense_rows(torch.empty((2, 128)), torch.empty((16, 128)))
