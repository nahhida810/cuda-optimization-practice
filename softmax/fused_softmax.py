import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=4),
    ],
    key=["n_cols"],
)
@triton.jit
def _fused_softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_input_ptr = input_ptr + row_idx * input_row_stride
    x = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float("inf"))

    x = x - tl.max(x, axis=0)
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    row_output_ptr = output_ptr + row_idx * output_row_stride
    tl.store(row_output_ptr + col_offsets, y, mask=mask)


def triton_fused_softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or (not x.is_cuda) or x.dtype != torch.float32:
        raise ValueError("Expected 2D CUDA float32 tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    rows, cols = x.shape
    y = torch.empty_like(x)

    block_size = triton.next_power_of_2(cols)
    _fused_softmax_kernel[(rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        cols,
        BLOCK_SIZE=block_size,
    )
    return y
