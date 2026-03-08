import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=3),
    ],
    key=["n_cols"],
)
@triton.jit
def _tiling_softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_input_ptr = input_ptr + row_idx * input_row_stride

    running_max = -float("inf")
    running_sum = 0.0

    start = 0
    while start < n_cols:
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float("inf"))

        tile_max = tl.max(x, axis=0)
        new_max = tl.maximum(running_max, tile_max)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
            tl.exp(x - new_max), axis=0
        )
        running_max = new_max
        start += BLOCK_SIZE

    row_output_ptr = output_ptr + row_idx * output_row_stride
    start = 0
    while start < n_cols:
        col_offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        x = tl.load(row_input_ptr + col_offsets, mask=mask, other=-float("inf"))
        y = tl.exp(x - running_max) / running_sum
        tl.store(row_output_ptr + col_offsets, y, mask=mask)
        start += BLOCK_SIZE


def triton_tiling_softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or (not x.is_cuda) or x.dtype != torch.float32:
        raise ValueError("Expected 2D CUDA float32 tensor")
    if not x.is_contiguous():
        x = x.contiguous()
    rows, cols = x.shape
    y = torch.empty_like(x)

    _tiling_softmax_kernel[(rows,)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        cols,
    )
    return y
