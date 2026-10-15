import torch
import triton
import triton.language as tl


@triton.jit
def row_min_dim0(x_ptr, values_ptr, indices_ptr, n_rows, n_cols, stride_row, BLOCK_N: tl.constexpr):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    best = tl.load(x_ptr + cols, mask=mask, other=float("inf"))
    best_idx = tl.zeros([BLOCK_N], dtype=tl.int32)
    for row in tl.range(1, n_rows):
        current = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float("inf"))
        better = current < best
        best = tl.where(better, current, best)
        best_idx = tl.where(better, row, best_idx)

    tl.store(values_ptr + cols, best, mask=mask)
    tl.store(indices_ptr + cols, best_idx, mask=mask)


def build(context):
    def wrapper(input_tensor, dim, keepdim=False):
        n_rows = input_tensor.shape[0]
        n_cols = input_tensor.shape[1]
        if keepdim:
            values = torch.empty((1, n_cols), device=input_tensor.device, dtype=input_tensor.dtype)
            indices = torch.empty((1, n_cols), device=input_tensor.device, dtype=torch.int64)
        else:
            values = torch.empty((n_cols,), device=input_tensor.device, dtype=input_tensor.dtype)
            indices = torch.empty((n_cols,), device=input_tensor.device, dtype=torch.int64)

        row_min_dim0[(triton.cdiv(n_cols, 128),)](
            input_tensor,
            values,
            indices,
            n_rows,
            n_cols,
            input_tensor.stride(0),
            BLOCK_N=128,
        )
        return values, indices

    return wrapper
