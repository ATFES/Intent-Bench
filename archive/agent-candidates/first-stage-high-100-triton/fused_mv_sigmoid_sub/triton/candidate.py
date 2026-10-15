import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["n_rows", "n_cols"],
)
@triton.jit
def _fused_mv_sigmoid_sub_scalar(
    input_ptr,
    vec_ptr,
    output_ptr,
    other,
    alpha,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, n_cols, BLOCK_K):
        a = tl.load(
            input_ptr + row[:, None] * n_cols + k + col[None, :],
            mask=(row[:, None] < n_rows) & (k + col[None, :] < n_cols),
            other=0.0,
        )
        v = tl.load(vec_ptr + k + col, mask=k + col < n_cols, other=0.0)
        acc += tl.sum(a * v[None, :], axis=1)

    sigmoid = 1.0 / (1.0 + tl.exp(-acc))
    result = sigmoid - alpha * other
    tl.store(output_ptr + row, result, mask=row < n_rows)


def build(context):
    def wrapper(input, vec, other, alpha=1, *, out=None):
        if out is None:
            out = torch.empty(
                (input.shape[0],), device=input.device, dtype=input.dtype
            )

        n_rows = input.shape[0]
        n_cols = input.shape[1]
        grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]),)
        _fused_mv_sigmoid_sub_scalar[grid](
            input,
            vec,
            out,
            other,
            alpha,
            n_rows,
            n_cols,
        )
        return out

    return wrapper
