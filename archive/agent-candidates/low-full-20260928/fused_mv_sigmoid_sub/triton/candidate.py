import torch
import triton
import triton.language as tl


@triton.jit
def _mv_sigmoid_sub_scalar(a_ptr, v_ptr, out_ptr, n_cols, other, alpha, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    a = tl.load(a_ptr + row * n_cols + cols, mask=mask, other=0.0)
    v = tl.load(v_ptr + cols, mask=mask, other=0.0)
    z = tl.sum(a * v, axis=0)
    s = 1.0 / (1.0 + tl.exp(-z))
    tl.store(out_ptr + row, s - alpha * other)


@triton.jit
def _mv_sigmoid_sub_tensor(a_ptr, v_ptr, other_ptr, out_ptr, n_cols, other_numel, alpha,
                           BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    a = tl.load(a_ptr + row * n_cols + cols, mask=mask, other=0.0)
    v = tl.load(v_ptr + cols, mask=mask, other=0.0)
    z = tl.sum(a * v, axis=0)
    s = 1.0 / (1.0 + tl.exp(-z))
    b = tl.load(other_ptr + tl.where(other_numel == 1, 0, row))
    tl.store(out_ptr + row, s - alpha * b)


def build(context):
    def wrapper(input, vec, other, alpha=1, *, out=None):
        n_rows, n_cols = input.shape
        result = out if out is not None else torch.empty((n_rows,), device=input.device, dtype=input.dtype)
        grid = (n_rows,)
        if isinstance(other, torch.Tensor):
            _mv_sigmoid_sub_tensor[grid](
                input, vec, other, result, n_cols, other.numel(), alpha,
                BLOCK=triton.next_power_of_2(n_cols), num_warps=8,
            )
        else:
            _mv_sigmoid_sub_scalar[grid](
                input, vec, result, n_cols, other, alpha,
                BLOCK=triton.next_power_of_2(n_cols), num_warps=8,
            )
        return result

    return wrapper
