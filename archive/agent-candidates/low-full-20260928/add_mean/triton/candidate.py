import torch
import triton
import triton.language as tl


@triton.jit
def _partial_mean_kernel(x_ptr, y_ptr, partial_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    value = x + alpha * y
    partial = tl.sum(tl.where(mask, value, 0.0), axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def _final_mean_kernel(partial_ptr, out_ptr, n_partials, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partial = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(tl.where(mask, partial, 0.0), axis=0)
    tl.store(out_ptr, total / n_elements)


def build(context):
    block = 1024

    def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        if dtype is not None:
            input = input.to(dtype)
            if torch.is_tensor(other):
                other = other.to(dtype)
        n_elements = input.numel()
        n_partials = (n_elements + block - 1) // block
        partial = torch.empty((n_partials,), device=input.device, dtype=input.dtype)
        result = out if out is not None else torch.empty((), device=input.device, dtype=input.dtype)
        _partial_mean_kernel[(n_partials,)](input, other, partial, n_elements, alpha, BLOCK=block)
        _final_mean_kernel[(1,)](partial, result, n_partials, n_elements, BLOCK=block)
        return result

    return add_mean
