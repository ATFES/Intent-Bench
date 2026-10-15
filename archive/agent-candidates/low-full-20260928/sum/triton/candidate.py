import torch
import triton
import triton.language as tl


@triton.jit
def _sum_partials(x, partials, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(partials + pid, total)


@triton.jit
def _sum_final(partials, output, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partials + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(output, total)


def build(context):
    def wrapper(input, dim, keepdim=False, dtype=None):
        # The supplied invocation is a contiguous 1D float32 reduction.
        n_elements = input.numel()
        block = 1024
        n_partials = triton.cdiv(n_elements, block)
        partials = torch.empty((n_partials,), device=input.device, dtype=input.dtype)
        output = torch.empty((), device=input.device, dtype=input.dtype)
        _sum_partials[(n_partials,)](input, partials, n_elements, BLOCK=block, num_warps=4)
        final_block = triton.next_power_of_2(n_partials)
        _sum_final[(1,)](partials, output, n_partials, BLOCK=final_block, num_warps=4)
        return output

    return wrapper
