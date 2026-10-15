import torch
import triton
import triton.language as tl


@triton.jit
def _sum_partials(x_ptr, partials_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(partials_ptr + pid, total)


@triton.jit
def _sum_partials_final(partials_ptr, output_ptr, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(output_ptr, total)


def build(context):
    def wrapper(input, dim, keepdim=False, *, dtype=None):
        n_elements = input.numel()
        partial_block = 4096
        n_partials = triton.cdiv(n_elements, partial_block)

        partials = torch.empty(
            (n_partials,), device=input.device, dtype=input.dtype
        )
        output = torch.empty((), device=input.device, dtype=input.dtype)

        _sum_partials[(n_partials,)](
            input, partials, n_elements, BLOCK=partial_block, num_warps=8
        )
        _sum_partials_final[(1,)](
            partials, output, n_partials, BLOCK=256, num_warps=8
        )
        return output

    return wrapper
