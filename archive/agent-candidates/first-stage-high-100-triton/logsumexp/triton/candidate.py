import torch
import triton
import triton.language as tl


@triton.jit
def _partial_max_kernel(x_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    partial = tl.max(values, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def _reduce_max_kernel(partial_ptr, max_ptr, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partial_ptr + offsets, mask=mask, other=-float("inf"))
    maximum = tl.max(values, axis=0)
    tl.store(max_ptr, maximum)


@triton.jit
def _partial_sum_kernel(
    x_ptr, max_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    maximum = tl.load(max_ptr)
    values = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))
    terms = tl.exp(values - maximum)
    terms = tl.where(mask, terms, 0.0)
    partial = tl.sum(terms, axis=0)
    tl.store(partial_ptr + pid, partial)


@triton.jit
def _final_sum_kernel(partial_ptr, max_ptr, output_ptr, n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    maximum = tl.load(max_ptr)
    tl.store(output_ptr, maximum + tl.log(total))


def build(context):
    def wrapper(input, dim, keepdim=False, *, out=None):
        n_elements = input.numel()
        block = 4096
        n_partials = triton.cdiv(n_elements, block)

        partial_max = torch.empty(
            (n_partials,), device=input.device, dtype=torch.float32
        )
        maximum = torch.empty((), device=input.device, dtype=torch.float32)
        partial_sum = torch.empty(
            (n_partials,), device=input.device, dtype=torch.float32
        )

        _partial_max_kernel[(n_partials,)](
            input, partial_max, n_elements, BLOCK=block, num_warps=8
        )
        max_block = triton.next_power_of_2(n_partials)
        _reduce_max_kernel[(1,)](
            partial_max, maximum, n_partials, BLOCK=max_block, num_warps=1
        )
        _partial_sum_kernel[(n_partials,)](
            input, maximum, partial_sum, n_elements, BLOCK=block, num_warps=8
        )

        if out is None:
            if keepdim:
                output = torch.empty((1,), device=input.device, dtype=input.dtype)
            else:
                output = torch.empty((), device=input.device, dtype=input.dtype)
        else:
            output = out

        _final_sum_kernel[(1,)](
            partial_sum,
            maximum,
            output,
            n_partials,
            BLOCK=max_block,
            num_warps=1,
        )
        return output

    return wrapper
