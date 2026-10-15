import torch
import triton
import triton.language as tl


@triton.jit
def _exp_sum_kernel(
    input_ptr,
    partial_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    values = tl.exp(values)
    tl.store(partial_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _mean_partials_kernel(
    partial_ptr,
    output_ptr,
    n_partials,
    inv_n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    partials = tl.load(partial_ptr + offsets, mask=offsets < n_partials, other=0.0)
    total = tl.sum(partials, axis=0)
    tl.store(output_ptr, total * inv_n_elements)


def build(context):
    del context

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        if dim is not None or keepdim:
            raise ValueError("the supplied profile only supports dim=None and keepdim=False")
        if dtype is not None and dtype != input.dtype:
            raise ValueError("the supplied profile only supports the input dtype")

        n_elements = input.numel()
        block = 1024
        n_partials = triton.cdiv(n_elements, block)
        partials = torch.empty((n_partials,), device=input.device, dtype=torch.float32)
        output = out if out is not None else torch.empty((), device=input.device, dtype=input.dtype)

        _exp_sum_kernel[(n_partials,)](
            input,
            partials,
            n_elements,
            BLOCK=block,
            num_warps=8,
        )
        _mean_partials_kernel[(1,)](
            partials,
            output,
            n_partials,
            1.0 / n_elements,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return exp_mean
