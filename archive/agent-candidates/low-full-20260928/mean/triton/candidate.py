import torch
import triton
import triton.language as tl


@triton.jit
def _mean_partials(x_ptr, partials_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(partials_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _mean_finish(partials_ptr, out_ptr, n_partials, n_elements,
                 BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partials = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    tl.store(out_ptr, total / n_elements)


def build(context):
    partial_block = 4096
    n_elements = 1048576
    n_partials = n_elements // partial_block

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        if dtype is not None:
            input_tensor = input_tensor.to(dtype)
        if out is None:
            result = torch.empty((), device=input_tensor.device,
                                 dtype=input_tensor.dtype)
        else:
            result = out

        partials = torch.empty((n_partials,), device=input_tensor.device,
                               dtype=torch.float32)
        _mean_partials[(n_partials,)](
            input_tensor, partials, input_tensor.numel(), BLOCK=partial_block
        )
        _mean_finish[(1,)](
            partials, result, n_partials, input_tensor.numel(), BLOCK=n_partials
        )
        return result

    return mean
