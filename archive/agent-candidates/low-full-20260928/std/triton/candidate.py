import torch
import triton
import triton.language as tl


@triton.jit
def _std_partials(x, sums, squares, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    vals = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(sums + pid, tl.sum(vals, axis=0))
    tl.store(squares + pid, tl.sum(vals * vals, axis=0))


@triton.jit
def _std_finish(sums, squares, output, n_partials, n_elements,
                correction: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    total = tl.sum(tl.load(sums + offsets, mask=mask, other=0.0), axis=0)
    total_sq = tl.sum(tl.load(squares + offsets, mask=mask, other=0.0), axis=0)
    mean = total / n_elements
    variance = (total_sq - total * mean) / tl.maximum(1, n_elements - correction)
    variance = tl.maximum(variance, 0.0)
    tl.store(output, tl.sqrt(variance))


def build(context):
    block = 1024

    def wrapper(input, dim=None, correction=1, keepdim=False, out=None):
        n_elements = input.numel()
        n_partials = triton.cdiv(n_elements, block)
        sums = torch.empty((n_partials,), device=input.device, dtype=torch.float32)
        squares = torch.empty_like(sums)
        result = out if out is not None else torch.empty((), device=input.device, dtype=input.dtype)
        _std_partials[(n_partials,)](input, sums, squares, n_elements, BLOCK=block)
        _std_finish[(1,)](sums, squares, result, n_partials, n_elements,
                          correction=correction, BLOCK=triton.next_power_of_2(n_partials))
        return result

    return wrapper
