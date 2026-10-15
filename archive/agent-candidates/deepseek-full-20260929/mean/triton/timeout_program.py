from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _partial_sum_aligned(x_ptr, partial_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tl.store(partial_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _partial_sum_masked(x_ptr, partial_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(partial_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _finalize_mean(partial_ptr, out_ptr, n_partials, inv_n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    tl.store(out_ptr, total * inv_n)


def build(context):
    partial_cache = {}

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        if dim != 0 or input_tensor.dim() != 1:
            raise NotImplementedError(
                "mean is specialized for the fixed 1D, dim=0 invocation"
            )
        if not input_tensor.is_contiguous():
            raise NotImplementedError("mean requires a contiguous input tensor")
        if dtype is not None and dtype != input_tensor.dtype:
            raise NotImplementedError("mean is specialized to the input dtype")

        n_elements = input_tensor.numel()
        n_blocks = triton.cdiv(n_elements, _BLOCK)

        key = (input_tensor.device, n_elements)
        partial = partial_cache.get(key)
        if partial is None:
            partial = torch.empty(
                (n_blocks,), device=input_tensor.device, dtype=input_tensor.dtype
            )
            partial_cache[key] = partial

        output_dtype = dtype if dtype is not None else input_tensor.dtype
        output = out if out is not None else torch.empty(
            (), device=input_tensor.device, dtype=output_dtype
        )

        if n_elements % _BLOCK == 0:
            _partial_sum_aligned[(n_blocks,)](
                input_tensor, partial, BLOCK=_BLOCK
            )
        else:
            _partial_sum_masked[(n_blocks,)](
                input_tensor, partial, n_elements, BLOCK=_BLOCK
            )

        final_block = triton.next_power_of_2(n_blocks)
        _finalize_mean[(1,)](
            partial,
            output,
            n_blocks,
            1.0 / n_elements,
            BLOCK=final_block,
        )
        return output

    return mean
