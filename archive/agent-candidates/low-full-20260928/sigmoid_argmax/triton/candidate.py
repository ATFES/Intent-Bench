from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_argmax_2d(
    x_ptr,
    out_ptr,
    n0,
    n1,
    stride0,
    stride1,
    reduce_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    if reduce_dim == 1:
        valid = offsets < n1
        x = tl.load(x_ptr + pid * stride0 + offsets * stride1, mask=valid, other=-float("inf"))
    else:
        valid = offsets < n0
        x = tl.load(x_ptr + offsets * stride0 + pid * stride1, mask=valid, other=-float("inf"))

    # Sigmoid is monotonic, but evaluating it here preserves the requested stage
    # and gives the same tie behavior as the reduced values.
    values = 1.0 / (1.0 + tl.exp(-x))
    values = tl.where(valid, values, -float("inf"))
    index = tl.argmax(values, axis=0)
    tl.store(out_ptr + pid, index.to(tl.int64))


@triton.jit
def _sigmoid_argmax_flat(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    valid = offsets < n
    x = tl.load(x_ptr + offsets, mask=valid, other=-float("inf"))
    values = 1.0 / (1.0 + tl.exp(-x))
    values = tl.where(valid, values, -float("inf"))
    index = tl.argmax(values, axis=0)
    tl.store(out_ptr, index.to(tl.int64))


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def build(context):
    def sigmoid_argmax(input, dim=None, keepdim=False):
        if input.ndim != 2 or input.dtype != torch.float32 or not input.is_cuda:
            raise ValueError("expected a CUDA float32 matrix")

        n0, n1 = input.shape
        if dim is None:
            flat = input.reshape(-1)
            out = torch.empty((), device=input.device, dtype=torch.int64)
            block = _next_power_of_two(flat.numel())
            _sigmoid_argmax_flat[(1,)](flat, out, flat.numel(), BLOCK=block)
            return out

        dim = dim + input.ndim if dim < 0 else dim
        if dim not in (0, 1):
            raise ValueError("dimension out of range")
        size = n0 if dim == 1 else n1
        out_shape = (n0, 1) if dim == 1 and keepdim else (1, n1) if keepdim else ((n0,) if dim == 1 else (n1,))
        out = torch.empty(out_shape, device=input.device, dtype=torch.int64)
        programs = n0 if dim == 1 else n1
        block = _next_power_of_two(size)
        _sigmoid_argmax_2d[(programs,)](
            input,
            out.reshape(-1),
            n0,
            n1,
            input.stride(0),
            input.stride(1),
            reduce_dim=dim,
            BLOCK=block,
        )
        return out

    return sigmoid_argmax
