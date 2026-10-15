from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch
import triton
import triton.language as tl


_BLOCK = 1024


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@triton.jit
def _argmax_contig_one_shot(
    in_ptr,
    out_ptr,
    inner_size,
    RED_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per output element.  The input is row-major contiguous,
    # so an output's base address is obtained by decomposing its flat output
    # index into (outer, inner) around the reduced dimension.
    pid = tl.program_id(0)
    outer = pid // inner_size
    inner = pid - outer * inner_size
    base = outer * (RED_SIZE * inner_size) + inner

    offs = tl.arange(0, BLOCK)
    mask = offs < RED_SIZE
    x = tl.load(in_ptr + base + offs * inner_size, mask=mask, other=0.0)
    s = tl.sigmoid(x)
    s = tl.where(mask, s, float("-inf"))

    _max_val, idx = tl.max(
        s,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    tl.store(out_ptr + pid, idx.to(tl.int64))


@triton.jit
def _argmax_contig_loop(
    in_ptr,
    out_ptr,
    red_size,
    inner_size,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    outer = pid // inner_size
    inner = pid - outer * inner_size
    base = outer * (red_size * inner_size) + inner

    best_val = float("-inf")
    best_idx = 0
    for start in tl.range(0, red_size, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < red_size
        x = tl.load(in_ptr + base + offs * inner_size, mask=mask, other=0.0)
        s = tl.sigmoid(x)
        s = tl.where(mask, s, float("-inf"))

        chunk_val, chunk_idx = tl.max(
            s,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_idx = start + chunk_idx
        take = (chunk_val > best_val) or (
            (chunk_val == best_val) and (chunk_idx < best_idx)
        )
        best_val = tl.where(take, chunk_val, best_val)
        best_idx = tl.where(take, chunk_idx, best_idx)

    tl.store(out_ptr + pid, best_idx.to(tl.int64))


@triton.jit
def _argmax_strided_2d(
    in_ptr,
    out_ptr,
    m,
    n,
    stride0,
    stride1,
    AXIS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Fallback for 2D tensors that are not row-major contiguous.  `AXIS` is
    # 0, 1, or 2 (2 means reduce over all elements).
    pid = tl.program_id(0)

    if AXIS == 0:
        base = pid * stride1
        red_size = m
        stride = stride0
    elif AXIS == 1:
        base = pid * stride0
        red_size = n
        stride = stride1
    else:
        base = 0
        red_size = m * n
        stride = 1

    best_val = float("-inf")
    best_idx = 0
    for start in tl.range(0, red_size, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < red_size

        if AXIS == 2:
            row = offs // n
            col = offs - row * n
            ptr = in_ptr + row * stride0 + col * stride1
        else:
            ptr = in_ptr + base + offs * stride

        x = tl.load(ptr, mask=mask, other=0.0)
        s = tl.sigmoid(x)
        s = tl.where(mask, s, float("-inf"))

        chunk_val, chunk_idx = tl.max(
            s,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        chunk_idx = start + chunk_idx
        take = (chunk_val > best_val) or (
            (chunk_val == best_val) and (chunk_idx < best_idx)
        )
        best_val = tl.where(take, chunk_val, best_val)
        best_idx = tl.where(take, chunk_idx, best_idx)

    tl.store(out_ptr + pid, best_idx.to(tl.int64))


def _output_shape(shape: Tuple[int, ...], dim: Optional[int], keepdim: bool) -> Tuple[int, ...]:
    if dim is None:
        return (1,) * len(shape) if keepdim else ()
    if keepdim:
        return shape[:dim] + (1,) + shape[dim + 1 :]
    return shape[:dim] + shape[dim + 1 :]


def _reduce_params(
    shape: Tuple[int, ...], dim: Optional[int]
) -> Tuple[int, int, int]:
    """Return (red_size, inner_size, outer_size) for a contiguous tensor."""
    if dim is None:
        red_size = 1
        for s in shape:
            red_size *= s
        return red_size, 1, 1

    red_size = shape[dim]
    inner_size = 1
    for s in shape[dim + 1 :]:
        inner_size *= s
    outer_size = 1
    for s in shape[:dim]:
        outer_size *= s
    return red_size, inner_size, outer_size


def _run_contiguous(
    x: torch.Tensor,
    dim: Optional[int],
    keepdim: bool,
) -> torch.Tensor:
    shape = tuple(x.shape)
    red_size, inner_size, outer_size = _reduce_params(shape, dim)
    out_shape = _output_shape(shape, dim, keepdim)
    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)

    total = outer_size * inner_size
    grid = (total,)

    if red_size <= _BLOCK:
        block = _next_pow2(red_size)
        _argmax_contig_one_shot[grid](
            x,
            out,
            inner_size,
            RED_SIZE=red_size,
            BLOCK=block,
        )
    else:
        _argmax_contig_loop[grid](
            x,
            out,
            red_size,
            inner_size,
            BLOCK=_BLOCK,
        )
    return out


def _run_strided_2d(
    x: torch.Tensor,
    dim: Optional[int],
    keepdim: bool,
) -> torch.Tensor:
    m, n = x.shape
    stride0, stride1 = x.stride()

    if dim is None:
        axis = 2
        out_shape = (1, 1) if keepdim else ()
        total = 1
    else:
        axis = dim
        if keepdim:
            out_shape = (1, n) if dim == 0 else (m, 1)
        else:
            out_shape = (n,) if dim == 0 else (m,)
        total = n if dim == 0 else m

    out = torch.empty(out_shape, dtype=torch.int64, device=x.device)
    _argmax_strided_2d[(total,)](
        x,
        out,
        m,
        n,
        stride0,
        stride1,
        AXIS=axis,
        BLOCK=_BLOCK,
    )
    return out


def build(context: Any) -> Callable[..., torch.Tensor]:
    def sigmoid_argmax(
        input: torch.Tensor,
        dim: Optional[int] = None,
        keepdim: bool = False,
    ) -> torch.Tensor:
        if input.numel() == 0:
            shape = tuple(input.shape)
            out_shape = _output_shape(shape, dim, keepdim)
            return torch.empty(out_shape, dtype=torch.int64, device=input.device)

        if dim is not None:
            dim = int(dim)
            if dim < 0:
                dim += input.dim()
            if dim < 0 or dim >= input.dim():
                raise IndexError(
                    f"Dimension out of range (expected to be in range of "
                    f"[-{input.dim()}, {input.dim() - 1}], but got {dim})"
                )

        if input.is_contiguous():
            return _run_contiguous(input, dim, keepdim)

        if input.dim() == 2:
            return _run_strided_2d(input, dim, keepdim)

        raise NotImplementedError(
            "sigmoid_argmax currently requires a contiguous tensor or a 2D tensor"
        )

    return sigmoid_argmax
