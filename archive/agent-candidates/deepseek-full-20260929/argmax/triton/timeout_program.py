from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_combine(v1, i1, v2, i2):
    nan1 = v1 != v1
    nan2 = v2 != v2

    # For two ordinary numbers, choose the larger value, or the smaller
    # (left-most) index if the values compare equal.  NaNs are handled
    # separately so that the first NaN wins exactly like torch.argmax.
    numeric = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    choose1 = tl.where(
        nan1,
        tl.where(nan2, i1 < i2, i1 == i1),
        tl.where(nan2, i1 < i1, numeric),
    )
    return tl.where(choose1, v1, v2), tl.where(choose1, i1, i2)


@triton.jit
def _argmax_partial(x_ptr, vals_ptr, idxs_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offs = start + tl.arange(0, BLOCK)
    mask = offs < n_elements

    vals = tl.load(x_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
    idxs = tl.where(mask, offs, n_elements)

    val, idx = tl.reduce((vals, idxs), 0, _argmax_combine)
    tl.store(vals_ptr + pid, val)
    tl.store(idxs_ptr + pid, idx)


@triton.jit
def _argmax_final(vals_ptr, idxs_ptr, out_ptr, n_partials, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_partials

    vals = tl.load(vals_ptr + offs, mask=mask, other=float("-inf")).to(tl.float32)
    idxs = tl.load(idxs_ptr + offs, mask=mask, other=n_partials)

    _, idx = tl.reduce((vals, idxs), 0, _argmax_combine)
    tl.store(out_ptr, idx.to(tl.int64))


def build(context):
    PARTIAL_BLOCK = 2048
    PARTIAL_WARPS = 16
    FINAL_WARPS = 8

    def argmax(input_tensor, dim, keepdim=False):
        if dim is None:
            x = input_tensor.reshape(-1)
        else:
            if dim < 0:
                dim += input_tensor.ndim
            if input_tensor.ndim != 1 or dim != 0:
                raise NotImplementedError(
                    "this candidate targets the fixed 1-D, dim=0 profile"
                )
            x = input_tensor

        n_elements = x.numel()
        out = torch.empty((1,), device=x.device, dtype=torch.int64)

        if n_elements == 0:
            return out if keepdim else out.reshape(())

        n_partials = triton.cdiv(n_elements, PARTIAL_BLOCK)
        partial_vals = torch.empty((n_partials,), device=x.device, dtype=torch.float32)
        partial_idxs = torch.empty((n_partials,), device=x.device, dtype=torch.int32)

        _argmax_partial[(n_partials,)](
            x,
            partial_vals,
            partial_idxs,
            n_elements,
            BLOCK=PARTIAL_BLOCK,
            num_warps=PARTIAL_WARPS,
        )

        final_block = 1
        while final_block < max(n_partials, 32):
            final_block <<= 1
        final_warps = min(FINAL_WARPS, max(1, final_block // 32))

        _argmax_final[(1,)](
            partial_vals,
            partial_idxs,
            out,
            n_partials,
            BLOCK=final_block,
            num_warps=final_warps,
        )
        return out if keepdim else out.reshape(())

    return argmax
