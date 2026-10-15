import torch
import triton
import triton.language as tl


@triton.jit
def _nan_max(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _lse_block_kernel(x_ptr, pm_ptr, ps_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=float("-inf"))
    m = tl.reduce(x, 0, _nan_max)
    s = tl.sum(tl.exp(x - m), axis=0)
    tl.store(pm_ptr + pid, m)
    tl.store(ps_ptr + pid, s)


@triton.jit
def _lse_final_kernel(pm_ptr, ps_ptr, out_ptr, n_blocks, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_blocks
    pm = tl.load(pm_ptr + offsets, mask=mask, other=float("-inf"))
    ps = tl.load(ps_ptr + offsets, mask=mask, other=0.0)
    m = tl.reduce(pm, 0, _nan_max)
    total = tl.sum(ps * tl.exp(pm - m), axis=0)
    result = m + tl.log(total)
    result = tl.where(m == float("inf"), float("inf"), result)
    result = tl.where(m == float("-inf"), float("-inf"), result)
    tl.store(out_ptr, result)


_BLOCK = 1024
_NUM_WARPS = 4


def build(context):
    def logsumexp(input, dim, keepdim=False, *, out=None):
        if input.dim() != 1:
            raise NotImplementedError("logsumexp: only 1-D inputs are supported")
        dim = dim + input.dim() if dim < 0 else dim
        if dim != 0:
            raise NotImplementedError("logsumexp: only reduction over dim 0 is supported")

        n = input.numel()
        if n == 0:
            sizes = list(input.shape)
            if keepdim:
                sizes[0] = 1
            else:
                sizes = []
            if out is None:
                out = torch.empty(sizes, device=input.device, dtype=input.dtype)
            out.fill_(float("-inf"))
            return out

        n_blocks = triton.cdiv(n, _BLOCK)
        pm = torch.empty((n_blocks,), device=input.device, dtype=torch.float32)
        ps = torch.empty((n_blocks,), device=input.device, dtype=torch.float32)

        sizes = list(input.shape)
        if keepdim:
            sizes[0] = 1
        else:
            sizes = []
        if out is None:
            out = torch.empty(sizes, device=input.device, dtype=input.dtype)

        _lse_block_kernel[(n_blocks,)](input, pm, ps, n, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
        final_block = triton.next_power_of_2(n_blocks)
        _lse_final_kernel[(1,)](pm, ps, out, n_blocks, BLOCK=final_block, num_warps=_NUM_WARPS)
        return out

    return logsumexp
