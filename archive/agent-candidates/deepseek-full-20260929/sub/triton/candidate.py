import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4


@triton.jit
def _sub_alpha1_nomask(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + offs, a - b)


@triton.jit
def _sub_alpha_nomask(a_ptr, b_ptr, out_ptr, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + offs, a - alpha * b)


@triton.jit
def _sub_alpha1_mask(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a - b, mask=mask)


@triton.jit
def _sub_alpha_mask(a_ptr, b_ptr, out_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a - alpha * b, mask=mask)


@triton.jit
def _sub_scalar_nomask(a_ptr, out_ptr, other, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    a = tl.load(a_ptr + offs)
    tl.store(out_ptr + offs, a - alpha * other)


@triton.jit
def _sub_scalar_mask(a_ptr, out_ptr, n_elements, other, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a - alpha * other, mask=mask)


def _launch_tensor(a, b, out, alpha):
    n = a.numel()
    block = _BLOCK
    use_mask = (n % block) != 0
    grid = ((n + block - 1) // block,)

    if alpha == 1:
        if use_mask:
            _sub_alpha1_mask[grid](
                a, b, out, n,
                BLOCK=block,
                num_warps=_NUM_WARPS,
            )
        else:
            _sub_alpha1_nomask[grid](
                a, b, out,
                BLOCK=block,
                num_warps=_NUM_WARPS,
            )
    else:
        if use_mask:
            _sub_alpha_mask[grid](
                a, b, out, n, alpha,
                BLOCK=block,
                num_warps=_NUM_WARPS,
            )
        else:
            _sub_alpha_nomask[grid](
                a, b, out, alpha,
                BLOCK=block,
                num_warps=_NUM_WARPS,
            )


def _launch_scalar(a, out, other, alpha):
    n = a.numel()
    block = _BLOCK
    use_mask = (n % block) != 0
    grid = ((n + block - 1) // block,)

    if use_mask:
        _sub_scalar_mask[grid](
            a, out, n, other, alpha,
            BLOCK=block,
            num_warps=_NUM_WARPS,
        )
    else:
        _sub_scalar_nomask[grid](
            a, out, other, alpha,
            BLOCK=block,
            num_warps=_NUM_WARPS,
        )


def build(context):
    def sub(input: 'torch.Tensor', other: 'torch.Tensor', *, alpha: 'float' = 1, out: 'torch.Tensor' = None) -> 'torch.Tensor':
        if not isinstance(other, torch.Tensor):
            result_dtype = torch.result_type(input, other)
            if out is None:
                out = torch.empty(input.shape, dtype=result_dtype, device=input.device)
            if not input.is_contiguous():
                raise NotImplementedError("non-contiguous scalar path is not supported")
            _launch_scalar(input, out, other, alpha)
            return out

        if input.shape != other.shape:
            raise NotImplementedError("broadcast path is not supported")

        result_dtype = torch.result_type(input, other)
        if out is None:
            out = torch.empty(input.shape, dtype=result_dtype, device=input.device)

        if not (input.is_contiguous() and other.is_contiguous()):
            raise NotImplementedError("non-contiguous tensor path is not supported")

        _launch_tensor(input, other, out, alpha)
        return out

    return sub
