import torch
import triton
import triton.language as tl


@triton.jit
def _add_gelu_tensor(x_ptr, y_ptr, out_ptr, n, alpha, approximate: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + alpha * y
    if approximate == "none":
        result = z * (0.5 * (1.0 + tl.erf(z * 0.7071067811865476)))
    else:
        result = 0.5 * z * (1.0 + tl.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)))
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.jit
def _add_gelu_scalar(x_ptr, out_ptr, n, other, alpha, approximate: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + alpha * other
    if approximate == "none":
        result = z * (0.5 * (1.0 + tl.erf(z * 0.7071067811865476)))
    else:
        result = 0.5 * z * (1.0 + tl.tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)))
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def add_gelu(input, other, alpha=1, approximate="none", out=None):
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")
        result = out if out is not None else torch.empty_like(input)
        n = input.numel()
        grid = (triton.cdiv(n, 256),)
        if torch.is_tensor(other):
            _add_gelu_tensor[grid](input, other, result, n, alpha, approximate=approximate, BLOCK=256)
        else:
            _add_gelu_scalar[grid](input, result, n, other, alpha, approximate=approximate, BLOCK=256)
        return result

    return add_gelu
