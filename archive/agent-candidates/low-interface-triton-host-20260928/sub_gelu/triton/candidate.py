import torch
import triton
import triton.language as tl


@triton.jit
def _sub_gelu_kernel(input_ptr, other_ptr, out_ptr, n_elements, alpha, USE_TANH: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    z = x - alpha * y
    if USE_TANH:
        c = 0.7978845608028654
        inner = c * (z + 0.044715 * z * z * z)
        result = 0.5 * z * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    else:
        result = 0.5 * z * tl.erf(z * 0.7071067811865475)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        use_tanh = approximate == "tanh"
        grid = (triton.cdiv(n_elements, 1024),)
        _sub_gelu_kernel[grid](
            input,
            other,
            out,
            n_elements,
            float(alpha),
            USE_TANH=use_tanh,
            BLOCK=1024,
        )
        return out

    return sub_gelu
