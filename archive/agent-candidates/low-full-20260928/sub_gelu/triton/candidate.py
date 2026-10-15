import torch
import triton
import triton.language as tl


@triton.jit
def _sub_gelu_kernel(inp, other, out, alpha, n_elements, TANH_APPROX: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(inp + offsets, mask=mask)
    y = tl.load(other + offsets, mask=mask)
    z = x - alpha * y
    if TANH_APPROX:
        z3 = z * z * z
        value = 0.5 * z * (1.0 + tl.tanh(0.7978845608028654 * (z + 0.044715 * z3)))
    else:
        value = 0.5 * z * tl.erf(z * 0.7071067811865475)
    tl.store(out + offsets, value, mask=mask)


def build(context):
    exact_kernel = context.compile("sub_gelu_exact", _sub_gelu_kernel,
                                   constexprs={"TANH_APPROX": False, "BLOCK": 1024})
    tanh_kernel = context.compile("sub_gelu_tanh", _sub_gelu_kernel,
                                  constexprs={"TANH_APPROX": True, "BLOCK": 1024})

    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        kernel = tanh_kernel if approximate == "tanh" else exact_kernel
        kernel(input, other, out, alpha, n_elements, grid=grid)
        return out

    return sub_gelu
