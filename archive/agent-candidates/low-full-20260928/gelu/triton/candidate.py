import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, approximate: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    if approximate == "tanh":
        c = 0.044715
        k = 0.7978845608028654
        inner = k * (x + c * x * x * x)
        y = 0.5 * x * (1.0 + tl.tanh(inner))
    else:
        inv_sqrt2 = 0.7071067811865475
        y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def wrapper(input, approximate="none"):
        output = torch.empty_like(input)
        n_elements = input.numel()
        _gelu_kernel[(triton.cdiv(n_elements, 256),)](
            input, output, n_elements, approximate=approximate, BLOCK=256
        )
        return output

    return wrapper
