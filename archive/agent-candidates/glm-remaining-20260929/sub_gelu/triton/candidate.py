import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _sub_gelu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    alpha,
    n_elements,
    APPROXIMATE: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        input_value = tl.load(input_ptr + offsets)
        other_value = tl.load(other_ptr + offsets)
    else:
        valid = offsets < n_elements
        input_value = tl.load(input_ptr + offsets, mask=valid)
        other_value = tl.load(other_ptr + offsets, mask=valid)

    difference = input_value - alpha * other_value
    if APPROXIMATE:
        inner = difference + 0.044715 * difference * difference * difference
        activation = libdevice.tanh(0.7978845608028654 * inner)
        result = 0.5 * difference * (1.0 + activation)
    else:
        result = 0.5 * difference * tl.erf(difference * 0.7071067811865476)

    if EVEN:
        tl.store(output_ptr + offsets, result)
    else:
        tl.store(output_ptr + offsets, result, mask=valid)


def build(context):
    def sub_gelu(input, other, alpha=1, approximate="none", out=None):
        result = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        even = n_elements % 1024 == 0
        _sub_gelu_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            other,
            result,
            float(alpha),
            n_elements,
            APPROXIMATE=approximate == "tanh",
            EVEN=even,
            BLOCK=1024,
            num_warps=4,
        )
        return result

    return sub_gelu
