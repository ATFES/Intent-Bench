import torch
import triton
import triton.language as tl


@triton.jit
def _mul_sub_kernel(
    input_ptr,
    other_mul_ptr,
    other_sub_ptr,
    output_ptr,
    n_elements,
    alpha,
    HAS_MUL_TENSOR: tl.constexpr,
    HAS_SUB_TENSOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    if HAS_MUL_TENSOR:
        mul_value = tl.load(other_mul_ptr + offsets, mask=mask)
    else:
        mul_value = other_mul_ptr
    if HAS_SUB_TENSOR:
        sub_value = tl.load(other_sub_ptr + offsets, mask=mask)
    else:
        sub_value = other_sub_ptr

    result = x * mul_value - alpha * sub_value
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)

        has_mul_tensor = isinstance(other_mul, torch.Tensor)
        has_sub_tensor = isinstance(other_sub, torch.Tensor)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 256),)

        _mul_sub_kernel[grid](
            input,
            other_mul if has_mul_tensor else other_mul,
            other_sub if has_sub_tensor else other_sub,
            out,
            n_elements,
            alpha,
            HAS_MUL_TENSOR=has_mul_tensor,
            HAS_SUB_TENSOR=has_sub_tensor,
            BLOCK=256,
        )
        return out

    return mul_sub
