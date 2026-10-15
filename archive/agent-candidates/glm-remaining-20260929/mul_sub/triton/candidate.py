import torch
import triton
import triton.language as tl


@triton.jit
def _mul_sub_kernel(
    input_ptr,
    other_mul_ptr,
    other_sub_ptr,
    output_ptr,
    other_mul_scalar,
    other_sub_scalar,
    alpha,
    n_elements,
    OTHER_MUL_IS_SCALAR: tl.constexpr,
    OTHER_SUB_IS_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    input_value = tl.load(input_ptr + offsets, mask=valid)
    if OTHER_MUL_IS_SCALAR:
        other_mul_value = other_mul_scalar
    else:
        other_mul_value = tl.load(other_mul_ptr + offsets, mask=valid)

    if OTHER_SUB_IS_SCALAR:
        other_sub_value = other_sub_scalar
    else:
        other_sub_value = tl.load(other_sub_ptr + offsets, mask=valid)

    output_value = input_value * other_mul_value - alpha * other_sub_value
    tl.store(output_ptr + offsets, output_value, mask=valid)


def build(context):
    def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
        n_elements = input.numel()
        if out is None:
            output = torch.empty_like(input, memory_format=torch.contiguous_format)
        else:
            output = out

        other_mul_is_scalar = not isinstance(other_mul, torch.Tensor)
        other_sub_is_scalar = not isinstance(other_sub, torch.Tensor)
        grid = (triton.cdiv(n_elements, 1024),)
        _mul_sub_kernel[grid](
            input,
            input if other_mul_is_scalar else other_mul,
            input if other_sub_is_scalar else other_sub,
            output,
            other_mul if other_mul_is_scalar else 0.0,
            other_sub if other_sub_is_scalar else 0.0,
            alpha,
            n_elements,
            OTHER_MUL_IS_SCALAR=other_mul_is_scalar,
            OTHER_SUB_IS_SCALAR=other_sub_is_scalar,
            BLOCK_SIZE=1024,
        )
        return output

    return mul_sub
