import torch
import triton
import triton.language as tl


@triton.jit
def _mul_relu_tensor(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    other_value = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    value = input_value * other_value
    value = tl.where(value > 0, value, 0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


@triton.jit
def _mul_relu_scalar(
    input_ptr,
    other,
    output_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    input_value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    value = input_value * other
    value = tl.where(value > 0, value, 0.0)
    tl.store(output_ptr + offsets, value, mask=mask)


def build(context):
    block = 1024

    def wrapper(input, other, inplace=False, out=None):
        if inplace:
            output = input
        elif out is not None:
            output = out
        else:
            output = torch.empty_like(input)

        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, block),)
        if isinstance(other, torch.Tensor):
            _mul_relu_tensor[grid](
                input,
                other,
                output,
                n_elements,
                BLOCK=block,
                num_warps=4,
            )
        else:
            _mul_relu_scalar[grid](
                input,
                other,
                output,
                n_elements,
                BLOCK=block,
                num_warps=4,
            )
        return output

    return wrapper
