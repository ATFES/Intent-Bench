import torch
import triton
import triton.language as tl


@triton.jit
def _mul_sub_tensor_kernel(
    input_ptr,
    other_mul_ptr,
    other_sub_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    mul = tl.load(other_mul_ptr + offsets, mask=mask)
    sub = tl.load(other_sub_ptr + offsets, mask=mask)
    result = x * mul - alpha * sub
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_sub_scalar_kernel(
    input_ptr,
    output_ptr,
    other_mul,
    other_sub,
    alpha,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    result = x * other_mul - alpha * other_sub
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_sub_mul_tensor_kernel(
    input_ptr,
    other_mul_ptr,
    other_sub,
    output_ptr,
    alpha,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    mul = tl.load(other_mul_ptr + offsets, mask=mask)
    result = x * mul - alpha * other_sub
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _mul_sub_sub_tensor_kernel(
    input_ptr,
    other_mul,
    other_sub_ptr,
    output_ptr,
    alpha,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    sub = tl.load(other_sub_ptr + offsets, mask=mask)
    result = x * other_mul - alpha * sub
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, other_mul, other_sub, alpha=1, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)

        mul_is_tensor = isinstance(other_mul, torch.Tensor)
        sub_is_tensor = isinstance(other_sub, torch.Tensor)
        if mul_is_tensor and sub_is_tensor:
            _mul_sub_tensor_kernel[grid](
                input,
                other_mul,
                other_sub,
                output,
                alpha,
                n_elements,
                BLOCK=1024,
            )
        elif not mul_is_tensor and not sub_is_tensor:
            _mul_sub_scalar_kernel[grid](
                input,
                output,
                other_mul,
                other_sub,
                alpha,
                n_elements,
                BLOCK=1024,
            )
        elif mul_is_tensor:
            _mul_sub_mul_tensor_kernel[grid](
                input,
                other_mul,
                other_sub,
                output,
                alpha,
                n_elements,
                BLOCK=1024,
            )
        else:
            _mul_sub_sub_tensor_kernel[grid](
                input,
                other_mul,
                other_sub,
                output,
                alpha,
                n_elements,
                BLOCK=1024,
            )
        return output

    return wrapper
