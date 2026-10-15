import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def _bitwise_and_contiguous(
    input_ptr,
    other_ptr,
    output_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lhs = tl.load(input_ptr + offsets)
    rhs = tl.load(other_ptr + offsets)
    tl.store(output_ptr + offsets, lhs & rhs)


@triton.jit
def _bitwise_and_masked(
    input_ptr,
    other_ptr,
    output_ptr,
    element_count,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < element_count
    lhs = tl.load(input_ptr + offsets, mask=mask)
    rhs = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, lhs & rhs, mask=mask)


def build(context):
    def bitwise_and(input, other, out=None):
        result = out if out is not None else torch.empty_like(
            input, memory_format=torch.contiguous_format
        )
        element_count = input.numel()

        if element_count % BLOCK_SIZE == 0:
            grid = (element_count // BLOCK_SIZE,)
            _bitwise_and_contiguous[grid](
                input,
                other,
                result,
                BLOCK=BLOCK_SIZE,
            )
        else:
            grid = (triton.cdiv(element_count, BLOCK_SIZE),)
            _bitwise_and_masked[grid](
                input,
                other,
                result,
                element_count,
                BLOCK=BLOCK_SIZE,
            )

        return result

    return bitwise_and
