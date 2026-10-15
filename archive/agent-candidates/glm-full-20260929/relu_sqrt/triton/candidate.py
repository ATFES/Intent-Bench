import torch
import triton
import triton.language as tl


@triton.jit
def _relu_sqrt_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    value = tl.load(input_ptr + offsets, mask=mask)
    rectified = tl.maximum(value, 0.0, propagate_nan=tl.PropagateNan.ALL)
    result = tl.sqrt_rn(rectified)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def relu_sqrt(input, inplace=False, out=None):
        if inplace and out is not None and out is not input:
            destination = torch.empty_like(input)
            _relu_sqrt_kernel[(triton.cdiv(input.numel(), 1024),)](
                input,
                destination,
                input.numel(),
                BLOCK_SIZE=1024,
                num_warps=4,
            )
            _relu_sqrt_kernel[(triton.cdiv(input.numel(), 1024),)](
                input,
                out,
                input.numel(),
                BLOCK_SIZE=1024,
                num_warps=4,
            )
            return out

        if inplace:
            destination = input
        elif out is not None:
            destination = out
        else:
            destination = torch.empty_like(input)

        _relu_sqrt_kernel[(triton.cdiv(input.numel(), 1024),)](
            input,
            destination,
            input.numel(),
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return destination

    return relu_sqrt
