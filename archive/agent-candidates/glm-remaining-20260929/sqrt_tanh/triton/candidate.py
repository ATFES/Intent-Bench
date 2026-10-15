import torch
import triton
import triton.language as tl


@triton.jit
def _sqrt_tanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0)
    rooted = tl.sqrt_rn(values)
    activated = tl.extra.cuda.libdevice.tanh(rooted)
    tl.store(output_ptr + offsets, activated, mask=valid)


def build(context):
    def sqrt_tanh(input, out=None):
        output = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _sqrt_tanh_kernel[grid](
            input,
            output,
            n_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output

    return sqrt_tanh
