import torch
import triton
import triton.language as tl


@triton.jit
def _log1p_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)

    shifted = x + 1.0
    rounded_delta = x - (shifted - 1.0)
    correction = rounded_delta / shifted
    is_finite = (shifted < float("inf")) & (shifted > float("-inf"))
    log_value = tl.log(shifted) + tl.where(is_finite, correction, 0.0)

    invalid = tl.where(x == -1.0, float("-inf"), float("nan"))
    result = tl.where(x > -1.0, log_value, invalid)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def log1p(input, out=None):
        output = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _log1p_kernel[grid](input, output, n_elements, BLOCK_SIZE=block_size)
        return output

    return log1p
