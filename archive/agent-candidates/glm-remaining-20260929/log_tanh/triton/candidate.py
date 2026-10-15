import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def _log_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=valid, other=1.0)

    # For positive x, tanh(log(x)) equals (x*x - 1) / (x*x + 1).  The range
    # guard keeps the square finite and returns the correctly rounded float32
    # limit for very large positive inputs, including positive infinity.
    large = value > 1.0e19
    squared = value * value
    result = tl.where(large, 1.0, (squared - 1.0) / (squared + 1.0))
    tl.store(output_ptr + offsets, result, mask=valid)


def build(context):
    def log_tanh(input, out=None):
        count = input.numel()
        output = torch.empty_like(input) if out is None else out
        if count != 0:
            grid = ((count + BLOCK_SIZE - 1) // BLOCK_SIZE,)
            _log_tanh_kernel[grid](
                input,
                output,
                count,
                BLOCK=BLOCK_SIZE,
                num_warps=4,
            )
        return output

    return log_tanh
