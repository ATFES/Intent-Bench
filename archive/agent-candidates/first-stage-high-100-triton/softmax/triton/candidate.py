import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_rows(
    input_ptr,
    output_ptr,
    n_cols,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row = input_ptr + row * input_stride
    output_row = output_ptr + row * output_stride

    values = tl.load(input_row + offsets, mask=mask, other=-float("inf"))
    values = values - tl.max(values, axis=0)
    numerators = tl.exp(values)
    denominator = tl.sum(numerators, axis=0)
    tl.store(output_row + offsets, numerators / denominator, mask=mask)


def build(context):
    def wrapper(input, dim, dtype=None):
        output = torch.empty_like(input)
        n_rows = input.shape[0]
        n_cols = input.shape[1]
        _softmax_rows[(n_rows,)](
            input,
            output,
            n_cols,
            input.stride(0),
            output.stride(0),
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output

    return wrapper
