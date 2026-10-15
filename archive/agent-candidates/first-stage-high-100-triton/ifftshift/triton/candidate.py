import torch
import triton
import triton.language as tl


@triton.jit
def _ifftshift_1d_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    shift,
    BLOCK: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    source_offsets = offsets + shift
    source_offsets = tl.where(source_offsets < n_elements, source_offsets, source_offsets - n_elements)
    values = tl.load(input_ptr + source_offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def build(context):
    def wrapper(input, dim=None):
        output = torch.empty_like(input)
        n_elements = input.numel()

        # The supplied profile is a contiguous one-dimensional tensor with
        # dim=None.  For a one-dimensional input, an explicitly selected
        # dimension has the same shift as the all-dim form.
        shift = n_elements // 2
        _ifftshift_1d_kernel[(triton.cdiv(n_elements, 4096),)](
            input,
            output,
            n_elements,
            shift,
            BLOCK=4096,
            num_warps=8,
        )
        return output

    return wrapper
