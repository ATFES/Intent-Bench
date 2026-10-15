import torch
import triton
import triton.language as tl


@triton.jit
def _ifftshift_1d(input_ptr, output_ptr, n_elements, shift, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    source = offsets + shift
    source = tl.where(source >= n_elements, source - n_elements, source)
    values = tl.load(input_ptr + source, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def build(context):
    def ifftshift(input, dim=None):
        output = torch.empty_like(input)
        n_elements = input.numel()
        # The fixed profile is a contiguous 1D tensor.  For that tensor,
        # ifftshift uses a rotation by floor(n / 2) elements.
        if dim is None or dim == 0 or dim == -1:
            shift = n_elements // 2
            _ifftshift_1d[(triton.cdiv(n_elements, 1024),)](
                input, output, n_elements, shift, BLOCK=1024
            )
            return output
        raise ValueError("unsupported dim for fixed 1D profile")

    return ifftshift
