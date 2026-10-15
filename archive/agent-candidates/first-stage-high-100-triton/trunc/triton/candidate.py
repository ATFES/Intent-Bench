import torch
import triton
import triton.language as tl


@triton.jit
def _trunc_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(x >= 0.0, tl.floor(x), tl.ceil(x))
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def trunc(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _trunc_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            output,
            n_elements,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return trunc
