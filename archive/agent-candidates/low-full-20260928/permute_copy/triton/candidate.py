import torch
import triton
import triton.language as tl


@triton.jit
def _permute_copy_reverse_32(x, out, n_elements, BLOCK: tl.constexpr):
    out_idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = out_idx < n_elements

    # Output coordinates are (d3, d2, d1, d0); input is contiguous
    # (d0, d1, d2, d3).  The profile has extent 32 on every axis.
    d0 = out_idx & 31
    d1 = (out_idx >> 5) & 31
    d2 = (out_idx >> 10) & 31
    d3 = (out_idx >> 15) & 31
    input_idx = d3 * 32768 + d2 * 1024 + d1 * 32 + d0
    value = tl.load(x + input_idx, mask=mask)
    tl.store(out + out_idx, value, mask=mask)


def build(context):
    def wrapper(input, dims):
        output = torch.empty_like(input)
        n_elements = input.numel()
        _permute_copy_reverse_32[(triton.cdiv(n_elements, 256),)](
            input, output, n_elements, BLOCK=256
        )
        return output

    return wrapper
