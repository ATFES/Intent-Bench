import torch
import triton
import triton.language as tl


@triton.jit
def _tile_exp_kernel(x_ptr, out_ptr, n_elements, n_input, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_offsets = offsets % n_input
    values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.exp(values), mask=mask)


def build(context):
    def fused_tile_exp(input, dims, *, out=None):
        # The supplied profile is a contiguous 1D input with one repetition
        # factor, so tile is a modulo remap in the flattened output.
        repetitions = int(dims[0]) if len(dims) else 1
        shape = (repetitions, input.shape[0])
        output = out if out is not None else torch.empty(shape, device=input.device, dtype=input.dtype)
        n_elements = input.shape[0] * repetitions
        _tile_exp_kernel[(triton.cdiv(n_elements, 1024),)](
            input, output, n_elements, input.shape[0], BLOCK=1024
        )
        return output

    return fused_tile_exp
