import torch
import triton
import triton.language as tl


@triton.jit
def _log_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=1.0)
    result = tl.tanh(tl.log(values))
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def log_tanh(input, out=None):
        output = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        _log_tanh_kernel[(triton.cdiv(n_elements, 256),)](
            input, output, n_elements, BLOCK=256
        )
        return output

    return log_tanh
