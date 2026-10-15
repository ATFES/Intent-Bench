import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    result = tl.tanh(values)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def tanh(input_tensor, out_tensor=None):
        output = out_tensor if out_tensor is not None else torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        _tanh_kernel[(triton.cdiv(n_elements, 256),)](
            input_tensor, output, n_elements, BLOCK=256
        )
        return output

    return tanh
