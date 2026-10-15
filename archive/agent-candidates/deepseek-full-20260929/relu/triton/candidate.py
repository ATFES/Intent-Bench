import torch
import triton
import triton.language as tl


_BLOCK = 1024


@triton.jit
def _relu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def _relu_kernel_aligned(x_ptr, y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.maximum(x, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(y_ptr + offsets, y)


def build(context):
    def relu(input: 'torch.Tensor', inplace: 'bool' = False) -> 'torch.Tensor':
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        if n_elements == 0:
            return output

        if input.is_contiguous() and n_elements % _BLOCK == 0:
            _relu_kernel_aligned[(n_elements // _BLOCK,)](input, output, BLOCK=_BLOCK)
        else:
            grid = (triton.cdiv(n_elements, _BLOCK),)
            _relu_kernel[grid](input, output, n_elements, BLOCK=_BLOCK)
        return output

    return relu
