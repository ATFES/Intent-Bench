import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_partials(x, partial_values, partial_indices, n_elements,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=-float("inf"))
    local_index = tl.argmax(values, axis=0)
    value = tl.load(x + pid * BLOCK + local_index)
    tl.store(partial_values + pid, value)
    tl.store(partial_indices + pid, (pid * BLOCK + local_index).to(tl.int64))


@triton.jit
def _argmax_final(partial_values, partial_indices, output, n_partials,
                  BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partial_values + offsets, mask=mask, other=-float("inf"))
    local_index = tl.argmax(values, axis=0)
    index = tl.load(partial_indices + local_index)
    tl.store(output, index)


def build(context):
    partial_block = 1024

    def wrapper(input_tensor, dim, keepdim=False):
        n_elements = input_tensor.numel()
        n_partials = triton.cdiv(n_elements, partial_block)
        partial_values = torch.empty((n_partials,), device=input_tensor.device,
                                     dtype=input_tensor.dtype)
        partial_indices = torch.empty((n_partials,), device=input_tensor.device,
                                      dtype=torch.int64)
        output = torch.empty((), device=input_tensor.device, dtype=torch.int64)

        _argmax_partials[(n_partials,)](
            input_tensor, partial_values, partial_indices, n_elements,
            BLOCK=partial_block)
        _argmax_final[(1,)](
            partial_values, partial_indices, output, n_partials,
            BLOCK=partial_block)
        return output

    return wrapper
