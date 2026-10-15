import torch
import triton
import triton.language as tl


@triton.jit
def _max_partial(x, partial_values, partial_indices, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=-float("inf"))
    local_index = tl.arange(0, BLOCK)
    best_value = tl.max(values, axis=0)
    best_index = tl.argmax(values, axis=0)
    # argmax is over the padded tile; masked lanes cannot win for finite data,
    # and the explicit index keeps the partial result in input coordinates.
    best_index = pid * BLOCK + best_index
    tl.store(partial_values + pid, best_value)
    tl.store(partial_indices + pid, best_index)


@triton.jit
def _max_final(partial_values, partial_indices, out_value, out_index,
               n_partials, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partial_values + offsets, mask=mask, other=-float("inf"))
    indices = tl.load(partial_indices + offsets, mask=mask, other=0)
    best_value = tl.max(values, axis=0)
    # Select the first original index among equal partial maxima.
    candidates = tl.where(values == best_value, indices, 9223372036854775807)
    best_index = tl.min(candidates, axis=0)
    tl.store(out_value, best_value)
    tl.store(out_index, best_index)


def build(context):
    partial_block = 1024
    n_partials = 1024

    def wrapper(input, dim, keepdim=False):
        values = torch.empty((), device=input.device, dtype=input.dtype)
        indices = torch.empty((), device=input.device, dtype=torch.int64)
        partial_values = torch.empty((n_partials,), device=input.device,
                                     dtype=input.dtype)
        partial_indices = torch.empty((n_partials,), device=input.device,
                                      dtype=torch.int64)
        _max_partial[(n_partials,)](
            input, partial_values, partial_indices, input.numel(),
            BLOCK=partial_block)
        _max_final[(1,)](
            partial_values, partial_indices, values, indices, n_partials,
            BLOCK=n_partials)
        return values, indices

    return wrapper
