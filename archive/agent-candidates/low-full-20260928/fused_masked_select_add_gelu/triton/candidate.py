import torch
import triton
import triton.language as tl


N = 1048576
BLOCK = 1024
NBLOCKS = N // BLOCK
NSELECTED = 524236


@triton.jit
def _count_mask(mask_ptr, counts_ptr, n_elements, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    values = tl.load(mask_ptr + offsets, mask=valid, other=0)
    count = tl.sum(values.to(tl.int32), axis=0)
    tl.store(counts_ptr + block, count)


@triton.jit
def _block_offsets(counts_ptr, offsets_ptr, NBLOCKS: tl.constexpr):
    ids = tl.arange(0, NBLOCKS)
    counts = tl.load(counts_ptr + ids)
    inclusive = tl.cumsum(counts, axis=0)
    exclusive = inclusive - counts
    tl.store(offsets_ptr + ids, exclusive)


@triton.jit
def _compact_gelu_none(input_ptr, mask_ptr, block_offsets_ptr, output_ptr,
                       alpha, other, n_elements, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    mask = tl.load(mask_ptr + offsets, mask=valid, other=0)
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0)
    local = tl.cumsum(mask.to(tl.int32), axis=0) - 1
    base = tl.load(block_offsets_ptr + block)
    destination = base + local
    selected = values + alpha * other
    result = 0.5 * selected * (1.0 + tl.erf(selected * 0.7071067811865476))
    tl.store(output_ptr + destination, result, mask=mask & valid)


@triton.jit
def _compact_gelu_tanh(input_ptr, mask_ptr, block_offsets_ptr, output_ptr,
                       alpha, other, n_elements, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    mask = tl.load(mask_ptr + offsets, mask=valid, other=0)
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0)
    local = tl.cumsum(mask.to(tl.int32), axis=0) - 1
    base = tl.load(block_offsets_ptr + block)
    destination = base + local
    selected = values + alpha * other
    cubic = selected * selected * selected
    result = 0.5 * selected * (1.0 + tl.tanh(0.7978845608028654 * (selected + 0.044715 * cubic)))
    tl.store(output_ptr + destination, result, mask=mask & valid)


def build(context):
    def wrapper(input, mask, other, *, alpha=1, approximate="none", out=None):
        if out is None:
            output = torch.empty((NSELECTED,), device=input.device, dtype=input.dtype)
        else:
            output = out
        counts = torch.empty((NBLOCKS,), device=input.device, dtype=torch.int32)
        block_offsets = torch.empty_like(counts)
        _count_mask[(NBLOCKS,)](mask, counts, input.numel(), BLOCK=BLOCK)
        _block_offsets[(1,)](counts, block_offsets, NBLOCKS=NBLOCKS)
        if approximate == "tanh":
            _compact_gelu_tanh[(NBLOCKS,)](
                input, mask, block_offsets, output, alpha, other,
                input.numel(), BLOCK=BLOCK)
        else:
            _compact_gelu_none[(NBLOCKS,)](
                input, mask, block_offsets, output, alpha, other,
                input.numel(), BLOCK=BLOCK)
        return output

    return wrapper
