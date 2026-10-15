import torch
import triton
import triton.language as tl


_MAX_PROGRAMS = 1024
_BLOCK_SIZE = 1024
_NEG_INF = float("-inf")


@triton.jit
def _partial_logsumexp_kernel(
    input_ptr,
    max_ptr,
    sum_ptr,
    num_elements,
    chunk_size: tl.constexpr,
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    program_start = program_id * chunk_size

    tile_max = _NEG_INF
    tile_sum = 0.0
    for tile_offset in range(0, chunk_size, block_size):
        offsets = program_start + tile_offset + tl.arange(0, block_size)
        mask = offsets < num_elements
        values = tl.load(input_ptr + offsets, mask=mask, other=_NEG_INF)

        block_max = tl.max(values, axis=0)
        safe_values = tl.where((values == _NEG_INF) & (block_max == _NEG_INF), 0.0, values)
        block_sum = tl.sum(tl.exp(safe_values - block_max), axis=0)
        tile_max = tl.maximum(tile_max, block_max)
        tile_sum += block_sum

    tl.store(max_ptr + program_id, tile_max)
    tl.store(sum_ptr + program_id, tile_sum)


@triton.jit
def _combine_logsumexp_kernel(
    max_ptr,
    sum_ptr,
    output_ptr,
    num_programs,
    max_programs: tl.constexpr,
):
    offsets = tl.arange(0, max_programs)
    mask = offsets < num_programs
    partial_max = tl.load(max_ptr + offsets, mask=mask, other=_NEG_INF)
    partial_sum = tl.load(sum_ptr + offsets, mask=mask, other=0.0)

    global_max = tl.max(partial_max, axis=0)
    safe_partial_max = tl.where(partial_max == _NEG_INF, 0.0, partial_max)
    scales = tl.exp(safe_partial_max - global_max)
    total_sum = tl.sum(scales * partial_sum, axis=0)
    tl.store(output_ptr, tl.log(total_sum))


def build(context):
    del context

    def logsumexp(input, dim, keepdim=False, *, out=None):
        if (
            input.ndim != 1
            or input.dtype != torch.float32
            or input.device.type != "cuda"
            or input.stride(0) != 1
            or dim not in (0, -1)
            or (out is not None and (out.dtype != torch.float32 or out.shape != ()))
        ):
            raise ValueError("unsupported logsumexp invocation profile")

        num_elements = input.numel()
        num_programs = max(1, min(_MAX_PROGRAMS, num_elements))
        chunk_size = triton.cdiv(num_elements, num_programs)
        partials = torch.empty(
            2 * num_programs + 1, dtype=torch.float32, device=input.device
        )
        partial_max = partials[:num_programs]
        partial_sum = partials[num_programs:-1]
        output = partials[-1].view(()) if out is None else out

        _partial_logsumexp_kernel[(num_programs,)](
            input,
            partial_max,
            partial_sum,
            num_elements,
            chunk_size=chunk_size,
            block_size=_BLOCK_SIZE,
            num_warps=4,
        )
        _combine_logsumexp_kernel[(1,)](
            partial_max,
            partial_sum,
            output,
            num_programs,
            max_programs=_MAX_PROGRAMS,
            num_warps=4,
        )
        return output

    return logsumexp
