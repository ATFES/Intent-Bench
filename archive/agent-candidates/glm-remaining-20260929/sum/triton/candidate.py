import torch
import triton
import triton.language as tl


PARTIAL_BLOCK = 8192
FINAL_BLOCK = 128


@triton.jit
def _partial_sum(
    input_ptr,
    partial_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    chunk_offset = tl.program_id(0) * BLOCK
    offsets = chunk_offset + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    tl.store(partial_ptr + tl.program_id(0), tl.sum(values, axis=0))


@triton.jit
def _final_sum(
    partial_ptr,
    output_ptr,
    n_partials,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(partial_ptr + offsets, mask=offsets < n_partials, other=0.0)
    tl.store(output_ptr, tl.sum(values, axis=0))


def build(context):
    def sum(input, dim, keepdim=False, dtype=None):
        if input.ndim != 1 or input.stride(0) != 1:
            raise ValueError("this kernel requires a one-dimensional contiguous input")
        if dim not in (0, None):
            raise ValueError("dim must be 0 or None for a one-dimensional input")

        output_shape = [] if not keepdim else [1]
        output_dtype = input.dtype if dtype is None else dtype
        n_elements = input.numel()
        partials = (n_elements + PARTIAL_BLOCK - 1) // PARTIAL_BLOCK

        workspace = torch.empty(partials + 1, dtype=output_dtype, device=input.device)
        output = workspace[:1].view(output_shape)

        _partial_sum[(partials,)](
            input,
            workspace,
            n_elements,
            BLOCK=PARTIAL_BLOCK,
            num_warps=8,
        )
        _final_sum[(1,)](
            workspace[1:],
            output,
            partials,
            BLOCK=FINAL_BLOCK,
        )
        return output

    return sum
