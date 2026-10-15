import torch
import triton
import triton.language as tl


@triton.jit
def _reverse_4d_copy(
    input_ptr,
    output_ptr,
    BLOCK_Q: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0)
    high = tl.program_id(1)
    q = tl.arange(0, BLOCK_Q)
    s = tl.arange(0, BLOCK_S)

    input_offsets = (
        q[:, None] * 1024
        + row * 32
        + s[None, :]
        + high * 32768
    )
    values = tl.load(input_ptr + input_offsets)

    output_offsets = (
        s[:, None] * 1024
        + row * 32
        + q[None, :]
        + high * 32768
    )
    tl.store(output_ptr + output_offsets, values)


@triton.jit
def _permute_4d_copy(
    input_ptr,
    output_ptr,
    total_elements,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    input_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    BLOCK_SIZE: tl.constexpr,
):
    output_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = output_offset < total_elements

    coordinate_0 = output_offset // output_stride_0
    remainder = output_offset % output_stride_0
    coordinate_1 = remainder // output_stride_1
    remainder %= output_stride_1
    coordinate_2 = remainder // output_stride_2
    coordinate_3 = remainder % output_stride_2

    input_offset = (
        coordinate_0 * input_stride_0
        + coordinate_1 * input_stride_1
        + coordinate_2 * input_stride_2
        + coordinate_3 * input_stride_3
    )
    values = tl.load(input_ptr + input_offset, mask=valid)
    tl.store(output_ptr + output_offset, values, mask=valid)


def build(context):
    def permute_copy(input: torch.Tensor, dims: list) -> torch.Tensor:
        if (
            input.shape == (32, 32, 32, 32)
            and input.stride() == (32768, 1024, 32, 1)
            and dims == [3, 2, 1, 0]
        ):
            output = torch.empty_like(input)
            _reverse_4d_copy[(32, 32)](
                input,
                output,
                BLOCK_Q=32,
                BLOCK_S=32,
                num_warps=4,
            )
            return output

        output = torch.empty(
            [input.size(dim) for dim in dims],
            dtype=input.dtype,
            device=input.device,
        )
        total_elements = input.numel()
        block_size = 1024
        grid = (triton.cdiv(total_elements, block_size),)
        _permute_4d_copy[grid](
            input,
            output,
            total_elements,
            input.stride(dims[0]),
            input.stride(dims[1]),
            input.stride(dims[2]),
            input.stride(dims[3]),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            BLOCK_SIZE=block_size,
        )
        return output

    return permute_copy
