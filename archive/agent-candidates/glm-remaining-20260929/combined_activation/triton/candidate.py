import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _combined_activation_kernel(
    input_ptr,
    weight1_ptr,
    weight2_ptr,
    bias_ptr,
    output_ptr,
    input_stride,
    output_stride,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_offset = tl.program_id(0) * BLOCK_M
    column_offset = tl.program_id(1) * BLOCK_N

    rows = row_offset + tl.arange(0, BLOCK_M)
    columns = column_offset + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for depth_offset in range(0, K, BLOCK_K):
        depth = depth_offset + tl.arange(0, BLOCK_K)
        input_tile = tl.load(
            input_ptr + rows[:, None] * input_stride + depth[None, :]
        )
        weight_tile = tl.load(
            weight1_ptr + depth[:, None] + columns[None, :]
        )
        accumulator = tl.dot(
            input_tile,
            weight_tile,
            accumulator,
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    activation = tl.sigmoid(accumulator)
    result = libdevice.tanh(activation)
    result = result * tl.load(weight2_ptr + columns)[None, :]
    result = result + tl.load(bias_ptr + columns)[None, :]

    output_offsets = rows[:, None] * output_stride + columns[None, :]
    tl.store(output_ptr + output_offsets, result)


def build(context):
    def combined_activation(input, weight1, weight2, bias, *, out=None):
        if input.ndim != 2 or weight1.ndim != 2:
            raise ValueError("input and weight1 must be two-dimensional")
        if input.shape[1] != weight1.shape[0]:
            raise ValueError("input and weight1 have incompatible matrix dimensions")

        rows, depth = input.shape
        columns = weight1.shape[1]
        if out is None:
            output = torch.empty(
                (rows, columns), dtype=input.dtype, device=input.device
            )
        else:
            if out.shape != (rows, columns) or out.dtype != input.dtype:
                raise ValueError("out must have the GEMM result shape and dtype")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            output = out

        _combined_activation_kernel[(rows // 64, columns // 32)](
            input,
            weight1,
            weight2,
            bias,
            output,
            input.stride(0),
            output.stride(0),
            K=depth,
            BLOCK_M=64,
            BLOCK_N=32,
            BLOCK_K=64,
            num_warps=4,
            num_stages=4,
        )
        return output

    return combined_activation
