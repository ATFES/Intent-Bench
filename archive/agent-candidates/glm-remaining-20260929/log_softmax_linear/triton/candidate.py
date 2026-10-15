from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    rows,
    out_features,
    in_features,
    input_row_stride,
    weight_row_stride,
    weight_col_stride,
    bias_stride,
    output_row_stride,
    HAS_BIAS: tl.constexpr,
    INPUT_CAST: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    row_offsets = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row_offsets < rows
    col_mask = col_offsets < out_features

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, in_features, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < in_features
        input_offsets = (
            row_offsets[:, None] * input_row_stride
            + k_offsets[None, :]
        )
        if EVEN_M and EVEN_K:
            input_tile = tl.load(input_ptr + input_offsets)
        else:
            input_tile = tl.load(
                input_ptr + input_offsets,
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
        if INPUT_CAST:
            input_tile = input_tile.to(INPUT_DTYPE)
        input_tile = input_tile.to(tl.float32)

        weight_offsets = (
            k_offsets[:, None] * weight_row_stride
            + col_offsets[None, :] * weight_col_stride
        )
        if EVEN_N and EVEN_K:
            weight_tile = tl.load(weight_ptr + weight_offsets)
        else:
            weight_tile = tl.load(
                weight_ptr + weight_offsets,
                mask=k_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
        weight_tile = weight_tile.to(tl.float32)

        accumulator = tl.dot(
            input_tile,
            weight_tile,
            accumulator,
            input_precision="tf32x3",
            out_dtype=tl.float32,
        )

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + col_offsets * bias_stride,
            mask=col_mask,
            other=0.0,
        )
        accumulator += bias.to(tl.float32)

    output_offsets = (
        row_offsets[:, None] * output_row_stride + col_offsets[None, :]
    )
    if EVEN_M and EVEN_N:
        tl.store(output_ptr + output_offsets, accumulator.to(output_ptr.dtype.element_ty))
    else:
        tl.store(
            output_ptr + output_offsets,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )


@triton.jit
def _log_softmax_kernel(
    values_ptr,
    row_width,
    input_row_stride,
    output_row_stride,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK_N)
    mask = col < row_width
    values = tl.load(
        values_ptr + row * input_row_stride + col,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    maximum = tl.max(values, axis=0)
    probabilities = tl.exp(values - maximum)
    denominator = tl.sum(probabilities, axis=0)
    result = values - maximum - tl.log(denominator)
    tl.store(
        values_ptr + row * output_row_stride + col,
        result.to(values_ptr.dtype.element_ty),
        mask=mask,
    )


_TRITON_FLOAT_TYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def build(context):
    def log_softmax_linear(
        input,
        weight,
        bias=None,
        dim=-1,
        dtype=None,
    ):
        if input.device.type != "cuda" or weight.device.type != "cuda":
            raise ValueError("input and weight must be CUDA tensors")
        if dim != -1 and dim != input.ndim - 1:
            raise ValueError("only the final dimension is supported")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")
        if dtype is not None and dtype not in _TRITON_FLOAT_TYPES:
            raise ValueError("unsupported dtype")

        in_features = input.shape[-1]
        out_features = weight.shape[0]
        if weight.shape[1] != in_features:
            raise ValueError("input and weight have incompatible shapes")
        if bias is not None and bias.shape != (out_features,):
            raise ValueError("bias has an incompatible shape")

        rows = input.numel() // in_features
        matrix_input = input.reshape((rows, in_features))
        result_dtype = dtype if dtype is not None else input.dtype
        result_dtype = torch.promote_types(result_dtype, weight.dtype)
        if bias is not None:
            result_dtype = torch.promote_types(result_dtype, bias.dtype)
        output = torch.empty(
            (rows, out_features),
            dtype=result_dtype,
            device=input.device,
        )

        block_m = 16
        block_n = min(64, triton.next_power_of_2(out_features))
        block_k = min(64, triton.next_power_of_2(in_features))
        if block_n < 16 or block_k < 16:
            raise ValueError("matrix dimensions too small for the selected kernel")

        bias_kernel_arg = bias if bias is not None else weight
        input_cast_dtype = dtype if dtype is not None else input.dtype
        _linear_kernel[(triton.cdiv(rows, block_m), triton.cdiv(out_features, block_n))](
            matrix_input,
            weight,
            bias_kernel_arg,
            output,
            rows,
            out_features,
            in_features,
            matrix_input.stride(0),
            weight.stride(0),
            weight.stride(1),
            bias.stride(0) if bias is not None else 0,
            output.stride(0),
            HAS_BIAS=bias is not None,
            INPUT_CAST=dtype is not None and dtype != input.dtype,
            INPUT_DTYPE=_TRITON_FLOAT_TYPES[input_cast_dtype],
            EVEN_M=rows % block_m == 0,
            EVEN_N=out_features % block_n == 0,
            EVEN_K=in_features % block_k == 0,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )

        softmax_block = triton.next_power_of_2(out_features)
        _log_softmax_kernel[(rows,)](
            output,
            out_features,
            output.stride(0),
            output.stride(0),
            BLOCK_N=softmax_block,
            num_warps=4,
        )
        return output.reshape(input.shape[:-1] + (out_features,))

    return log_softmax_linear
