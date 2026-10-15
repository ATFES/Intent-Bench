import torch
import triton
import triton.language as tl


@triton.jit
def _softplus_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    K,
    input_stride_m,
    input_stride_k,
    weight_stride_n,
    weight_stride_k,
    output_stride_m,
    output_stride_n,
    beta,
    threshold,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    output_offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_valid = (output_offsets_m[:, None] < M) & (output_offsets_n[None, :] < N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        input_offsets_k = k_start + tl.arange(0, BLOCK_K)
        input_valid_k = input_offsets_k < K
        input_offsets = (
            output_offsets_m[:, None] * input_stride_m
            + input_offsets_k[None, :] * input_stride_k
        )
        input_tile = tl.load(
            input_ptr + input_offsets,
            mask=(output_offsets_m[:, None] < M) & input_valid_k[None, :],
            other=0.0,
        )

        weight_offsets = (
            output_offsets_n[:, None] * weight_stride_n
            + input_offsets_k[None, :] * weight_stride_k
        )
        weight_tile = tl.load(
            weight_ptr + weight_offsets,
            mask=(output_offsets_n[:, None] < N) & input_valid_k[None, :],
            other=0.0,
        )

        accumulator = tl.dot(
            input_tile,
            weight_tile,
            acc=accumulator,
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    if HAS_BIAS:
        bias = tl.load(bias_ptr + output_offsets_n, mask=output_offsets_n < N, other=0.0)
        accumulator += bias[None, :]

    scaled = beta * accumulator
    softplus = tl.log(1.0 + tl.exp(scaled)) / beta
    result = tl.where(accumulator > threshold, accumulator, softplus)
    output_offsets = (
        output_offsets_m[:, None] * output_stride_m
        + output_offsets_n[None, :] * output_stride_n
    )
    tl.store(output_ptr + output_offsets, result, mask=output_valid)


def build(context):
    def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
        M, K = input.shape
        N, weight_k = weight.shape
        output = torch.empty((M, N), device=input.device, dtype=input.dtype)
        _softplus_linear_kernel[(triton.cdiv(M, 16), triton.cdiv(N, 16))](
            input,
            weight,
            output if bias is None else bias,
            output,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            beta,
            threshold,
            HAS_BIAS=bias is not None,
            BLOCK_M=16,
            BLOCK_N=16,
            BLOCK_K=64,
        )
        return output

    return softplus_linear
