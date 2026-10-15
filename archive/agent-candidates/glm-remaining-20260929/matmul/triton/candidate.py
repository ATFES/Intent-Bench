import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a_tile = tl.load(a_ptrs)
        b_tile = tl.load(b_ptrs)
        accumulator = tl.dot(a_tile, b_tile, accumulator)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator.to(tl.float16))


def build(context):
    del context

    def matmul(tensor1, tensor2):
        if tensor1.shape != (1024, 1024) or tensor2.shape != (1024, 1024):
            raise ValueError("this specialization requires two 1024x1024 tensors")
        if tensor1.dtype is not torch.float16 or tensor2.dtype is not torch.float16:
            raise TypeError("this specialization requires float16 inputs")
        if tensor1.device != tensor2.device or tensor1.device.type != "cuda":
            raise ValueError("both inputs must be on the same CUDA device")
        if tensor1.stride() != (1024, 1) or tensor2.stride() != (1024, 1):
            raise ValueError("both inputs must be contiguous")

        output = torch.empty(
            (1024, 1024),
            dtype=torch.float16,
            device=tensor1.device,
        )
        _matmul_kernel[(16, 8)](
            tensor1,
            tensor2,
            output,
            1024,
            1024,
            1024,
            tensor1.stride(0),
            tensor1.stride(1),
            tensor2.stride(0),
            tensor2.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_M=64,
            BLOCK_N=128,
            BLOCK_K=64,
            num_warps=8,
            num_stages=4,
        )
        return output

    return matmul
