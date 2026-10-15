import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _logit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride,
    output_stride,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    result = tl.log(x / (1.0 - x))
    tl.store(output_ptr + offsets * output_stride, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _logit_eps_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    input_stride,
    output_stride,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    upper = 1.0 - eps
    x = tl.maximum(tl.minimum(x, upper), eps)
    result = tl.log(x / (1.0 - x))
    tl.store(output_ptr + offsets * output_stride, result, mask=mask)


def build(context):
    def wrapper(input, eps=None, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        if n_elements == 0:
            return output

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        if eps is None:
            _logit_kernel[grid](
                input,
                output,
                n_elements,
                input.stride(0),
                output.stride(0),
            )
        else:
            _logit_eps_kernel[grid](
                input,
                output,
                n_elements,
                input.stride(0),
                output.stride(0),
                eps=eps,
            )
        return output

    return wrapper
