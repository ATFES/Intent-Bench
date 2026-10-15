import torch
import triton
import triton.language as tl


@triton.jit
def _rsqrt_kernel(inp, out, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(inp + offsets, mask=mask)
    tl.store(out + offsets, tl.rsqrt(x), mask=mask)


def build(context):
    def rsqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _rsqrt_kernel[(triton.cdiv(n_elements, 256),)](
            input, result, n_elements, BLOCK=256
        )
        return result

    return rsqrt
