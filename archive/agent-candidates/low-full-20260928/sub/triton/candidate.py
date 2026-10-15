import torch
import triton
import triton.language as tl


@triton.jit
def _sub_kernel(x_ptr, y_ptr, out_ptr, alpha, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x - alpha * y
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def sub(input: torch.Tensor, other: torch.Tensor, *, alpha: float = 1,
            out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _sub_kernel[grid](input, other, out, alpha, n_elements,
                          BLOCK=1024, num_warps=4)
        return out

    return sub
