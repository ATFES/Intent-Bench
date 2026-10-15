import torch
import triton
import triton.language as tl


@triton.jit
def _floor_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.floor(values), mask=mask)


def build(context):
    def floor(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _floor_kernel[(triton.cdiv(n_elements, 1024),)](
            input, result, n_elements, BLOCK=1024
        )
        return result

    return floor
