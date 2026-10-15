import torch
import triton
import triton.language as tl


@triton.jit
def _log_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.log(x), mask=mask)


def build(context):
    def log(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        _log_kernel[(triton.cdiv(n_elements, 256),)](
            input, out, n_elements, BLOCK=256
        )
        return out

    return log
