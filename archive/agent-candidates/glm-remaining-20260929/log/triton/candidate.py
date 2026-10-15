import torch
import triton
import triton.language as tl


@triton.jit
def _log_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.log(values), mask=mask)


def build(context):
    def log(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        output = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        if n_elements:
            _log_kernel[(triton.cdiv(n_elements, 1024),)](
                input,
                output,
                n_elements,
                BLOCK=1024,
            )
        return output

    return log
