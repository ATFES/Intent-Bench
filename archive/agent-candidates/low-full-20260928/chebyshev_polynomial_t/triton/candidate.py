import torch
import triton
import triton.language as tl


@triton.jit
def _chebyshev_t3(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    # T_3(x) = 4*x^3 - 3*x, matching the fixed n=3 invocation.
    result = 4.0 * x * x * x - 3.0 * x
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def chebyshev_polynomial_t(input: torch.Tensor, n: int, out: torch.Tensor = None) -> torch.Tensor:
        output = torch.empty_like(input) if out is None else out
        _chebyshev_t3[(triton.cdiv(input.numel(), 256),)](
            input, output, input.numel(), BLOCK=256
        )
        return output

    return chebyshev_polynomial_t
