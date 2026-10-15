import torch
import triton
import triton.language as tl


@triton.jit
def _chebyshev_t3(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # T_3(x) = 2*x*T_2(x) - T_1(x), with T_2(x) = 2*x*x - 1.
    t2 = 2.0 * x * x - 1.0
    result = 2.0 * x * t2 - x
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, n, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        if n_elements:
            _chebyshev_t3[(triton.cdiv(n_elements, 1024),)](
                input,
                output,
                n_elements,
                BLOCK=1024,
                num_warps=4,
            )
        return output

    return wrapper
