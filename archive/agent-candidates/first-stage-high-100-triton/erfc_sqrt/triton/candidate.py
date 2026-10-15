import torch
import triton
import triton.language as tl


@triton.jit
def erfc_sqrt_kernel(
    input_ptr,
    erfc_ptr,
    sqrt_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # erfc(x) = 1 - erf(x).  Triton's erf maps to the CUDA libdevice
    # implementation for float32 and has the same useful endpoint behavior
    # as erfc for this tolerance (including infinities and NaNs).
    erfc_value = 1.0 - tl.math.erf(x)
    sqrt_value = tl.sqrt(x)

    tl.store(erfc_ptr + offsets, erfc_value, mask=mask)
    tl.store(sqrt_ptr + offsets, sqrt_value, mask=mask)


def build(context):
    def wrapper(input: torch.Tensor):
        erfc_output = torch.empty_like(input)
        sqrt_output = torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        erfc_sqrt_kernel[grid](
            input,
            erfc_output,
            sqrt_output,
            n_elements,
            BLOCK=1024,
            num_warps=8,
        )
        return erfc_output, sqrt_output

    return wrapper
