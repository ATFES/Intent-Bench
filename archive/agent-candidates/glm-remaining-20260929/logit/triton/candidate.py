import torch
import triton
import triton.language as tl


@triton.jit
def _logit_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    eps,
    upper,
    HAS_EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    if HAS_EPS:
        values = tl.where(values < eps, eps, values)
        values = tl.where(values > upper, upper, values)

    logits = tl.log(values / (1.0 - values))
    tl.store(output_ptr + offsets, logits, mask=mask)


def build(context):
    def logit(input: torch.Tensor, eps: float = None, out: torch.Tensor = None):
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out

        n_elements = input.numel()
        if eps is None:
            _logit_kernel[(triton.cdiv(n_elements, 1024),)](
                input,
                output,
                n_elements,
                0.0,
                0.0,
                HAS_EPS=False,
                BLOCK_SIZE=1024,
            )
        else:
            _logit_kernel[(triton.cdiv(n_elements, 1024),)](
                input,
                output,
                n_elements,
                eps,
                1.0 - eps,
                HAS_EPS=True,
                BLOCK_SIZE=1024,
            )

        return output

    return logit
