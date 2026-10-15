import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_log_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    x = tl.cast(x, tl.float32)
    z = tl.log(x)
    row_max = tl.max(z, axis=0)
    num = tl.exp(z - row_max)
    den = tl.sum(num, axis=0)
    out = num / den
    tl.store(y_ptr + row * n_cols + cols, out, mask=mask)


def build(context):
    def softmax_log(input, dim=-1, dtype=None):
        # The supplied profile is contiguous 2-D row-wise softmax.
        if dim != -1 and dim != input.ndim - 1:
            raise NotImplementedError("only the profiled last dimension is supported")
        out_dtype = input.dtype if dtype is None else dtype
        output = torch.empty_like(input, dtype=out_dtype)
        rows = input.numel() // input.shape[-1]
        block = triton.next_power_of_2(input.shape[-1])
        _softmax_log_kernel[(rows,)](
            input, output, input.shape[-1], BLOCK=block
        )
        return output

    return softmax_log
