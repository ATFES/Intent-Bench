import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_argmax_rows(
    input_ptr,
    output_ptr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    offsets = row * N_COLS + cols

    if N_COLS == BLOCK_N:
        values = tl.load(input_ptr + offsets)
    else:
        values = tl.load(input_ptr + offsets, mask=cols < N_COLS, other=-float("inf"))
    probabilities = 1.0 / (1.0 + tl.exp(-values))
    indices = tl.argmax(probabilities, axis=0, tie_break_left=True)
    tl.store(output_ptr + row, indices.to(tl.int64))


def build(context):
    def sigmoid_argmax(input, dim=None, keepdim=False):
        if input.ndim == 0:
            raise ValueError("dim=-1 reduction requires at least one dimension")
        if dim is not None and dim != -1 and dim != input.ndim - 1:
            raise ValueError("this implementation supports only the fixed dim=-1 reduction")
        if not input.is_contiguous():
            raise ValueError("input must be contiguous")

        n_rows = input.numel() // input.shape[-1]
        n_cols = input.shape[-1]
        output = torch.empty(
            input.shape[:-1], dtype=torch.int64, device=input.device
        )

        if n_rows != 0:
            block_n = triton.next_power_of_2(n_cols)
            _sigmoid_argmax_rows[(n_rows,)](
                input,
                output,
                N_COLS=n_cols,
                BLOCK_N=block_n,
                num_warps=4,
            )

        if keepdim:
            return output.unsqueeze(-1)
        return output

    return sigmoid_argmax
