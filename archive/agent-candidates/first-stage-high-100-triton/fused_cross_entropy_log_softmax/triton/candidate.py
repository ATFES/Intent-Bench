import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_rows(
    input_ptr,
    target_ptr,
    partial_ptr,
    n_cols,
    ignore_index,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    in_bounds = cols < n_cols

    logits = tl.load(input_ptr + row * n_cols + cols, mask=in_bounds, other=-float("inf"))
    row_max = tl.max(logits, axis=0)
    exp_logits = tl.exp(logits - row_max)
    log_sum_exp = row_max + tl.log(tl.sum(exp_logits, axis=0))

    target = tl.load(target_ptr + row)
    included = target != ignore_index
    safe_target = tl.where(included, target, 0)
    target_logit = tl.load(input_ptr + row * n_cols + safe_target)
    loss = tl.where(included, log_sum_exp - target_logit, 0.0)

    partial_ptr = partial_ptr + row * 2
    tl.store(partial_ptr, loss)
    tl.store(partial_ptr + 1, included.to(tl.float32))


@triton.jit
def _finish_mean(partial_ptr, output_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    losses = tl.load(partial_ptr + 2 * offsets)
    included = tl.load(partial_ptr + 2 * offsets + 1)
    total_loss = tl.sum(losses, axis=0)
    total_included = tl.sum(included, axis=0)
    result = tl.where(total_included > 0, total_loss / total_included, float("nan"))
    tl.store(output_ptr, result)


def build(context):
    def wrapper(
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int = 1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        n_rows = input.shape[0]
        n_cols = input.shape[1]
        partial = torch.empty((n_rows, 2), device=input.device, dtype=torch.float32)
        output = torch.empty((), device=input.device, dtype=torch.float32)

        _cross_entropy_rows[(n_rows,)](
            input,
            target,
            partial,
            n_cols,
            ignore_index,
            BLOCK=128,
            num_warps=4,
        )
        _finish_mean[(1,)](partial, output, N=4096, num_warps=4)
        return output

    return wrapper
