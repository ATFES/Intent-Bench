import torch
import intent
import intent.language as I

@intent.kernel
def fused_cross_entropy_kernel(
    input: I.In[I.f32, (4096, 128)],
    target: I.In[I.i64, (4096,)],
    output: I.Out[I.f32, ()],
    dim: I.Constexpr[int],
    ignore_index: I.Constexpr[int],
    label_smoothing: I.Constexpr[float]
):
    N = 4096
    C = 128

    losses = I.buffer((N,), I.f32)

    rows = I.domain(0, N)
    for row in I.parallel(rows):
        row_input = input[row, :]
        max_val = I.reduce.max(row_input, axis=0)
        shifted = row_input - max_val
        log_sum_exp = I.log(I.reduce.sum(I.exp(shifted), axis=0))
        log_softmax_values = shifted - log_sum_exp

        t = target[row]
        class_idx = I.cast(t, I.index)
        target_log_prob = I.gather(log_softmax_values, class_idx)

        loss = -target_log_prob

        if label_smoothing != 0.0:
            smooth_target = I.f32(1.0 - label_smoothing) + label_smoothing / I.f32(C)
            smooth_loss = -I.log(smooth_target)
            loss = loss * I.f32(1.0 - label_smoothing) + smooth_loss * I.f32(label_smoothing)

        valid = (t != ignore_index)
        losses[row] = I.select(valid, loss, I.full((), 0.0, I.f32))

    valid_mask = target != ignore_index
    valid_count = I.reduce.sum(I.cast(valid_mask, I.f32), axis=0)
    loss_sum = I.reduce.sum(losses[:], axis=0)

    output[()] = loss_sum / valid_count


def build(context):
    artifact = context.compile(
        "fused_cross_entropy_kernel",
        fused_cross_entropy_kernel,
        constexprs={
            "dim": 1,
            "ignore_index": -100,
            "label_smoothing": 0.0
        }
    )

    def wrapper(
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int = 1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        output = torch.empty((), dtype=torch.float32, device=input.device)
        artifact(input, target, output)
        return output

    return wrapper

