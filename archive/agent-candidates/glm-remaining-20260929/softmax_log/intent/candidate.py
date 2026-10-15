from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def softmax_log_rows(
    values: I.In[I.f32, (1024, 1024)],
    result: I.Out[I.f32, (1024, 1024)],
) -> None:
    input_values = values[:, :]
    log_values = I.log(input_values)

    row_maximum = I.reduce.max(log_values, axis=1)
    maximum = I.reshape(row_maximum, (log_values.shape[0], 1))
    shifted = log_values - maximum
    scores = I.exp2(shifted * I.cast(1.4426950408889634, I.f32), approximate=True)

    row_denominator = I.reduce.sum(scores, axis=1, acc_dtype=I.f32)
    denominator = I.reshape(row_denominator, (scores.shape[0], 1))
    result[:, :] = I.fdiv(scores, denominator, approximate=True)


def build(context):
    artifact = context.compile(
        "softmax_log_rows_f32_1024x1024",
        softmax_log_rows,
    )

    def softmax_log(input, dim=-1, dtype=None):
        normalized_dim = dim if dim >= 0 else dim + input.dim()
        if normalized_dim != input.dim() - 1:
            raise ValueError("this profile specializes softmax to the final dimension")
        if dtype is not None:
            raise ValueError("this profile specializes to the input dtype")
        return artifact.run(input)

    return softmax_log
