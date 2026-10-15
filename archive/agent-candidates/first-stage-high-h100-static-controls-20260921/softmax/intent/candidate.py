import torch
import intent
import intent.language as I


@intent.kernel
def _softmax_rows(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, x.shape[0])
    columns = I.domain(0, x.shape[1])

    for row in I.parallel(rows):
        values = x[row, columns]
        row_max = I.reduce.max(values, axis=0, acc_dtype=I.f32)
        shifted = values - row_max
        weights = I.exp(shifted)
        row_sum = I.reduce.sum(weights, axis=0, acc_dtype=I.f32)
        y[row, columns] = I.fdiv(weights, row_sum)


def build(context):
    softmax_rows = context.compile("softmax_rows", _softmax_rows)

    def softmax(input, dim, dtype=None):
        output_dtype = input.dtype if dtype is None else dtype
        output = torch.empty(input.shape, dtype=output_dtype, device=input.device)
        softmax_rows(input, output)
        return output

    return softmax
