import intent
import intent.language as I


@intent.kernel
def _softmax_log_last(
    x: I.In[I.f32, (1024, 1024)],
    y: I.Out[I.f32, (1024, 1024)],
):
    rows = I.domain(0, 1024)
    columns = I.domain(0, 1024)

    log_x = I.log(x[rows, columns])
    row_max = I.reduce.max(log_x, axis=1, acc_dtype=I.f32)
    row_max = I.reshape(row_max, (1024, 1))
    exp_x = I.exp(log_x - row_max)
    row_sum = I.reduce.sum(exp_x, axis=1, acc_dtype=I.f32)
    row_sum = I.reshape(row_sum, (1024, 1))
    y[rows, columns] = exp_x / row_sum


def build(context):
    kernel = context.compile("softmax_log_last_1024", _softmax_log_last)

    def softmax_log(input, dim=-1, dtype=None):
        if dim not in (-1, 1):
            raise ValueError("the fixed profile supports dim=-1")
        if dtype is not None:
            raise ValueError("the fixed profile supports dtype=None")
        return kernel.run(input)

    return softmax_log
