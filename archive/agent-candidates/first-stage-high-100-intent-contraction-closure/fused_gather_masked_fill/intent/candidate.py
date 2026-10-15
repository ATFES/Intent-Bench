import intent
import intent.language as I


@intent.kernel
def _fused_gather_masked_fill(
    input: I.In[I.f32, (1024, 1024)],
    index: I.In[I.i64, (1024, 1024)],
    mask: I.In[I.bool, (1024, 1024)],
    output: I.Out[I.f32, (1024, 1024)],
    value: I.f32,
):
    rows = I.domain(0, 1024)
    columns = I.domain(0, 1024)

    gather_rows = index[rows, columns]
    column_indices = I.indices(columns)
    gathered = input[gather_rows, column_indices]
    output[rows, columns] = I.select(mask[rows, columns], value, gathered)


def build(context):
    kernel = context.compile("fused_gather_masked_fill", _fused_gather_masked_fill)

    def fused_gather_masked_fill(
        input,
        dim,
        index,
        mask,
        value,
        *,
        sparse_grad=False,
        out=None,
    ):
        del dim, sparse_grad
        if out is None:
            return kernel.run(input, index, mask, value)
        kernel(input, index, mask, out, value)
        return out

    return fused_gather_masked_fill
