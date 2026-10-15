import torch
import intent
import intent.language as I


@intent.kernel
def fused_embedding_add_tanh_kernel(
    input_indices: I.In[I.i64, ("M", "N")],
    weight: I.In[I.f32, ("V", "D")],
    other: I.In[I.f32, ("M", "N", "D")],
    output: I.Out[I.f32, ("M", "N", "D")],
):
    M, N, D = other.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    features = I.domain(0, D)

    lookup = input_indices[rows, columns]
    embeddings = weight[lookup, features]
    summed = embeddings + other[rows, columns, features]
    output[rows, columns, features] = I.tanh(summed, approximate=False)


def build(context):
    fused = context.compile(
        "fused_embedding_add_tanh",
        fused_embedding_add_tanh_kernel,
        constexprs={},
    )

    def fused_embedding_add_tanh(
        input_indices,
        weight,
        other,
        *,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        out=None,
    ):
        result = out if out is not None else torch.empty_like(other)
        fused(input_indices, weight, other, result)
        return result

    return fused_embedding_add_tanh
