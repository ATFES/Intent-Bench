import torch
import intent
import intent.language as I


@intent.kernel
def fused_kernel(
    input_indices: I.In[I.i64, (16, 16)],
    weight: I.In[I.f32, (1024, 16)],
    other: I.In[I.f32, (16, 16, 16)],
    out: I.Out[I.f32, (16, 16, 16)],
):
    # The indexed read is the embedding stage.  Its free trailing axis is
    # the embedding dimension, so the following pointwise stages retain the
    # complete [16, 16, 16] independent work domain.
    embedding = I.gather(weight, input_indices)
    summed = embedding + other
    out[:, :, :] = I.tanh(summed)


def build(context):
    kernel = context.compile("fused_embedding_add_tanh", fused_kernel)

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
        if out is None:
            out = torch.empty((16, 16, 16), device=weight.device, dtype=weight.dtype)
        kernel(input_indices, weight, other, out)
        return out

    return fused_embedding_add_tanh
