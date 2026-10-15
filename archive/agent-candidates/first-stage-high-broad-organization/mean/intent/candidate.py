import torch
import intent
import intent.language as I


@intent.kernel
def mean_partial(
    input_tensor: I.In[I.f32, (1048576,)],
    partials: I.Out[I.f32, (1024,)],
):
    source = I.domain(0, 1048576)
    for group in I.parallel(I.domain(0, 1024)):
        begin = group * 1024
        window = source[begin : begin + 1024]
        partials[group] = I.reduce.sum(
            input_tensor[window], axis=0, acc_dtype=I.f32
        )


@intent.kernel
def mean_finalize(
    partials: I.In[I.f32, (1024,)],
    output: I.Out[I.f32, ()],
):
    total = I.reduce.sum(
        partials[I.domain(0, 1024)], axis=0, acc_dtype=I.f32
    )
    output[()] = total / I.cast(1048576, I.f32)


def build(context):
    partial_artifact = context.compile("mean_partial_1024", mean_partial)
    finalize_artifact = context.compile("mean_finalize_1024", mean_finalize)

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        partials = torch.empty(
            (1024,), device=input_tensor.device, dtype=torch.float32
        )
        result = out
        if result is None:
            result = torch.empty(
                (), device=input_tensor.device, dtype=torch.float32
            )
        partial_artifact(input_tensor, partials)
        finalize_artifact(partials, result)
        return result

    return mean
