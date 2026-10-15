import torch
import intent
import intent.language as I


@intent.kernel
def mean_partial_sums(x: I.In[I.f32, (1048576,)], partials: I.Out[I.f32, (4096,)]):
    # Read the full input, reshape into a grid of 4096 groups of 256 elements each,
    # and reduce each group independently along the inner axis.
    vals = x[:]
    grid = I.reshape(vals, (4096, 256))
    sums = I.reduce.sum(grid, axis=1, acc_dtype=I.f32)
    partials[:] = sums


@intent.kernel
def mean_finalize(partials: I.In[I.f32, (4096,)], out: I.Out[I.f32, ()],
                  count: I.f32):
    vals = partials[:]
    total = I.reduce.sum(vals, axis=0, acc_dtype=I.f32)
    out[()] = total / count


def build(context):
    stage1 = context.compile("mean_partial_sums", mean_partial_sums)
    stage2 = context.compile("mean_finalize", mean_finalize)

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        n = input_tensor.shape[0]
        if dtype is not None:
            input_tensor = input_tensor.to(dtype=dtype)
        partials = torch.empty((4096,), dtype=input_tensor.dtype,
                               device=input_tensor.device)
        if out is None:
            result = torch.empty((), dtype=input_tensor.dtype,
                                 device=input_tensor.device)
        else:
            result = out
        stage1(input_tensor, partials)
        stage2(partials, result, float(n))
        return result

    return mean

