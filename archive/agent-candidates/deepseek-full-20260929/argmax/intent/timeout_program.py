import torch
import intent
import intent.language as I


G = 1024


@intent.kernel
def argmax_partial_kernel(
    input_tensor: I.In[I.f16, (1048576,)],
    partial_vals: I.Out[I.f16, (1024,)],
    partial_inds: I.Out[I.i64, (1024,)],
):
    rows = I.domain(0, 1048576)
    values = input_tensor[rows]
    blocks = I.reshape(values, (1024, 1024))

    block_vals, block_inds = I.arg_reduce.max(blocks, axis=1)

    groups = I.domain(0, 1024)
    partial_vals[groups] = I.cast(block_vals, I.f16)
    partial_inds[groups] = I.cast(block_inds, I.i64)


@intent.kernel
def argmax_final_kernel(
    partial_vals: I.In[I.f16, (1024,)],
    partial_inds: I.In[I.i64, (1024,)],
    out: I.Out[I.i64, ()],
):
    groups = I.domain(0, 1024)
    vals = partial_vals[groups]

    _, best_pos = I.arg_reduce.max(vals, axis=0)
    local_pos = partial_inds[best_pos]
    global_pos = I.cast(best_pos, I.i64) * I.cast(1024, I.i64) + local_pos
    out[()] = global_pos


def build(context):
    partial_kernel = context.compile("argmax_partial_1048576", argmax_partial_kernel)
    final_kernel = context.compile("argmax_final_1024", argmax_final_kernel)

    def argmax(input_tensor, dim, keepdim=False):
        device = input_tensor.device
        partial_vals = torch.empty((G,), dtype=torch.float16, device=device)
        partial_inds = torch.empty((G,), dtype=torch.int64, device=device)
        out = torch.empty((), dtype=torch.int64, device=device)

        partial_kernel(input_tensor, partial_vals, partial_inds)
        final_kernel(partial_vals, partial_inds, out)

        if keepdim and dim is not None:
            return out.reshape((1,))
        return out

    return argmax
