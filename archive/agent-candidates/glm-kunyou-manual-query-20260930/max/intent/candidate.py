import torch
import intent
import intent.language as I

@intent.kernel
def max_argmax_kernel(input: I.In[I.f32, ("N",)],
                       max_val: I.Out[I.f32, ()],
                       max_idx: I.Out[I.i64, ()]):
    values, indices = I.arg_reduce.max(input, axis=0)
    max_val[()] = values
    max_idx[()] = I.cast(indices, I.i64)

def build(context):
    artifact = context.compile("max_argmax", max_argmax_kernel)
    def wrapper(input, dim, keepdim=False):
        max_val = torch.empty([], dtype=torch.float32, device=input.device)
        max_idx = torch.empty([], dtype=torch.int64, device=input.device)
        artifact(input, max_val, max_idx)
        return (max_val, max_idx)
    return wrapper

