import torch
import intent
import intent.language as I


@intent.kernel
def fused_tile_exp_kernel(
    input: I.In[I.f32, (1048576,)],
    output: I.Out[I.f32, (2, 1048576)],
):
    source = I.domain(0, 1048576)
    values = I.exp(input[source])
    repetitions = I.domain(0, 2)
    output[repetitions, source] = values


def build(context):
    artifact = context.compile("fused_tile_exp", fused_tile_exp_kernel)

    def fused_tile_exp(input, dims, *, out=None):
        if out is None:
            out = torch.empty(
                (dims[0], input.shape[0]),
                device=input.device,
                dtype=input.dtype,
            )
            artifact(input, out)
            return out
        artifact(input, out)
        return out

    return fused_tile_exp
