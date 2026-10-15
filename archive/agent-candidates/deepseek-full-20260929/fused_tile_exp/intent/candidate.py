import torch
import intent
import intent.language as I


@intent.kernel
def _fused_tile_exp_kernel(
    x: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("R", "N")],
):
    R = y.shape[0]
    N = x.shape[0]
    rows = I.domain(0, R)
    cols = I.domain(0, N)
    y[rows, cols] = I.exp(x[cols])


def build(context):
    artifact = context.compile(
        "fused_tile_exp", _fused_tile_exp_kernel, constexprs={}
    )

    def fused_tile_exp(input, dims, *, out=None):
        dims = tuple(dims)
        if input.dim() != 1 or len(dims) != 1:
            raise ValueError(
                "fused_tile_exp requires a 1-D input and exactly one repetition count"
            )
        reps = int(dims[0])
        n = input.shape[0]
        output_shape = (reps, n)

        if out is None:
            out = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        else:
            if (
                tuple(out.shape) != output_shape
                or out.dtype != input.dtype
                or out.device != input.device
            ):
                raise ValueError("out has incompatible shape, dtype, or device")

        artifact(input, out)
        return out

    return fused_tile_exp
