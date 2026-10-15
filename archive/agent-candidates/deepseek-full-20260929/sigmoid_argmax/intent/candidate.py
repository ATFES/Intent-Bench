import torch
import intent
import intent.language as I


@intent.kernel
def sigmoid_argmax_lastdim_kernel(
    x: I.In[I.f32, ("M", "N")],
    out: I.Out[I.i64, ("M",)],
):
    M, N = x.shape
    rows = I.domain(0, M)
    cols = I.domain(0, N)

    sig = I.sigmoid(x[rows, cols])
    max_values, idx = I.arg_reduce.max(sig, axis=1)
    out[rows] = I.cast(idx, I.i64)


def build(context):
    artifact = context.compile(
        "sigmoid_argmax_lastdim",
        sigmoid_argmax_lastdim_kernel,
    )

    def sigmoid_argmax(input, dim=None, keepdim=False):
        if keepdim:
            raise ValueError("keepdim=True is not supported by this profile")
        if dim is None:
            raise ValueError("dim=None is not supported by this profile")

        rank = input.dim()
        axis = dim if dim >= 0 else dim + rank
        if axis != rank - 1:
            raise ValueError(
                "this build supports only the supplied last-dimension profile"
            )

        out = torch.empty(
            (input.shape[0],),
            dtype=torch.int64,
            device=input.device,
        )
        artifact(input, out)
        return out

    return sigmoid_argmax
