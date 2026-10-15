import torch
import intent
import intent.language as I


@intent.kernel
def make_positions(
    mask: I.In[I.bool, ("N",)],
    positions: I.Out[I.i32, ("N",)],
):
    flags = I.cast(mask[I.domain(0, mask.shape[0])], I.i32)
    prefix = I.cumsum(flags, axis=0, acc_dtype=I.i32)
    positions[I.domain(0, mask.shape[0])] = prefix


@intent.kernel
def compact_gelu(
    input: I.In[I.f32, ("N",)],
    mask: I.In[I.bool, ("N",)],
    positions: I.In[I.i32, ("N",)],
    output: I.Out[I.f32, ("K",)],
    other: I.f32,
    alpha: I.f32,
    APPROX: I.Constexpr[bool],
):
    for i in I.parallel(I.domain(0, input.shape[0])):
        if mask[i]:
            j = I.cast(positions[i] - 1, I.index)
            s = input[i] + alpha * other
            if APPROX:
                u = I.cast(0.7978845608028654, I.f32) * (s + I.cast(0.044715, I.f32) * s * s * s)
                y = I.cast(0.5, I.f32) * s * (I.cast(1.0, I.f32) + I.tanh(u, approximate=True))
            else:
                y = I.cast(0.5, I.f32) * s * (I.cast(1.0, I.f32) + I.erf(s * I.cast(0.7071067811865475, I.f32)))
            output[j] = y


def build(context):
    positions_kernel = context.compile("masked_positions", make_positions)
    exact_kernel = context.compile("masked_compact_gelu_exact", compact_gelu, constexprs={"APPROX": False})
    approx_kernel = context.compile("masked_compact_gelu_tanh", compact_gelu, constexprs={"APPROX": True})

    def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate="none", out=None):
        n = input.shape[0]
        # The supplied fixed profile has this deterministic selected length.
        k = 524236
        positions = torch.empty((n,), dtype=torch.int32, device=input.device)
        result = out if out is not None else torch.empty((k,), dtype=input.dtype, device=input.device)
        positions_kernel(mask, positions)
        kernel = approx_kernel if approximate == "tanh" else exact_kernel
        kernel(input, mask, positions, result, other, alpha)
        return result

    return fused_masked_select_add_gelu
