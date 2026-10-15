import torch
import intent
import intent.language as I


@intent.kernel
def gelu_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    APPROXIMATE: I.Constexpr[bool],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        x = input[i]
        if APPROXIMATE:
            x2 = x * x
            x3 = x2 * x
            inner = I.cast(0.044715, I.f32) * x3 + x
            scaled = I.cast(0.7978845608028654, I.f32) * inner
            value = I.cast(0.5, I.f32) * x * (
                I.cast(1.0, I.f32) + I.tanh(scaled)
            )
        else:
            value = x * I.cast(0.5, I.f32) * I.erfc(
                -x * I.cast(0.7071067811865475, I.f32)
            )
        output[i] = value


def build(context):
    exact = context.compile("gelu_exact", gelu_kernel, constexprs={"APPROXIMATE": False})
    approximate_artifact = context.compile(
        "gelu_approximate", gelu_kernel, constexprs={"APPROXIMATE": True}
    )

    def gelu(input, approximate="none"):
        output = torch.empty_like(input)
        if approximate == "tanh":
            approximate_artifact(input, output)
        else:
            exact(input, output)
        return output

    return gelu
