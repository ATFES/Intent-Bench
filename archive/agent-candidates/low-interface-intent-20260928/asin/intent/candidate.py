import torch
import intent
import intent.language as I


@intent.kernel
def _asin_kernel(
    input_tensor: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    values = I.domain(0, input_tensor.shape[0])
    for i in I.parallel(values):
        x = I.cast(input_tensor[i], I.f32)
        ax = I.abs(x)
        # Minimax-style approximation, accurate over the complete unit interval.
        t = I.sqrt(I.cast(1.0, I.f32) - ax)
        p = I.cast(-0.0187293, I.f32)
        p = p * ax + I.cast(0.0742610, I.f32)
        p = p * ax + I.cast(-0.2121144, I.f32)
        p = p * ax + I.cast(1.5707288, I.f32)
        magnitude = I.cast(1.5707963267948966, I.f32) - p * t
        result = I.select(x < I.cast(0.0, I.f32), -magnitude, magnitude)
        valid = (x >= I.cast(-1.0, I.f32)) & (x <= I.cast(1.0, I.f32))
        nan_value = I.cast(0.0, I.f32) / I.cast(0.0, I.f32)
        result = I.select(valid, result, nan_value)
        output[i] = I.cast(result, I.f16)


def build(context):
    compiled = context.compile("asin_pointwise", _asin_kernel)

    def asin(input_tensor):
        output = torch.empty_like(input_tensor)
        compiled(input_tensor, output)
        return output

    return asin
