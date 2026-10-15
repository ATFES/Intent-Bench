import intent
import intent.language as I


@intent.kernel
def _softplus_linear_with_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    M, K = input.shape
    N, K_weight = weight.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    z = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    z = z + bias
    softplus = I.log(I.cast(1.0, I.f32) + I.exp(beta * z)) / beta
    output[rows, columns] = I.select(z > threshold, z, softplus)


@intent.kernel
def _softplus_linear_without_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
):
    M, K = input.shape
    N, K_weight = weight.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    z = I.matmul(input, weight, acc_dtype=I.f32, transpose_rhs=True)
    softplus = I.log(I.cast(1.0, I.f32) + I.exp(beta * z)) / beta
    output[rows, columns] = I.select(z > threshold, z, softplus)


def build(context):
    with_bias = context.compile("softplus_linear_with_bias", _softplus_linear_with_bias)
    without_bias = context.compile("softplus_linear_without_bias", _softplus_linear_without_bias)

    def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
        beta_value = float(beta)
        threshold_value = float(threshold)
        if bias is None:
            return without_bias.run(input, weight, beta_value, threshold_value)
        return with_bias.run(input, weight, bias, beta_value, threshold_value)

    return softplus_linear
