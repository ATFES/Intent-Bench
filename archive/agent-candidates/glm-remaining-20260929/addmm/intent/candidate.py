import intent
import intent.language as I


@intent.kernel
def addmm_kernel(
    input: I.In[I.f16, ("M", "N")],
    mat1: I.In[I.f16, ("M", "K")],
    mat2: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
    beta: I.f64,
    alpha: I.f64,
) -> None:
    input_value = input[:, :]
    mat1_value = mat1[:, :]
    mat2_value = mat2[:, :]
    product = I.matmul(mat1_value, mat2_value, acc_dtype=I.f32)
    alpha_value = I.cast(alpha, I.f32)
    scaled_product = product * alpha_value
    if beta != 0.0:
        scaled_input = I.cast(input_value, I.f32)
        beta_value = I.cast(beta, I.f32)
        result = scaled_product + beta_value * scaled_input
    else:
        result = scaled_product
    output[:, :] = I.cast(result, I.f16)


def build(context):
    artifact = context.compile("addmm_f16_contract", addmm_kernel)

    def addmm(input, mat1, mat2, beta=1, alpha=1, out=None):
        if out is None:
            output = input.new_empty((mat1.shape[0], mat2.shape[1]))
        else:
            output = out
        artifact(input, mat1, mat2, output, beta, alpha)
        return output

    return addmm
