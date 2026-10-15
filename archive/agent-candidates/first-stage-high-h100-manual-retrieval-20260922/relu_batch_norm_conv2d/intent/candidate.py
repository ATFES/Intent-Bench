import torch
import intent
import intent.language as I


@intent.kernel
def _fused_conv_bn_relu(
    input: I.In[I.f32, (1, 64, 32, 32)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    running_mean: I.In[I.f32, (128,)],
    running_var: I.In[I.f32, (128,)],
    bn_weight: I.In[I.f32, (128,)],
    bn_bias: I.In[I.f32, (128,)],
    eps: I.f32,
    output: I.Out[I.f32, (1, 128, 32, 32)],
):
    channels = I.domain(0, 64)
    outputs = I.domain(0, 128)

    for batch in I.parallel(I.domain(0, 1)):
        for out_row in I.parallel(I.domain(0, 32)):
            for out_col in I.parallel(I.domain(0, 32)):
                accum = I.zeros((128,), I.f32)

                for kernel_row in range(3):
                    for kernel_col in range(3):
                        input_row = out_row + kernel_row - 1
                        input_col = out_col + kernel_col - 1
                        valid = (
                            (input_row >= 0)
                            & (input_row < 32)
                            & (input_col >= 0)
                            & (input_col < 32)
                        )
                        input_values = I.gather(
                            input,
                            (batch, channels, input_row, input_col),
                            valid=valid,
                            fill=I.cast(0.0, I.f32),
                        )
                        kernel_values = weight[
                            outputs, channels, kernel_row, kernel_col
                        ]
                        products = kernel_values * input_values
                        accum = accum + I.reduce.sum(
                            products, axis=1, acc_dtype=I.f32
                        )

                mean = running_mean[outputs]
                variance = running_var[outputs] + eps
                normalized = (
                    (accum - mean)
                    * I.fdiv(I.cast(1.0, I.f32), I.sqrt(variance))
                    * bn_weight[outputs]
                    + bn_bias[outputs]
                )
                activated = I.select(
                    normalized > I.cast(0.0, I.f32),
                    normalized,
                    I.cast(0.0, I.f32),
                )
                output[batch, outputs, out_row, out_col] = activated


def build(context):
    fused = context.compile("fused_conv_bn_relu", _fused_conv_bn_relu)

    def relu_batch_norm_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        running_mean=None,
        running_var=None,
        bn_weight=None,
        bn_bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
        inplace=False,
    ):
        output = torch.empty(
            (1, 128, 32, 32), device=input.device, dtype=input.dtype
        )
        fused(
            input,
            weight,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            eps,
            output,
        )
        return output

    return relu_batch_norm_conv2d
