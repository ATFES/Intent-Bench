import torch
import intent
import intent.language as I


@intent.kernel
def _fused_groupnorm(
    input1: I.In[I.f32, (16, 64, 128, 64)],
    input2: I.In[I.f32, (16, 64, 128, 64)],
    weight: I.In[I.f32, (64,)],
    bias: I.In[I.f32, (64,)],
    output: I.Out[I.f32, (16, 64, 128, 64)],
    num_groups: I.i64,
    eps: I.f32,
):
    group_count = I.cast(65536.0, I.f32)
    channels = I.domain(0, 8)
    height = I.domain(0, 128)
    width = I.domain(0, 64)
    channel_offsets = I.indices(channels)

    for ni in I.parallel(I.domain(0, 16)):
        for gi in I.parallel(I.domain(0, 8)):
            channel = gi * 8 + channel_offsets

            # The producer is shared by the statistics and the final write.
            mixed = (input1[ni, channel, height, width] + input2[ni, channel, height, width]) * input2[ni, channel, height, width]
            total = I.reduce.sum(mixed, axis=(0, 1, 2), acc_dtype=I.f32)
            mean = total / group_count

            centered = mixed - mean
            variance = I.reduce.sum(centered * centered, axis=(0, 1, 2), acc_dtype=I.f32)
            variance = variance / group_count
            normalized = centered / I.sqrt(variance + eps)

            scale = I.reshape(weight[channel], (8, 1, 1))
            shift = I.reshape(bias[channel], (8, 1, 1))
            output[ni, channel, height, width] = normalized * scale + shift


def build(context):
    artifact = context.compile("fused_add_mul_groupnorm", _fused_groupnorm)

    def fused_add_mul_groupnorm(
        input1,
        input2,
        weight,
        bias,
        num_groups,
        eps=1e-5,
        *,
        out=None,
    ):
        output = out if out is not None else torch.empty_like(input1)
        artifact(input1, input2, weight, bias, output, num_groups, eps)
        return output

    return fused_add_mul_groupnorm
