import torch
import intent
import intent.language as I


@intent.kernel
def _cos_avg_pool1d_kernel(
    input: I.In[I.f32, (32, 32, 1536)],
    output: I.Out[I.f32, (32, 32, 512)],
):
    batches = I.domain(0, 32)
    channels = I.domain(0, 32)
    outputs = I.domain(0, 512)
    window = I.domain(0, 3)

    output_indices = I.reshape(I.indices(outputs), (512, 1))
    window_indices = I.reshape(I.indices(window), (1, 3))
    input_indices = output_indices * 3 + window_indices

    values = I.cos(input[batches, channels, input_indices])
    pooled = I.reduce.sum(values, axis=3, acc_dtype=I.f32)
    output[batches, channels, outputs] = pooled / I.cast(3, I.f32)


def build(context):
    artifact = context.compile("cos_avg_pool1d", _cos_avg_pool1d_kernel, constexprs={})

    def cos_avg_pool1d(
        input: torch.Tensor,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> torch.Tensor:
        return artifact.run(input)

    return cos_avg_pool1d
