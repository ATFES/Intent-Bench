from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _cos_avg_pool_kernel(
    values: I.In[I.f32, ("BATCH", "CHANNELS", "WIDTH")],
    output: I.Out[I.f32, ("BATCH", "CHANNELS", "OUTPUT_WIDTH")],
    stride: I.i64,
    padding: I.i64,
    kernel_size: I.i64,
    COUNT_INCLUDE_PAD: I.Constexpr[bool],
):
    stride_index = I.cast(stride, I.index)
    padding_index = I.cast(padding, I.index)
    width = values.shape[2]

    for batch in I.parallel(I.domain(0, values.shape[0])):
        for channel in I.parallel(I.domain(0, values.shape[1])):
            cosine_values = I.cos(values[batch, channel, :])

            for output_position in I.parallel(I.domain(0, output.shape[2])):
                window_start = output_position * stride_index - padding_index
                offsets = I.indices(I.domain(0, kernel_size))
                source_positions = window_start + offsets
                valid = I.full(offsets.shape, True, I.bool)
                valid = I.mask(valid, source_positions >= 0, False)
                valid = I.mask(valid, source_positions < width, False)
                window_values = I.gather(
                    cosine_values,
                    source_positions,
                    valid=valid,
                    fill=I.cast(0.0, I.f32),
                )
                total = I.reduce.sum(window_values, axis=0, acc_dtype=I.f32)

                if COUNT_INCLUDE_PAD:
                    divisor = I.cast(kernel_size, I.f32)
                else:
                    ones = I.full(cosine_values.shape, 1.0, I.f32)
                    valid_counts = I.gather(
                        ones,
                        source_positions,
                        valid=valid,
                        fill=I.cast(0.0, I.f32),
                    )
                    divisor = I.reduce.sum(valid_counts, axis=0, acc_dtype=I.f32)

                output[batch, channel, output_position] = I.fdiv(total, divisor)


def _output_width(
    input_width: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
) -> int:
    numerator = input_width + 2 * padding - kernel_size
    if ceil_mode:
        width = -(-numerator // stride) + 1
        if (width - 1) * stride >= input_width + padding:
            width -= 1
    else:
        if numerator < 0:
            return 0
        width = numerator // stride + 1

    return max(width, 0)


def build(context):
    include_pad = context.compile(
        "cos_avg_pool_include_pad",
        _cos_avg_pool_kernel,
        constexprs={"COUNT_INCLUDE_PAD": True},
    )
    exclude_pad = context.compile(
        "cos_avg_pool_exclude_pad",
        _cos_avg_pool_kernel,
        constexprs={"COUNT_INCLUDE_PAD": False},
    )

    def cos_avg_pool1d(
        input: torch.Tensor,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> torch.Tensor:
        if stride is None:
            stride = kernel_size

        batch_size, channels, input_width = input.shape
        output_width = _output_width(
            input_width,
            kernel_size,
            stride,
            padding,
            bool(ceil_mode),
        )
        output = torch.empty(
            (batch_size, channels, output_width),
            dtype=input.dtype,
            device=input.device,
        )
        artifact = include_pad if count_include_pad else exclude_pad
        artifact(
            input,
            output,
            stride,
            padding,
            kernel_size,
        )
        return output

    return cos_avg_pool1d
