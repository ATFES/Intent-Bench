import intent
import intent.language as I


@intent.kernel
def cosine_avg_pool_cosine_kernel(
    x1: I.In[I.f32, ("BATCH", "CHANNELS", "HEIGHT", "WIDTH")],
    x2: I.In[I.f32, ("BATCH", "CHANNELS", "HEIGHT", "WIDTH")],
    cosine: I.Out[I.f32, ("BATCH", "PADDED_HEIGHT", "PADDED_WIDTH")],
    epsilon: I.f32,
    PADDING: I.Constexpr[int],
):
    batch_points = I.domain(0, x1.shape[0])
    padded_rows = I.domain(0, cosine.shape[1])
    padded_columns = I.domain(0, cosine.shape[2])

    for batch in I.parallel(batch_points):
        for padded_row in I.parallel(padded_rows):
            for padded_column in I.parallel(padded_columns):
                source_row = padded_row - PADDING
                source_column = padded_column - PADDING
                similarity = I.cast(0.0, I.f32)

                if source_row >= 0:
                    if source_row < x1.shape[2]:
                        if source_column >= 0:
                            if source_column < x1.shape[3]:
                                channel_points = I.domain(0, x1.shape[1])
                                left = x1[batch, channel_points, source_row, source_column]
                                right = x2[batch, channel_points, source_row, source_column]
                                numerator = I.reduce.sum(
                                    left * right,
                                    axis=0,
                                    acc_dtype=I.f32,
                                )
                                left_norm_squared = I.reduce.sum(
                                    left * left,
                                    axis=0,
                                    acc_dtype=I.f32,
                                )
                                right_norm_squared = I.reduce.sum(
                                    right * right,
                                    axis=0,
                                    acc_dtype=I.f32,
                                )
                                denominator = I.maximum(
                                    I.sqrt(left_norm_squared)
                                    * I.sqrt(right_norm_squared),
                                    epsilon,
                                )
                                similarity = numerator / denominator

                cosine[batch, padded_row, padded_column] = similarity


@intent.kernel
def cosine_avg_pool_pool_kernel(
    cosine: I.In[I.f32, ("BATCH", "PADDED_HEIGHT", "PADDED_WIDTH")],
    pooled: I.Out[I.f32, ("BATCH", 1, "POOLED_HEIGHT", "POOLED_WIDTH")],
    KERNEL_SIZE: I.Constexpr[int],
    STRIDE: I.Constexpr[int],
    PADDING: I.Constexpr[int],
):
    inverse_pool_area = I.cast(
        1.0 / float(KERNEL_SIZE * KERNEL_SIZE),
        I.f32,
    )
    batch_points = I.domain(0, cosine.shape[0])
    pooled_rows = I.domain(0, pooled.shape[2])
    pooled_columns = I.domain(0, pooled.shape[3])

    for batch in I.parallel(batch_points):
        for pooled_row in I.parallel(pooled_rows):
            for pooled_column in I.parallel(pooled_columns):
                kernel_rows = I.domain(0, KERNEL_SIZE)
                kernel_columns = I.domain(0, KERNEL_SIZE)
                row_base = pooled_row * STRIDE - PADDING
                column_base = pooled_column * STRIDE - PADDING
                window = cosine[
                    batch,
                    row_base + kernel_rows,
                    column_base + kernel_columns,
                ]
                window_sum = I.reduce.sum(
                    window,
                    axis=(0, 1),
                    acc_dtype=I.f32,
                )
                pooled[batch, 0, pooled_row, pooled_column] = (
                    window_sum * inverse_pool_area
                )


def build(context):
    cosine_kernel = context.compile(
        "cosine_avg_pool_cosine_stage",
        cosine_avg_pool_cosine_kernel,
        constexprs={"PADDING": 1},
    )
    pooling_kernel = context.compile(
        "cosine_avg_pool_pooling_stage",
        cosine_avg_pool_pool_kernel,
        constexprs={
            "KERNEL_SIZE": 3,
            "STRIDE": 2,
            "PADDING": 1,
        },
    )

    def fused_avg_pool2d_cosine_similarity(
        x1,
        x2,
        kernel_size,
        stride=None,
        padding=0,
        eps=1e-08,
    ):
        batch_size, _, height, width = x1.shape
        effective_stride = kernel_size if stride is None else stride
        padded_height = height + 2 * padding
        padded_width = width + 2 * padding
        pooled_height = (
            padded_height - kernel_size
        ) // effective_stride + 1
        pooled_width = (padded_width - kernel_size) // effective_stride + 1
        cosine = x1.new_empty(
            (batch_size, padded_height, padded_width),
        )
        pooled = x1.new_empty(
            (batch_size, 1, pooled_height, pooled_width),
        )
        cosine_kernel(x1, x2, cosine, eps)
        pooling_kernel(cosine, pooled)
        return pooled

    return fused_avg_pool2d_cosine_similarity
