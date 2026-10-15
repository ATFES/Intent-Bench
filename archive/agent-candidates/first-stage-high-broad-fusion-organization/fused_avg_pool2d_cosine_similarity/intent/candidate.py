import torch
import intent
import intent.language as I


@intent.kernel
def _cosine_stage(
    x1: I.In[I.f32, ("N", "C", "H", "W")],
    x2: I.In[I.f32, ("N", "C", "H", "W")],
    result: I.Out[I.f32, ("N", "ONE", "H", "W")],
    eps: I.f32,
):
    n_extent, c_extent, h_extent, w_extent = x1.shape
    n_domain = I.domain(0, n_extent)
    c_domain = I.domain(0, c_extent)
    h_domain = I.domain(0, h_extent)
    w_domain = I.domain(0, w_extent)
    one_domain = I.domain(0, 1)

    x1_values = x1[n_domain, c_domain, h_domain, w_domain]
    x2_values = x2[n_domain, c_domain, h_domain, w_domain]
    numerator = I.reduce.sum(x1_values * x2_values, axis=1, acc_dtype=I.f32)
    x1_norm = I.sqrt(I.reduce.sum(x1_values * x1_values, axis=1, acc_dtype=I.f32))
    x2_norm = I.sqrt(I.reduce.sum(x2_values * x2_values, axis=1, acc_dtype=I.f32))
    denominator = I.maximum(x1_norm, eps) * I.maximum(x2_norm, eps)
    similarity = numerator / denominator
    result[n_domain, one_domain, h_domain, w_domain] = I.reshape(
        similarity, (n_extent, 1, h_extent, w_extent)
    )


@intent.kernel
def _avg_pool_stage(
    source: I.In[I.f32, ("N", "ONE", "H", "W")],
    result: I.Out[I.f32, ("N", "ONE", "OH", "OW")],
    kernel_size: I.i64,
    stride: I.i64,
    padding: I.i64,
):
    n_extent, one_extent, h_extent, w_extent = source.shape
    _, _, oh_extent, ow_extent = result.shape
    n_domain = I.domain(0, n_extent)
    one_domain = I.domain(0, one_extent)
    oh_domain = I.domain(0, oh_extent)
    ow_domain = I.domain(0, ow_extent)
    kh_domain = I.domain(0, I.cast(kernel_size, I.index))
    kw_domain = I.domain(0, I.cast(kernel_size, I.index))
    stride_index = I.cast(stride, I.index)
    padding_index = I.cast(padding, I.index)
    h_limit = I.cast(h_extent, I.index)
    w_limit = I.cast(w_extent, I.index)
    kernel_size_f32 = I.cast(kernel_size, I.f32)
    divisor = kernel_size_f32 * kernel_size_f32

    for n in I.parallel(n_domain):
        for one in I.parallel(one_domain):
            for oh in I.parallel(oh_domain):
                for ow in I.parallel(ow_domain):
                    h_start = oh * stride_index - padding_index
                    w_start = ow * stride_index - padding_index
                    total = I.cast(0.0, I.f32)
                    for kh in kh_domain:
                        input_h = h_start + kh
                        for kw in kw_domain:
                            input_w = w_start + kw
                            if input_h >= 0:
                                if input_h < h_limit:
                                    if input_w >= 0:
                                        if input_w < w_limit:
                                            total = total + source[
                                                n, one, input_h, input_w
                                            ]
                    result[n, one, oh, ow] = total / divisor


def build(context):
    cosine_artifact = context.compile("cosine_stage", _cosine_stage, constexprs={})
    pool_artifact = context.compile("avg_pool_stage", _avg_pool_stage, constexprs={})

    def fused_avg_pool2d_cosine_similarity(
        x1: torch.Tensor,
        x2: torch.Tensor,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if stride is None:
            stride = kernel_size

        n, _, height, width = x1.shape
        pooled_height = (height + 2 * padding - kernel_size) // stride + 1
        pooled_width = (width + 2 * padding - kernel_size) // stride + 1
        cosine = torch.empty(
            (n, 1, height, width), device=x1.device, dtype=x1.dtype
        )
        result = torch.empty(
            (n, 1, pooled_height, pooled_width),
            device=x1.device,
            dtype=x1.dtype,
        )
        cosine_artifact(x1, x2, cosine, eps)
        pool_artifact(cosine, result, kernel_size, stride, padding)
        return result

    return fused_avg_pool2d_cosine_similarity
