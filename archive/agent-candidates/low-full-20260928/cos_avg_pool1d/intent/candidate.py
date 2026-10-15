import torch
import intent
import intent.language as I


@intent.kernel
def _cosine(
    x: I.In[I.f32, ("B", "C", "W")],
    y: I.Out[I.f32, ("B", "C", "W")],
):
    B, C, W = x.shape
    for b in I.parallel(I.domain(0, B)):
        for c in I.parallel(I.domain(0, C)):
            for w in I.parallel(I.domain(0, W)):
                y[b, c, w] = I.cos(x[b, c, w])


@intent.kernel
def _pool1d(
    x: I.In[I.f32, ("B", "C", "W")],
    y: I.Out[I.f32, ("B", "C", "OW")],
    kernel_size: I.i64,
    stride: I.i64,
    padding: I.i64,
    count_include_pad: I.bool,
):
    B, C, W = x.shape
    _, _, OW = y.shape
    ks = I.cast(kernel_size, I.index)
    st = I.cast(stride, I.index)
    pad = I.cast(padding, I.index)
    for b in I.parallel(I.domain(0, B)):
        for c in I.parallel(I.domain(0, C)):
            for ow in I.parallel(I.domain(0, OW)):
                start = ow * st - pad
                total = I.cast(0.0, I.f32)
                valid = I.cast(0, I.index)
                for k in range(kernel_size):
                    ix = start + I.cast(k, I.index)
                    if ix >= 0 and ix < W:
                        total = total + x[b, c, ix]
                        valid = valid + 1
                divisor = I.select(count_include_pad, ks, valid)
                y[b, c, ow] = total / I.cast(divisor, I.f32)


def build(context):
    cosine_artifact = context.compile("cosine_stage", _cosine)
    pool_artifact = context.compile("pool_stage", _pool1d)

    def cos_avg_pool1d(
        input: torch.Tensor,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> torch.Tensor:
        actual_stride = kernel_size if stride is None else stride
        _, _, width = input.shape
        numerator = width + 2 * padding - kernel_size
        if ceil_mode:
            out_width = (numerator + actual_stride - 1) // actual_stride + 1
            while out_width > 0 and (out_width - 1) * actual_stride >= width + padding:
                out_width -= 1
        else:
            out_width = numerator // actual_stride + 1
        cosine = torch.empty_like(input)
        output = torch.empty(
            (input.shape[0], input.shape[1], out_width),
            device=input.device,
            dtype=input.dtype,
        )
        cosine_artifact(input, cosine)
        pool_artifact(
            cosine,
            output,
            kernel_size,
            actual_stride,
            padding,
            count_include_pad,
        )
        return output

    return cos_avg_pool1d
