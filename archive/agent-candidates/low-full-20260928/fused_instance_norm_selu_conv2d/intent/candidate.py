import torch
import intent
import intent.language as I


@intent.kernel
def _conv_selu(
    x: I.In[I.f32, (2, 16, 64, 64)],
    w: I.In[I.f32, (64, 16, 3, 3)],
    b: I.In[I.f32, (64,)],
    y: I.Out[I.f32, (2, 64, 62, 62)],
):
    for n in I.parallel(I.domain(0, 2)):
        for oc in I.parallel(I.domain(0, 64)):
            for oh in I.parallel(I.domain(0, 62)):
                for ow in I.parallel(I.domain(0, 62)):
                    acc = b[oc]
                    for ic in range(16):
                        for kh in range(3):
                            for kw in range(3):
                                acc = acc + x[n, ic, oh + kh, ow + kw] * w[oc, ic, kh, kw]
                    neg = I.cast(1.6732632423543772, I.f32) * (I.exp(acc) - I.cast(1.0, I.f32))
                    val = I.select(acc > I.cast(0.0, I.f32), I.cast(1.0507009873554805, I.f32) * acc, neg)
                    y[n, oc, oh, ow] = val


@intent.kernel
def _instance_stats(
    x: I.In[I.f32, (2, 64, 62, 62)],
    mean: I.Out[I.f32, (2, 64)],
    var: I.Out[I.f32, (2, 64)],
):
    for n in I.parallel(I.domain(0, 2)):
        for c in I.parallel(I.domain(0, 64)):
            values = x[n, c, :, :]
            total = I.reduce.sum(values, axis=(0, 1))
            squares = I.reduce.sum(values * values, axis=(0, 1))
            m = total / I.cast(3844.0, I.f32)
            mean[n, c] = m
            var[n, c] = squares / I.cast(3844.0, I.f32) - m * m


@intent.kernel
def _normalize(
    x: I.In[I.f32, (2, 64, 62, 62)],
    mean: I.In[I.f32, (2, 64)],
    var: I.In[I.f32, (2, 64)],
    y: I.Out[I.f32, (2, 64, 62, 62)],
    eps: I.f32,
):
    for n in I.parallel(I.domain(0, 2)):
        for c in I.parallel(I.domain(0, 64)):
            denom = I.sqrt(var[n, c] + eps)
            for h in I.parallel(I.domain(0, 62)):
                for q in I.parallel(I.domain(0, 62)):
                    y[n, c, h, q] = (x[n, c, h, q] - mean[n, c]) / denom


def build(context):
    conv = context.compile("conv_selu", _conv_selu)
    stats = context.compile("instance_stats", _instance_stats)
    norm = context.compile("normalize", _normalize)

    def fused_instance_norm_selu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_features=None,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ) -> torch.Tensor:
        if bias is None:
            bias = torch.zeros((64,), device=input.device, dtype=input.dtype)
        workspace = torch.empty((2, 64, 62, 62), device=input.device, dtype=input.dtype)
        mean = torch.empty((2, 64), device=input.device, dtype=input.dtype)
        var = torch.empty((2, 64), device=input.device, dtype=input.dtype)
        output = torch.empty((2, 64, 62, 62), device=input.device, dtype=input.dtype)
        conv(input, weight, bias, workspace)
        stats(workspace, mean, var)
        norm(workspace, mean, var, output, float(eps))
        return output

    return fused_instance_norm_selu_conv2d
