import torch
import intent
import intent.language as I

# SELU constants
SELU_ALPHA = 1.6732632423543772847450425849686
SELU_SCALE = 1.0507009873554804934193349852946


@intent.kernel
def conv_selu_kernel(
    input: I.In[I.f32, (2, 16, 64, 64)],
    weight: I.In[I.f32, (64, 16, 3, 3)],
    bias: I.In[I.f32, (64,)],
    stride: I.Constexpr[int],
    padding: I.Constexpr[int],
    dilation: I.Constexpr[int],
    groups: I.Constexpr[int],
    temp: I.Out[I.f32, (2, 64, 62, 62)]
):
    b_dim = I.domain(0, 2)
    c_dim = I.domain(0, 64)
    h_dim = I.domain(0, 62)
    w_dim = I.domain(0, 62)

    for b in I.parallel(b_dim):
        for c in I.parallel(c_dim):
            for h in I.parallel(h_dim):
                for w in I.parallel(w_dim):
                    acc = I.cast(0.0, I.f32)
                    for c_in in I.domain(0, 16):
                        for kh in I.domain(0, 3):
                            for kw in I.domain(0, 3):
                                in_h = h * stride - padding + kh * dilation
                                in_w = w * stride - padding + kw * dilation
                                acc = acc + input[b, c_in, in_h, in_w] * weight[c, c_in, kh, kw]

                    acc = acc + bias[c]

                    if acc > I.cast(0.0, I.f32):
                        acc = SELU_SCALE * acc
                    else:
                        acc = SELU_SCALE * (SELU_ALPHA * (I.exp(acc) - I.cast(1.0, I.f32)))

                    temp[b, c, h, w] = acc


@intent.kernel
def norm_kernel(
    conv_out: I.In[I.f32, (2, 64, 62, 62)],
    eps: I.Constexpr[float],
    output: I.Out[I.f32, (2, 64, 62, 62)]
):
    b_dim = I.domain(0, 2)
    c_dim = I.domain(0, 64)
    h_dim = I.domain(0, 62)
    w_dim = I.domain(0, 62)

    for b in I.parallel(b_dim):
        for c in I.parallel(c_dim):
            # Compute mean
            mean_sum = I.cast(0.0, I.f32)
            for h in I.parallel(h_dim):
                for w in I.parallel(w_dim):
                    mean_sum = mean_sum + conv_out[b, c, h, w]
            spatial_count = I.cast(62 * 62, I.f32)
            mean = mean_sum / spatial_count

            # Compute variance
            var_sum = I.cast(0.0, I.f32)
            for h in I.parallel(h_dim):
                for w in I.parallel(w_dim):
                    diff = conv_out[b, c, h, w] - mean
                    var_sum = var_sum + diff * diff
            var = var_sum / spatial_count

            # Normalize
            inv_std = I.cast(1.0, I.f32) / I.sqrt(var + I.cast(eps, I.f32))
            for h in I.parallel(h_dim):
                for w in I.parallel(w_dim):
                    norm_val = (conv_out[b, c, h, w] - mean) * inv_std
                    output[b, c, h, w] = norm_val


def build(context):
    conv_artifact = context.compile(
        "conv_selu",
        conv_selu_kernel,
        constexprs={"stride": 1, "padding": 0, "dilation": 1, "groups": 1}
    )

    norm_artifact = context.compile(
        "norm",
        norm_kernel,
        constexprs={"eps": 1e-05}
    )

    def wrapper(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_features=None,
        eps=1e-05,
        momentum=0.1,
        affine=False,
        track_running_stats=False
    ) -> torch.Tensor:
        temp = torch.empty((2, 64, 62, 62), dtype=torch.float32, device="cuda")
        output = torch.empty((2, 64, 62, 62), dtype=torch.float32, device="cuda")

        conv_artifact(input, weight, bias, temp)
        norm_artifact(temp, output)

        return output

    return wrapper

