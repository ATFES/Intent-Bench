import torch
import intent
import intent.language as I


@intent.kernel
def batch_norm_stats_kernel(
    x: I.In[I.f16, (32, 32, 640, 128)],
    running_mean: I.In[I.f32, (32,)],
    running_var: I.In[I.f32, (32,)],
    training: I.Constexpr[bool],
    momentum: I.Constexpr[float],
    eps: I.Constexpr[float],
    new_running_mean: I.Out[I.f32, (32,)],
    new_running_var: I.Out[I.f32, (32,)],
    out_mean: I.Out[I.f32, (32,)],
    out_var: I.Out[I.f32, (32,)],
):
    N, C, H, W = 32, 32, 640, 128
    inv_count = I.cast(1.0 / (N * H * W), I.f32)

    x_f32 = I.cast(x, I.f32)

    batch_mean = I.reduce.sum(x_f32, axis=(0, 2, 3)) * inv_count

    mean_bcast = I.reshape(batch_mean, (1, C, 1, 1))
    centered = x_f32 - mean_bcast
    centered_sq = centered * centered
    batch_var = I.reduce.sum(centered_sq, axis=(0, 2, 3)) * inv_count

    mom = I.cast(momentum, I.f32)
    one_minus_mom = I.cast(1.0 - momentum, I.f32)
    updated_rm = running_mean * mom + batch_mean * one_minus_mom
    updated_rv = running_var * mom + batch_var * one_minus_mom

    mean_to_use = I.select(training, updated_rm, running_mean)
    var_to_use = I.select(training, updated_rv, running_var)

    new_running_mean[:] = updated_rm
    new_running_var[:] = updated_rv
    out_mean[:] = mean_to_use
    out_var[:] = var_to_use


@intent.kernel
def batch_norm_normalize_kernel(
    x: I.In[I.f16, (32, 32, 640, 128)],
    mean: I.In[I.f32, (32,)],
    var: I.In[I.f32, (32,)],
    weight: I.In[I.f32, (32,)],
    bias: I.In[I.f32, (32,)],
    eps: I.Constexpr[float],
    y: I.Out[I.f16, (32, 32, 640, 128)],
):
    C = 32

    x_f32 = I.cast(x, I.f32)

    mean_b = I.reshape(mean, (1, C, 1, 1))
    var_b = I.reshape(var, (1, C, 1, 1))
    weight_b = I.reshape(weight, (1, C, 1, 1))
    bias_b = I.reshape(bias, (1, C, 1, 1))

    inv_std = I.rsqrt(var_b + I.cast(eps, I.f32))
    normalized = (x_f32 - mean_b) * inv_std

    y_f32 = normalized * weight_b + bias_b

    y[:] = I.cast(y_f32, I.f16)


def build(context):
    train_stats = context.compile(
        "batch_norm_stats_train",
        batch_norm_stats_kernel,
        constexprs={"training": True, "momentum": 0.1, "eps": 1e-05},
    )

    infer_stats = context.compile(
        "batch_norm_stats_infer",
        batch_norm_stats_kernel,
        constexprs={"training": False, "momentum": 0.1, "eps": 1e-05},
    )

    norm = context.compile(
        "batch_norm_normalize",
        batch_norm_normalize_kernel,
        constexprs={"eps": 1e-05},
    )

    def batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
    ):
        output = torch.empty_like(input)
        used_mean = torch.empty(32, dtype=torch.float32, device=input.device)
        used_var = torch.empty(32, dtype=torch.float32, device=input.device)

        if weight is None:
            w = torch.ones(32, dtype=torch.float32, device=input.device)
        else:
            w = weight

        if bias is None:
            b = torch.zeros(32, dtype=torch.float32, device=input.device)
        else:
            b = bias

        if training:
            new_running_mean = torch.empty(32, dtype=torch.float32, device=input.device)
            new_running_var = torch.empty(32, dtype=torch.float32, device=input.device)
            train_stats(
                input,
                running_mean,
                running_var,
                new_running_mean,
                new_running_var,
                used_mean,
                used_var,
            )
            running_mean.copy_(new_running_mean)
            running_var.copy_(new_running_var)
        else:
            used_mean.copy_(running_mean)
            used_var.copy_(running_var)

        norm(input, used_mean, used_var, w, b, output)

        return output

    return batch_norm

