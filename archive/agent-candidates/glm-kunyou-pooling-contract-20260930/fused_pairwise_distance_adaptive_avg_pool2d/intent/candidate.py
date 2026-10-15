import torch
import intent
import intent.language as I


@intent.kernel
def pooled_pairwise_distance(x1_pooled: I.In[I.f32, (16, 3, 16, 16)],
                               x2_pooled: I.In[I.f32, (16, 3, 16, 16)],
                               p: I.f32,
                               eps: I.f32,
                               keepdim: I.Constexpr[bool],
                               result: I.Out[I.f32, (16,)]):
    diff = x1_pooled - x2_pooled
    abs_diff = I.abs(diff)
    abs_diff_p = I.exp(p * I.log(abs_diff))
    sum_p = I.reduce.sum(abs_diff_p, axis=(1, 2, 3))
    distance = I.exp(I.log(sum_p + eps) / p)
    result[:] = I.full((16,), distance, I.f32)


def build(context):
    artifact = context.compile(
        "fused_pairwise_distance_adaptive_avg_pool2d",
        pooled_pairwise_distance,
        constexprs={"p": 2.0, "eps": 1e-6, "keepdim": False}
    )

    def wrapper(x1, x2, output_size, p=2.0, eps=1e-6, keepdim=False):
        x1_pooled = torch.nn.functional.adaptive_avg_pool2d(x1, output_size)
        x2_pooled = torch.nn.functional.adaptive_avg_pool2d(x2, output_size)
        result = torch.empty(x1.shape[0], dtype=torch.float32, device=x1.device)
        artifact(x1_pooled, x2_pooled, p, eps, result)
        if keepdim:
            result = result.unsqueeze(-1)
        return result

    return wrapper

