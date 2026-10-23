from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.normalization.batch_norm import batch_norm_training
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((32, 64, 4096), dtype=torch.float16)
    weight = torch.randn((64,), dtype=torch.float32)
    bias = torch.randn_like(weight)
    initial_mean = torch.randn_like(weight) * 0.1
    initial_variance = torch.rand_like(weight) + 0.5
    (epsilon, momentum) = (1e-05, 0.1)
    artifact = context.compile(batch_norm_training)

    def side():
        (mean, variance) = (initial_mean.clone(), initial_variance.clone())

        def prepare():
            mean.copy_(initial_mean)
            variance.copy_(initial_variance)

        return intent_call(artifact, (x, weight, bias, mean, variance, epsilon, momentum),
                           reset=prepare, observe=lambda outputs: (mean, variance, *outputs))
    return checked_call(side(), (Tolerance(atol=2e-05), Tolerance(atol=2e-05), Tolerance(atol=0.01), Tolerance(atol=2e-05), Tolerance(atol=2e-05)), cuda_graph=False, device_type='cpu', note='既有B32-C64-S4096 f16训练态batch normalization；输出含running mean/variance、归一化值、saved mean/rstd；单NUMA8核，PyTorch CPU reference，prepared native调用；运行统计每次恢复且恢复不计时。', owners=locals())
