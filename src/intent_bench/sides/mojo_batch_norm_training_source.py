from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    x = torch.randn((32, 64, 4096), dtype=torch.float16)
    weight = torch.randn((64,), dtype=torch.float32)
    bias = torch.randn_like(weight)
    initial_mean = torch.randn_like(weight) * 0.1
    initial_variance = torch.rand_like(weight) + 0.5
    (epsilon, momentum) = (1e-05, 0.1)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function, *, include_running_state=False):
        (mean, variance) = (initial_mean.clone(), initial_variance.clone())
        state = {}

        def prepare():
            mean.copy_(initial_mean)
            variance.copy_(initial_variance)

        def launch():
            outputs = function(x, weight, bias, mean, variance, epsilon, momentum)
            state['output'] = (mean, variance, *outputs) if include_running_state else outputs
        return PreparedLaunch(launch, lambda : state['output'], prepare=prepare)
    return Side(side(runtime.batch_norm_training), (Tolerance(atol=2e-05), Tolerance(atol=2e-05), Tolerance(atol=0.01), Tolerance(atol=2e-05), Tolerance(atol=2e-05)), cuda_graph=False, device_type='cpu', note='既有B32-C64-S4096 f16训练态batch normalization；输出含running mean/variance、归一化值、saved mean/rstd；单NUMA8核，PyTorch CPU reference，完整host调用；运行统计每次恢复且恢复不计时。', owners=locals())
