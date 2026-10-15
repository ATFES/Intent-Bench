from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    x = torch.randn((64, 4096), dtype=torch.float16) * 0.1
    initial = torch.randn((64, 4096, 4), dtype=torch.float16) * 0.1
    weight = torch.randn((4096, 4), dtype=torch.float16) * 0.1
    bias = torch.randn((4096,), dtype=torch.float16) * 0.1
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function, *, include_state=False):
        state = initial.clone()
        result = {}

        def launch():
            output = function(x, state, weight, bias)
            result['output'] = (state, output) if include_state else output
        return PreparedLaunch(launch, lambda : result['output'], prepare=lambda : state.copy_(initial))
    return Side(side(runtime.causal_conv_update), (Tolerance(atol=0.0), Tolerance(atol=0.002)), cuda_graph=False, device_type='cpu', note='既有B64-D4096-W4 f16缓存卷积更新及SiLU；单NUMA8核，PyTorch eager同算法，完整host调用；每次恢复同一state且恢复不计时。', owners=locals())
