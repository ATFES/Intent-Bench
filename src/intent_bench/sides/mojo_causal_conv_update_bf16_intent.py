from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.convolution.direct import causal_depthwise_conv1d_update_bf16
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    x = torch.randn((32, 4096), dtype=torch.bfloat16)
    initial = torch.randn((32, 4096, 4), dtype=torch.bfloat16)
    weight = torch.randn((4096, 4), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    artifact = intent.compile(causal_depthwise_conv1d_update_bf16, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'SILU': True})

    def side(function, *, include_state=False):
        state = initial.clone()
        result = {}

        def launch():
            output = function(x, state, weight, bias)
            result['output'] = (state, output) if include_state else output
        return PreparedLaunch(launch, lambda : result['output'], prepare=lambda : state.copy_(initial))
    return Side(side(artifact.run, include_state=True), (Tolerance(atol=0.0), Tolerance(atol=0.05, rtol=0.05)), cuda_graph=False, device_type='cpu', note='既有 B32-D4096-W4 bf16 state/input、f32 weight/bias、SiLU；完整缓存更新及输出，PyTorch CPU reference，沿用原容差；单 NUMA 8 核，完整 host 调用，每次恢复 state 且恢复不计时。', owners=locals())
