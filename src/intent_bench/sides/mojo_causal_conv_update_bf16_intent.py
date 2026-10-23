from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.convolution.direct import causal_depthwise_conv1d_update_bf16
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((32, 4096), dtype=torch.bfloat16)
    initial = torch.randn((32, 4096, 4), dtype=torch.bfloat16)
    weight = torch.randn((4096, 4), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    artifact = context.compile(causal_depthwise_conv1d_update_bf16, constexprs={'SILU': True})

    def side():
        state = initial.clone()
        return intent_call(artifact, (x, state, weight, bias), reset=lambda: state.copy_(initial),
                           observe=lambda output: (state, output))
    return checked_call(side(), (Tolerance(atol=0.0), Tolerance(atol=0.05, rtol=0.05)), cuda_graph=False, device_type='cpu', note='既有 B32-D4096-W4 bf16 state/input、f32 weight/bias、SiLU；完整缓存更新及输出，PyTorch CPU reference，沿用原容差；单 NUMA 8 核，prepared native调用，每次恢复 state 且恢复不计时。', owners=locals())
