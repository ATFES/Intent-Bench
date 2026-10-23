from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.convolution.direct import causal_depthwise_conv1d_update
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((64, 4096), dtype=torch.float16) * 0.1
    initial = torch.randn((64, 4096, 4), dtype=torch.float16) * 0.1
    weight = torch.randn((4096, 4), dtype=torch.float16) * 0.1
    bias = torch.randn((4096,), dtype=torch.float16) * 0.1
    artifact = context.compile(causal_depthwise_conv1d_update, constexprs={'SILU': True})

    def side():
        state = initial.clone()
        return intent_call(artifact, (x, state, weight, bias), reset=lambda: state.copy_(initial),
                           observe=lambda output: (state, output))
    return checked_call(side(), (Tolerance(atol=0.0), Tolerance(atol=0.002)), cuda_graph=False, device_type='cpu', note='既有B64-D4096-W4 f16缓存卷积更新及SiLU；单NUMA8核，PyTorch eager同算法，prepared native调用；每次恢复同一state且恢复不计时。', owners=locals())
