from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.convolution.direct import causal_depthwise_conv1d_update
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    x = torch.randn((64, 4096), dtype=torch.float16) * 0.1
    initial = torch.randn((64, 4096, 4), dtype=torch.float16) * 0.1
    weight = torch.randn((4096, 4), dtype=torch.float16) * 0.1
    bias = torch.randn((4096,), dtype=torch.float16) * 0.1
    artifact = context.compile(causal_depthwise_conv1d_update, constexprs={'SILU': True})

    def side(function, *, include_state=False):
        state = initial.clone()
        result = {}

        def launch():
            output = function(x, state, weight, bias)
            result['output'] = (state, output) if include_state else output
        return PreparedCall(launch, lambda : result['output'], reset=lambda : state.copy_(initial))
    return checked_call(side(artifact.run, include_state=True), (Tolerance(atol=0.0), Tolerance(atol=0.002)), cuda_graph=False, device_type='cpu', note='既有B64-D4096-W4 f16缓存卷积更新及SiLU；单NUMA8核，PyTorch eager同算法，完整host调用；每次恢复同一state且恢复不计时。', owners=locals())
