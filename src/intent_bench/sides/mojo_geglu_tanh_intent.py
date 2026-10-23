from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.activation.pointwise import geglu_tanh
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((4096, 2 * 14336), dtype=torch.float16)
    artifact = context.compile(geglu_tanh)

    def side():
        output = torch.empty((4096, 14336), dtype=torch.float16)
        return intent_call(artifact, (x, output), observe=lambda _: output)
    return checked_call(side(), Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, device_type='cpu', note='既有4096x28672 f16 GEGLU，独立InOut输出，prepared native调用；单NUMA8核，PyTorch CPU reference。', owners=locals())
