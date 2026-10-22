from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.activation.pointwise import geglu_tanh
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    x = torch.randn((4096, 2 * 14336), dtype=torch.float16)
    artifact = context.compile(geglu_tanh)

    def side(function):
        output = torch.empty((4096, 14336), dtype=torch.float16)
        return PreparedCall(lambda : function(x, output), lambda : output)
    return checked_call(side(artifact.run), Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, device_type='cpu', note='既有4096x28672 f16 GEGLU，独立InOut输出，完整host调用；单NUMA8核，PyTorch CPU reference。', owners=locals())
