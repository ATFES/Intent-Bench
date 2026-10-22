from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    x = torch.randn((4096, 2 * 14336), dtype=torch.float16)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        output = torch.empty((4096, 14336), dtype=torch.float16)
        return PreparedCall(lambda : function(x, output), lambda : output)
    return checked_call(side(runtime.geglu_tanh), Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, device_type='cpu', note='既有4096x28672 f16 GEGLU，独立InOut输出，完整host调用；单NUMA8核，PyTorch CPU reference。', owners=locals())
