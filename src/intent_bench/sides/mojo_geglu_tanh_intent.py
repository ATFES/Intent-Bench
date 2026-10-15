from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.activation.pointwise import geglu_tanh
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    x = torch.randn((4096, 2 * 14336), dtype=torch.float16)
    artifact = intent.compile(geglu_tanh, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def side(function):
        output = torch.empty((4096, 14336), dtype=torch.float16)
        return PreparedLaunch(lambda : function(x, output), lambda : output)
    return Side(side(artifact.run), Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, device_type='cpu', note='既有4096x28672 f16 GEGLU，独立InOut输出，完整host调用；单NUMA8核，PyTorch CPU reference。', owners=locals())
