from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.quantization.fp8 import bf16_groupwise_fp8_quantize
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
from intent_bench.observed import with_writes

def prepare(context):
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    artifact = intent.compile(bf16_groupwise_fp8_quantize, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def side(function):
        scales = torch.zeros((8192, 32), dtype=torch.float32)
        state = {}

        def launch():
            state['outputs'] = function(x, scales)
        return PreparedLaunch(launch, lambda : state['outputs'], prepare=lambda : scales.zero_())
    return Side(side(with_writes(artifact)), (Tolerance(atol=16.0, rtol=0.125), Tolerance(atol=1e-06, rtol=0.0001)), cuda_graph=False, device_type='cpu', note='既有8192x4096 bf16、group128、E4M3FN量化；原输出及scale容差；独立scale存储，单NUMA8核，PyTorch CPU reference，完整host调用。', owners=locals())
