from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.quantization.fp8 import bf16_groupwise_fp8_quantize
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), dtype=torch.bfloat16)
    artifact = context.compile(bf16_groupwise_fp8_quantize)

    def side():
        scales = torch.zeros((8192, 32), dtype=torch.float32)
        return intent_call(artifact, (x, scales), reset=lambda: scales.zero_(), observe=lambda output: (output, scales))
    return checked_call(side(), (Tolerance(atol=16.0, rtol=0.125), Tolerance(atol=1e-06, rtol=0.0001)), cuda_graph=False, device_type='cpu', note='既有8192x4096 bf16、group128、E4M3FN量化；原输出及scale容差；独立scale存储，单NUMA8核，PyTorch CPU reference，prepared native调用。', owners=locals())
