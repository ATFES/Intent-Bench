from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.quantization.fp8 import f32_groupwise_fp8_quantize
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), dtype=torch.float32)
    artifact = context.compile(f32_groupwise_fp8_quantize)

    def side():
        scales = torch.zeros((8192, 64), dtype=torch.float32)
        return intent_call(artifact, (x, scales), reset=lambda: scales.zero_(),
                           observe=lambda output: (output, scales))
    return checked_call(side(), (Tolerance(atol=16.0, rtol=0.125), Tolerance(atol=1e-06, rtol=0.0001)), cuda_graph=False, device_type='cpu', note='既有 8192x8192 f32 group128 E4M3FN quantize；absmax 最小值1e-4，保留原输出和scale容差；独立scale存储，PyTorch CPU reference，prepared native调用，单 NUMA 8 核。', owners=locals())
