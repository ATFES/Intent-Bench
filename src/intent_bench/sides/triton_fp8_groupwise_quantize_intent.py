from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.quantization.fp8 import bf16_groupwise_fp8_quantize
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.bfloat16)
    scales = torch.empty((8192, 32), device='cuda', dtype=torch.float32)
    (_, generated_base) = compile_single(context, bf16_groupwise_fp8_quantize, (x, scales))
    generated = PreparedCall(launch=generated_base.launch, result=lambda : (generated_base.result(), scales))
    return checked_call(generated, (Tolerance(atol=16.0, rtol=0.125), Tolerance(atol=1e-06, rtol=0.0001)), cuda_graph=True, owners=locals())
