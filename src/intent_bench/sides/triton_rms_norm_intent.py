from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.normalization.rms_norm import rms_norm_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    hidden = 4096
    x = torch.randn((8192, hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    arguments = (x, weight, 1.0 / hidden, 1e-06)
    (_, generated) = compile_single(context, rms_norm_bf16, arguments)
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
