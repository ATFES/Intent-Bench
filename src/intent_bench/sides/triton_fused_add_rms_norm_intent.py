from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    hidden = 4096
    shape = (8192, hidden)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    residual = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    arguments = (x, residual, weight, 1.0 / hidden, 1e-06, 0.0)
    (_, generated) = compile_single(context, fused_add_rms_norm, arguments)
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
