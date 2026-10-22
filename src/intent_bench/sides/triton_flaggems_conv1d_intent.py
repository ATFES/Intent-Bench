from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.convolution.direct import conv1d_same
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, length, width) = (64, 16384, 5)
    x = torch.randn((batch, length), device='cuda', dtype=torch.float16)
    weight = torch.randn((width,), device='cuda', dtype=torch.float16)
    (_, generated) = compile_single(context, conv1d_same, (x, weight))
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
