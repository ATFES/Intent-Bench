from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.normalization.softmax import stable_softmax_f16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), device='cuda', dtype=torch.float16)
    (_, generated) = compile_single(context, stable_softmax_f16, (x,))
    return checked_call(generated, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, owners=locals())
