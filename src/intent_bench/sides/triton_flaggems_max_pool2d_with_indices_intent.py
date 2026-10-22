from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.vision.max_pool_with_indices import max_pool2d_with_indices
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8, 32, 128, 128), device='cuda', dtype=torch.float16)
    (_, generated) = compile_single(context, max_pool2d_with_indices, (x,))
    return checked_call(generated, (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=True, owners=locals())
