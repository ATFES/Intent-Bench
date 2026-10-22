from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.ordered_prefix import row_cumsum_f32
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), device='cuda', dtype=torch.float32)
    (_, generated) = compile_single(context, row_cumsum_f32, (x,))
    return checked_call(generated, Tolerance(atol=0.002, rtol=1e-05), cuda_graph=True, owners=locals())
