from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.contraction.qkv import fused_qkv_projection
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    tokens = hidden = projection = 4096
    x = torch.randn((tokens, hidden), device='cuda', dtype=torch.float16)
    weights = tuple((torch.randn((hidden, projection), device='cuda', dtype=torch.float16) for _ in range(3)))
    packed_weights = torch.stack(weights)
    (_, generated_base) = compile_single(context, fused_qkv_projection, (x, packed_weights))
    generated = PreparedCall(launch=generated_base.launch, result=lambda : tuple((generated_base.result()[index] for index in range(3))))
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())
