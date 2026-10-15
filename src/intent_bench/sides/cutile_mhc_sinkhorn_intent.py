from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.routing.mhc import mhc_sinkhorn
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (tokens, streams) = (8192, 4)
    logits = torch.randn((tokens, streams, streams), device='cuda', dtype=torch.float32)
    generated_values = logits.clone()
    (_, generated_call) = compile_single(context, mhc_sinkhorn, (generated_values,))

    def prepare_generated():
        generated_values.copy_(logits)
    generated = PreparedLaunch(launch=generated_call.launch, outputs=lambda : generated_values, prepare=prepare_generated)
    return Side(generated, Tolerance(atol=0.0002, rtol=0.0001), cuda_graph=False, note='同算法；source 用 exp2(x * log2(e)) 实现 exp', owners=locals())
