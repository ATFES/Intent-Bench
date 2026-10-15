from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.routing.mhc import mhc_apply_residual
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (tokens, hidden, streams) = (2048, 4096, 4)
    residual = torch.randn((tokens, streams, hidden), device='cuda', dtype=torch.bfloat16)
    layer_output = torch.randn((tokens, hidden), device='cuda', dtype=torch.bfloat16)
    post_bf16 = torch.randn((tokens, streams), device='cuda', dtype=torch.bfloat16)
    residual_bf16 = torch.randn((tokens, streams, streams), device='cuda', dtype=torch.bfloat16)
    post_mix = post_bf16.float()
    residual_mix = residual_bf16.float()
    (_, generated) = compile_single(context, mhc_apply_residual, (residual, layer_output, post_mix, residual_mix))
    return Side(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=True, note='同算法；source 保留 bf16 中间累加', owners=locals())
