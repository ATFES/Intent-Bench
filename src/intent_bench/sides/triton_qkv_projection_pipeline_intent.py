from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.contraction.qkv import fused_qkv_projection
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    tokens = hidden = projection = 4096
    x = torch.randn((tokens, hidden), device='cuda', dtype=torch.float16)
    weights = tuple((torch.randn((hidden, projection), device='cuda', dtype=torch.float16) for _ in range(3)))
    packed_weights = torch.stack(weights)
    (_, generated_base) = compile_single(context, fused_qkv_projection, (x, packed_weights))
    generated = PreparedLaunch(launch=generated_base.launch, outputs=lambda : tuple((generated_base.outputs()[index] for index in range(3))))
    return Side(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())
