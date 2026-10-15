from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.normalization.softmax import chunked_softmax_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8192, 32768), device='cuda', dtype=torch.bfloat16)
    (_, generated) = compile_single(context, chunked_softmax_bf16, (x,))
    return Side(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
