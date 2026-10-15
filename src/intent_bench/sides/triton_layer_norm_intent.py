from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.normalization.layer_norm import layer_norm_f16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    hidden = 4096
    x = torch.randn((8192, hidden), device='cuda', dtype=torch.float16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.float16)
    bias = torch.randn((hidden,), device='cuda', dtype=torch.float16)
    arguments = (x, weight, bias, 1.0 / hidden, 1e-05)
    (_, generated) = compile_single(context, layer_norm_f16, arguments)
    return Side(generated, Tolerance(atol=0.01), cuda_graph=True, owners=locals())
