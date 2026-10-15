from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.convolution.direct import causal_depthwise_conv1d_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, channels, sequence, width) = (4, 4096, 4096, 4)
    x = torch.randn((batch, sequence, channels), device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    weight = torch.randn((channels, width), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn((channels,), device='cuda', dtype=torch.bfloat16)
    (_, generated) = compile_single(context, causal_depthwise_conv1d_bf16, (x, weight, bias), constexprs={'SILU': True})
    return Side(generated, Tolerance(atol=0.05, rtol=0.05), cuda_graph=True, owners=locals())
