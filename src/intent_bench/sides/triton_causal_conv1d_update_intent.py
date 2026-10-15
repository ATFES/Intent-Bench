from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.convolution.direct import causal_depthwise_conv1d_update_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, hidden, width) = (32, 4096, 4)
    x = torch.randn((batch, hidden), device='cuda', dtype=torch.bfloat16)
    initial = torch.randn((batch, hidden, width), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden, width), device='cuda', dtype=torch.float32)
    bias = torch.randn((hidden,), device='cuda', dtype=torch.float32)
    generated_state = initial.clone()
    (_, generated_base) = compile_single(context, causal_depthwise_conv1d_update_bf16, (x, generated_state, weight, bias), constexprs={'SILU': True})
    generated = PreparedLaunch(launch=generated_base.launch, outputs=lambda : (generated_base.outputs(), generated_state), prepare=lambda : generated_state.copy_(initial))
    return Side(generated, (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.0)), cuda_graph=False, owners=locals())
