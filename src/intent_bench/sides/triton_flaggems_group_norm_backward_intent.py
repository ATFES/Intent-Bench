from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.backward.group_norm import group_norm_backward_dx
from intent.examples.kernels.backward.group_norm import group_norm_backward_weight_bias
from intent.examples.programs.composition import GroupNormBackward
from intent_bench.corpus._common.measurement import compile_kernel, prepare_program
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, channels, spatial, groups) = (32, 256, 1024, 32)
    x = torch.randn((batch, channels, spatial), device='cuda', dtype=torch.float16)
    grad_y = torch.randn_like(x)
    weight = torch.randn((channels,), device='cuda', dtype=torch.float16)
    mean = torch.randn((batch, groups), device='cuda', dtype=torch.float16)
    rstd = torch.rand((batch, groups), device='cuda', dtype=torch.float16) + 0.5
    program = GroupNormBackward(compile_kernel(context, group_norm_backward_dx), compile_kernel(context, group_norm_backward_weight_bias))
    generated = prepare_program(program, (x, grad_y, weight, mean, rstd, 1.0 / (channels // groups * spatial)))
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())
