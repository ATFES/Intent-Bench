from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.backward.group_norm_silu import group_norm_silu_backward
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    x = torch.randn((32, 256, 1024), dtype=torch.bfloat16)
    upstream = torch.randn_like(x)
    weight = torch.randn((256,), dtype=torch.float32)
    bias = torch.randn_like(weight)
    grouped = x.float().reshape(32, 32, 8, 1024)
    mean = grouped.mean(dim=(2, 3))
    rstd = torch.rsqrt(grouped.var(dim=(2, 3), unbiased=False) + 1e-05)
    artifact = context.compile(group_norm_silu_backward)

    def side(function):
        (dweight, dbias) = (torch.zeros_like(weight), torch.zeros_like(bias))
        state = {}

        def launch():
            state['output'] = function(x, upstream, weight, bias, mean, rstd, dweight, dbias, 1.0 / 8192)

        def prepare():
            dweight.zero_()
            dbias.zero_()
        return PreparedCall(launch, lambda : (state['output'], dweight, dbias), reset=prepare)
    return checked_call(side(artifact.run), (Tolerance(atol=0.04), Tolerance(atol=0.03), Tolerance(atol=0.03)), cuda_graph=False, device_type='cpu', note='既有B32-C256-S1024-G32 bf16 GroupNorm-SiLU backward，dx及scatter-reduce参数梯度；单NUMA8核，PyTorch CPU reference，完整host调用；每次清零梯度且清零不计时。', owners=locals())
