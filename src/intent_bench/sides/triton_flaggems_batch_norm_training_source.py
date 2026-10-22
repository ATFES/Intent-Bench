from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.references.torch_ops import batch_norm_training
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, channels, spatial) = (32, 64, 4096)
    x = torch.randn((batch, channels, spatial), device='cuda', dtype=torch.float16)
    weight = torch.randn((channels,), device='cuda', dtype=torch.float32)
    bias = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_mean = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_variance = torch.rand((channels,), device='cuda', dtype=torch.float32) + 1.0
    source_mean = initial_mean.clone()
    source_variance = initial_variance.clone()
    state: dict[str, object] = {}

    def source_launch():
        state['outputs'] = batch_norm_training(*(x, weight, bias, source_mean, source_variance, 1e-05, 0.1))
    source_launch()
    source = PreparedCall(launch=source_launch, result=lambda : (*state['outputs'], source_mean, source_variance), reset=lambda : (source_mean.copy_(initial_mean), source_variance.copy_(initial_variance)))
    return checked_call(source, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001)), cuda_graph=False, note='Torch reference returns output, mean, inverse std and updated running statistics; state restoration is outside timing.', owners=locals())
