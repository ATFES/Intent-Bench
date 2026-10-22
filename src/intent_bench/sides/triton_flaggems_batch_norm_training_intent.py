from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.normalization.batch_norm import batch_norm_training as batch_norm_training_kernel
from intent_bench.corpus._common.measurement import compile_kernel
from intent_bench.implementations.prepared import intent_call
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, channels, spatial) = (32, 64, 4096)
    x = torch.randn((batch, channels, spatial), device='cuda', dtype=torch.float16)
    weight = torch.randn((channels,), device='cuda', dtype=torch.float32)
    bias = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_mean = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_variance = torch.rand((channels,), device='cuda', dtype=torch.float32) + 1.0
    generated_mean = initial_mean.clone()
    generated_variance = initial_variance.clone()
    artifact = compile_kernel(context, batch_norm_training_kernel)
    arguments = (x, weight, bias, generated_mean, generated_variance, 1e-05, 0.1)
    def host_factory():
        operator = torch.compile(artifact.as_torch_op('intent_gpu_benchmark::batch_norm_training'), fullgraph=True)
        return lambda: operator(*arguments)
    generated = intent_call(artifact, arguments, device=str(x.device),
        reset=lambda: (generated_mean.copy_(initial_mean), generated_variance.copy_(initial_variance)),
        observe=lambda output: (*output, generated_mean, generated_variance), host_factory=host_factory)
    return checked_call(generated, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001)), cuda_graph=True, note='Prepared BatchNorm execution including running-state updates; torch.compile integration is opt-in via host_call.', owners=locals())
