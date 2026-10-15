from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.normalization.batch_norm import batch_norm_training as batch_norm_training_kernel
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, channels, spatial) = (32, 64, 4096)
    x = torch.randn((batch, channels, spatial), device='cuda', dtype=torch.float16)
    weight = torch.randn((channels,), device='cuda', dtype=torch.float32)
    bias = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_mean = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_variance = torch.rand((channels,), device='cuda', dtype=torch.float32) + 1.0
    generated_mean = initial_mean.clone()
    generated_variance = initial_variance.clone()
    (artifact, _) = compile_single(context, batch_norm_training_kernel, (x, weight, bias, generated_mean, generated_variance, 1e-05, 0.1))
    operator = torch.compile(artifact.as_torch_op('intent_gpu_benchmark::batch_norm_training'), fullgraph=True)
    generated_state = {}

    def generated_launch():
        generated_state['outputs'] = operator(x, weight, bias, generated_mean, generated_variance, 1e-05, 0.1)
    initial_launch(generated_launch, side='generated')
    generated = PreparedLaunch(launch=generated_launch, outputs=lambda : (*generated_state['outputs'], generated_mean, generated_variance), prepare=lambda : (generated_mean.copy_(initial_mean), generated_variance.copy_(initial_variance)))
    return Side(generated, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001)), cuda_graph=False, note='Intent 通过 mutable custom op 与 torch.compile(fullgraph=True) 调用，计完整执行，含框架产生的状态处理；原输入与容差。', owners=locals())
