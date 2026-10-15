from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, channels, spatial) = (32, 64, 4096)
    x = torch.randn((batch, channels, spatial), device='cuda', dtype=torch.float16)
    weight = torch.randn((channels,), device='cuda', dtype=torch.float32)
    bias = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_mean = torch.randn((channels,), device='cuda', dtype=torch.float32)
    initial_variance = torch.rand((channels,), device='cuda', dtype=torch.float32) + 1.0
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/flag-gems/normalization/batch_norm/batch_norm_runtime.py', 'intent_v2_triton_flaggems_batch_norm')
    source_mean = initial_mean.clone()
    source_variance = initial_variance.clone()
    state: dict[str, object] = {}

    def source_launch():
        state['outputs'] = runtime.upstream((x, weight, bias, source_mean, source_variance, 1e-05, 0.1))
    source_launch()
    source = PreparedLaunch(launch=source_launch, outputs=lambda : (*state['outputs'], source_mean, source_variance), prepare=lambda : (source_mean.copy_(initial_mean), source_variance.copy_(initial_variance)))
    return Side(source, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001), Tolerance(atol=0.0001, rtol=0.0001)), cuda_graph=False, note='Intent 通过 mutable custom op 与 torch.compile(fullgraph=True) 调用，计完整执行，含框架产生的状态处理；原输入与容差。', owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
