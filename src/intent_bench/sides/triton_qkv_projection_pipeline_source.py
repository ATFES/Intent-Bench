from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    tokens = hidden = projection = 4096
    x = torch.randn((tokens, hidden), device='cuda', dtype=torch.float16)
    weights = tuple((torch.randn((hidden, projection), device='cuda', dtype=torch.float16) for _ in range(3)))
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/xformers/gemm/tiled/tiled_matmul_kernels_runtime.py', 'intent_v2_triton_qkv_projection')
    source_module = runtime.load_source()
    source_outputs = tuple((torch.empty((tokens, projection), device='cuda', dtype=torch.float16) for _ in range(3)))
    a = [[x]]
    b = [[*weights]]
    c = [[*source_outputs]]

    def launch():
        source_module._launch_triton_matmul(a, b, c, [tokens], [projection] * 3, [hidden])
    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda : source_outputs)
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
