from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (destination_rows, routed_rows, hidden) = (65536, 32768, 4096)
    initial = torch.randn((destination_rows, 1, hidden), device='cuda', dtype=torch.float16)
    routed = torch.randn((routed_rows, 1, hidden), device='cuda', dtype=torch.float16)
    indices = torch.arange(0, routed_rows * 2, 2, device='cuda', dtype=torch.int64)
    scaling = torch.randn((hidden,), device='cuda', dtype=torch.float16)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/xformers/indexing/scaled_index_add/k_scaled_index_add_runtime.py', 'intent_v2_triton_scaled_index_add')
    source_module = runtime.load_source()
    source_destination = initial.clone()

    def source_launch():
        source_module.scaled_index_add_fwd(source_destination, indices, routed, scaling, 1.0)
    source_launch()
    source = PreparedLaunch(launch=source_launch, outputs=lambda : source_destination, prepare=lambda : source_destination.copy_(initial))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
