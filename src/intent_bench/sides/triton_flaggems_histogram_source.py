from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    samples = torch.randint(0, 256, (8 * 1024 * 1024,), device='cuda', dtype=torch.int32).to(torch.float32)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/flag-gems/statistics/histogram/histc_runtime.py', 'intent_v2_triton_flaggems_histogram')
    source_histogram = torch.zeros((256,), device=samples.device, dtype=torch.int32)

    def source_launch():
        runtime.MODULE.histc_kernel_simple[runtime.MODULE.triton.cdiv(samples.numel(), 1024),](samples, source_histogram, samples.numel(), 256, 0.0, 256.0, BLOCK_SIZE=1024)
    source_launch()
    source = PreparedLaunch(launch=source_launch, outputs=lambda : source_histogram, prepare=source_histogram.zero_)
    return Side(source, Tolerance(atol=0.0), cuda_graph=False, owners=locals())
