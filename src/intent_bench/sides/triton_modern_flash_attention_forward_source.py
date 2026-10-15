from __future__ import annotations
from intent_bench.sides import Side
import torch
from triton.runtime.errors import OutOfResources
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import ComparisonUnavailable
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, heads, sequence, dimension) = (2, 16, 2048, 128)
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/support/runtime.py', 'intent_v2_triton_modern_flash_runtime')
    source_module = runtime.load_source(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/attention/flash_backward/flash_backward.py', 'intent_v2_triton_modern_flash_source')
    source_output = torch.empty_like(q)
    source_lse = torch.empty((batch, heads, sequence), device='cuda', dtype=torch.float32)

    def source_launch():
        source_module.flash(q, k, v, source_output, source_lse)
    _prepare_source_launch(source_launch)
    source = PreparedLaunch(source_launch, lambda : source_output)
    return Side(source, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, note='source launch uses num_stages=1 and num_warps=4.', owners=locals())

def _prepare_source_launch(launch):
    try:
        launch()
    except OutOfResources as error:
        raise ComparisonUnavailable('source_device_resource_gap', f"vendored Triton source exceeds this device's resources: {error}") from error
