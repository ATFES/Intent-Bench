from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from triton.runtime.errors import OutOfResources
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import ComparisonUnavailable
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, heads, sequence, dimension) = (2, 16, 2048, 128)
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device='cuda', dtype=torch.float16) * 0.5
    k = torch.randn_like(q) * 0.5
    v = torch.randn_like(q) * 0.5
    output = torch.empty_like(q)
    lse = torch.empty((batch, heads, sequence), device='cuda', dtype=torch.float32)
    runtime_helper = load_module(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/support/runtime.py', 'intent_v2_triton_flash_backward_runtime')
    source_module = runtime_helper.load_source(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/attention/flash_backward/flash_backward.py', 'intent_v2_triton_flash_backward_source')
    _prepare_source_launch(lambda : source_module.flash(q, k, v, output, lse))
    grad_output = torch.randn_like(output) * 0.05
    source = functional_launch(lambda : source_module.flash_bwd(q, k, v, output, lse, grad_output))
    return checked_call(source, (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=False, note='source 修正了中间量写死 bf16 与原 fp16 输入冲突的 dtype 转换；保持原输入与容差。', owners=locals())

def _prepare_source_launch(launch):
    try:
        launch()
    except OutOfResources as error:
        raise ComparisonUnavailable('source_device_resource_gap', f"vendored Triton source exceeds this device's resources: {error}") from error
