from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from triton.runtime.errors import OutOfResources
from intent.examples.kernels.backward.attention import attention_backward_delta
from intent.examples.kernels.backward.attention import attention_backward_dkdv
from intent.examples.kernels.backward.attention import attention_backward_dq
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import ComparisonUnavailable
from intent_bench.model import PreparedCall
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
    scale = dimension ** (-0.5)
    (_, delta) = compile_single(context, attention_backward_delta, (output, grad_output))
    delta_value = delta.result()
    generated_lse = lse / math.log2(math.e)
    common = (q, k, v, grad_output, generated_lse, delta_value, scale)
    constexprs = {'HEAD_GROUP': 1, 'CAUSAL': True}
    (_, dkdv) = compile_single(context, attention_backward_dkdv, common, constexprs=constexprs)
    (_, dq) = compile_single(context, attention_backward_dq, common, constexprs=constexprs)

    def generated_launch():
        delta.launch()
        dkdv.launch()
        dq.launch()
    generated = PreparedCall(launch=generated_launch, result=lambda : (dq.result(), *dkdv.result()))
    return checked_call(generated, (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=False, note='source 修正了中间量写死 bf16 与原 fp16 输入冲突的 dtype 转换；保持原输入与容差。', owners=locals())

def _prepare_source_launch(launch):
    try:
        launch()
    except OutOfResources as error:
        raise ComparisonUnavailable('source_device_resource_gap', f"vendored Triton source exceeds this device's resources: {error}") from error
