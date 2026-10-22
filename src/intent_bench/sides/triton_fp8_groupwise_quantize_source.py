from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.bfloat16)
    runtime_helper = load_module(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/support/runtime.py', 'intent_v2_triton_meta_runtime_quantize')
    source_module = runtime_helper.load_source(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/quantization/fp8_groupwise/float8_groupwise_quant.py', 'intent_v2_triton_fp8_groupwise_quantize')
    source = functional_launch(lambda : source_module.float8_groupwise_quantize(x, 128))
    return checked_call(source, (Tolerance(atol=16.0, rtol=0.125), Tolerance(atol=1e-06, rtol=0.0001)), cuda_graph=True, owners=locals())
