from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import reference_launch

def prepare(context):
    from intent_bench.trees import to_device
    values, offsets = to_device(context.inputs, 'cuda')
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/tritonbench/ragged/jagged_mean/jagged_mean_runtime.py', 'intent_bangc_jagged_mean')
    return checked_call(reference_launch(lambda : runtime.upstream((values, offsets))), Tolerance(atol=2e-05, rtol=1e-05), device_type='cuda', owners=locals())
