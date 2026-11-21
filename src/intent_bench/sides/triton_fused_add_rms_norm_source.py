from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    arguments = context.inputs
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/liger-kernel/normalization/fused_add_rms_norm/fused_add_rms_norm_runtime.py', 'intent_v2_triton_fused_add_rms_norm')
    source = functional_launch(lambda : runtime.upstream(arguments))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
