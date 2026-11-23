from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    from .attention_inputs import latent
    arguments = context.inputs if context.inputs is not None else latent().arguments
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    (generated, source) = ({}, {})

    def reference():
        source['output'] = runtime.paged_mla_decode(*arguments)
    return checked_call(PreparedCall(reference, lambda : source['output']), Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, device_type='cpu', note='原B8/HQ128/HK1/K8192/C512/R64/page16/split8 f16 MLA完整partial+reduce，含workspace分配；生成flattened page cache与f32 partial/LSE，CPU完整page-table attention数学参考，最终f16，原容差。', owners=locals())
