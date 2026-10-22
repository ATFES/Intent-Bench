from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    q = torch.randn((32, 32, 128), dtype=torch.bfloat16)
    k = torch.randn((32, 8192, 8, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((32,), dtype=torch.bfloat16)
    start = torch.tensor([8191], dtype=torch.int32)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    (generated, source) = ({}, {})

    def reference():
        source['output'] = runtime.attention_sink_decode(q, k, v, sinks, start, 128 ** (-0.5)).reshape(32, 1, 4096)
    return checked_call(PreparedCall(reference, lambda : source['output']), Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='原BF16 sink decode B32/S8192/HQ32/HK8/D128、start8191、split256/P32；完整partial+reduce含workspace分配，生成LSEf32/partial BF16；CPU数学reference f32 softmax，最终BF16，原容差。', owners=locals())
