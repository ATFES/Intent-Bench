from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.streaming.attention_specialized import attention_sink_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    q = torch.randn((32, 32, 128), dtype=torch.bfloat16)
    k = torch.randn((32, 8192, 8, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((32,), dtype=torch.bfloat16)
    start = torch.tensor([8191], dtype=torch.int32)
    partials = context.compile(attention_sink_decode_partials, constexprs={'HEAD_GROUP': 4, 'WINDOW': 0, 'P': 32})
    reduction = context.compile(splitk_attention_reduce)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(q, k, v, sinks, start, 128 ** (-0.5))
        generated['output'] = reduction.run(partial, lse).reshape(32, 1, 4096)
    sequence = IntentSequence()
    lse, partial = sequence.add(partials, q, k, v, sinks, start, 128 ** (-0.5))
    output = sequence.add(reduction, partial, lse).reshape(32, 1, 4096)
    call = sequence.prepared(lambda: output, host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, Tolerance(atol=0.1, rtol=0.05), device_type='cpu', note='Original BF16 sink decode B32/S8192/HQ32/HK8/D128, start8191, split256/P32. Prepared partial+reduce; allocation outside operator timing.', owners=locals())
