from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.streaming.attention import grouped_flash_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    q = torch.randn((8, 32, 1, 128), dtype=torch.bfloat16)
    k = torch.randn((8, 8, 8192, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    partials = context.compile(grouped_flash_decode_partials, constexprs={'HEAD_GROUP': 4, 'P': 32, 'SPLIT_SIZE': 256})
    reduction = context.compile(splitk_attention_reduce)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(q, k, v, 128 ** (-0.5))
        generated['output'] = reduction.run(partial, lse).unsqueeze(2)
    sequence = IntentSequence()
    lse, partial = sequence.add(partials, q, k, v, 128 ** (-0.5))
    output = sequence.add(reduction, partial, lse).unsqueeze(2)
    call = sequence.prepared(lambda: output, host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, Tolerance(atol=0.1, rtol=0.05), device_type='cpu', note='Prepared grouped decode B8/HQ32/HK8/K8192/D128 BF16, split256/P32. Original partial/LSE precision and output ABI.', owners=locals())
