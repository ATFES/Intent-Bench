from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.streaming.mla import splitk_mla_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_reduce_f16
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    arguments = _absorbed_decode_inputs()
    partials = context.compile(splitk_mla_decode_partials, constexprs={'SPLITS': 16, 'SPLIT_SIZE': 512})
    reduction = context.compile(splitk_attention_reduce_f16)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(*arguments)
        generated['output'] = reduction.run(partial, lse)
    sequence = IntentSequence()
    lse, partial = sequence.add(partials, *arguments)
    output = sequence.add(reduction, partial, lse)
    call = sequence.prepared(lambda: output, host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, Tolerance(atol=0.1, rtol=0.05), device_type='cpu', note='Prepared B8/H64/S8192/C512/R64 f16 split512/P16 MLA partial+reduce; original partial f16/LSE f32 precision.', owners=locals())

def _absorbed_decode_inputs():
    query = torch.randn((8, 64, 512), dtype=torch.float16)
    query_rope = torch.randn((8, 64, 64), dtype=torch.float16)
    cache = torch.randn((8, 8192, 512), dtype=torch.float16)
    cache_rope = torch.randn((8, 8192, 64), dtype=torch.float16)
    return (query, query_rope, cache, cache_rope, 576 ** (-0.5))
