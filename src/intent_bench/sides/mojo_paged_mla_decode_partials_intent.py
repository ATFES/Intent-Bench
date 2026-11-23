from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.streaming.mla import paged_mla_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    from .attention_inputs import latent
    arguments = context.inputs if context.inputs is not None else latent().arguments
    query,query_rope,cache,cache_rope,offsets,indices,lengths,scale = arguments
    flat_arguments = (query,query_rope,cache.flatten(0,1),cache_rope.flatten(0,1),offsets,indices,lengths,scale)
    partials = context.compile(paged_mla_decode_partials, constexprs={'PAGE_SIZE': 16, 'HEAD_GROUP': 128, 'SPLITS': 8})
    reduction = context.compile(splitk_attention_f32_to_f16_reduce)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(*flat_arguments)
        generated['output'] = reduction.run(partial, lse)
    sequence = IntentSequence()
    lse, partial = sequence.add(partials, *flat_arguments)
    output = sequence.add(reduction, partial, lse)
    call = sequence.prepared(lambda: output, host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, Tolerance(atol=0.05, rtol=0.05), device_type='cpu', note='Prepared B8/HQ128/HK1/K8192/C512/R64/page16/split8 MLA; original f32 partial/LSE and f16 final output.', owners=locals())
