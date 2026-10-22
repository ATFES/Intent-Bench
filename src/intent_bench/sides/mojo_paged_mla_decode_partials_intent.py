from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.streaming.mla import paged_mla_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    query = torch.randn((8, 128, 512), dtype=torch.float16)
    query_rope = torch.randn((8, 128, 64), dtype=torch.float16)
    cache = torch.randn((4096, 16, 1, 512), dtype=torch.float16)
    cache_rope = torch.randn((4096, 16, 1, 64), dtype=torch.float16)
    offsets = torch.arange(0, 4097, 512, dtype=torch.int32)
    indices = torch.arange(4096, dtype=torch.int32)
    lengths = torch.full((8,), 8192, dtype=torch.int32)
    flat_arguments = (query, query_rope, cache.reshape(65536, 1, 512), cache_rope.reshape(65536, 1, 64), offsets, indices, lengths, 576 ** (-0.5))
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
