from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.streaming.paged_attention import splitk_paged_gqa_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_weighted_sum_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    _inlined_paged_gqa_split_q = torch.randn((16, 32, 128), dtype=torch.float16)
    _inlined_paged_gqa_split_key = torch.randn((8192, 16, 8, 128), dtype=torch.float16)
    _inlined_paged_gqa_split_value = torch.randn_like(_inlined_paged_gqa_split_key)
    _inlined_paged_gqa_split_indices = torch.arange(8192, dtype=torch.int32)
    _inlined_paged_gqa_split_offsets = torch.arange(0, 8193, 512, dtype=torch.int32)
    _inlined_paged_gqa_split_splits = torch.arange(0, 8193, 64, dtype=torch.int32)
    _inlined_paged_gqa_split_lengths = torch.full((16,), 8192, dtype=torch.int32)
    _inlined_paged_gqa_split_arguments = (_inlined_paged_gqa_split_q, _inlined_paged_gqa_split_key, _inlined_paged_gqa_split_value, _inlined_paged_gqa_split_offsets, _inlined_paged_gqa_split_indices, _inlined_paged_gqa_split_lengths, _inlined_paged_gqa_split_splits, 128 ** (-0.5))
    _inlined_paged_gqa_split_partials = context.compile(splitk_paged_gqa_decode_partials, constexprs={'PAGE_SIZE': 16, 'HEAD_GROUP': 4, 'SPLITS': 8})
    _inlined_paged_gqa_split_reduction = context.compile(splitk_attention_weighted_sum_reduce)
    _inlined_paged_gqa_split_generated = {}

    def _inlined_paged_gqa_split_launch():
        (_inlined_paged_gqa_split_lse, _inlined_paged_gqa_split_partial) = _inlined_paged_gqa_split_partials.run(*_inlined_paged_gqa_split_arguments)
        _inlined_paged_gqa_split_generated['output'] = _inlined_paged_gqa_split_reduction.run(_inlined_paged_gqa_split_partial, _inlined_paged_gqa_split_lse)
    sequence = IntentSequence()
    lse, partial = sequence.add(_inlined_paged_gqa_split_partials, *_inlined_paged_gqa_split_arguments)
    output = sequence.add(_inlined_paged_gqa_split_reduction, partial, lse)
    call = sequence.prepared(lambda: output, host_run=_inlined_paged_gqa_split_launch,
                             host_result=lambda: _inlined_paged_gqa_split_generated['output'])
    return checked_call(call, Tolerance(atol=0.1, rtol=0.05), device_type='cpu', note='Prepared split-K paged GQA; original BF16 partial+weighted reduction and output contract.', owners=locals())
