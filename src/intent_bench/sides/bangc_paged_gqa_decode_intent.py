from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.paged_attention import paged_gqa_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.bangc_sequence import LocalSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    (batch, query_heads, kv_heads, dimension) = (16, 32, 8, 128)
    (length, page_size, splits) = (8192, 16, 8)
    pages_per_sequence = length // page_size
    pages = batch * pages_per_sequence
    q = torch.randn((batch, query_heads, dimension), device='cpu', dtype=torch.float16)
    keys = torch.randn((pages, page_size, kv_heads, dimension), device='cpu', dtype=torch.float16)
    values = torch.randn_like(keys)
    indices = torch.arange(pages, device='cpu', dtype=torch.int32)
    offsets = torch.arange(0, pages + 1, pages_per_sequence, device='cpu', dtype=torch.int32)
    split_offsets = torch.arange(0, pages + 1, pages_per_sequence // splits, device='cpu', dtype=torch.int32)
    lengths = torch.full((batch,), length, device='cpu', dtype=torch.int32)
    scale = dimension ** (-0.5)
    sequence = LocalSequence(context)
    partials = sequence.add(paged_gqa_decode_partials, {'q': q, 'key_cache': keys, 'value_cache': values, 'page_offsets': offsets, 'page_indices': indices, 'sequence_lengths': lengths, 'split_offsets': split_offsets, 'scale': scale}, constexprs={'PAGE_SIZE': page_size, 'HEAD_GROUP': query_heads // kv_heads, 'SPLITS': splits})
    output = sequence.add(splitk_attention_f32_to_f16_reduce, {'partial': partials['partial_output'], 'partial_lse': partials['partial_lse']})['output']
    return checked_call(prepare_sequence(sequence, output), Tolerance(atol=0.05, rtol=0.05), device_type='mlu', owners=locals())
