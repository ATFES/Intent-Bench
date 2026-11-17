from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import reference_launch

def prepare(context):
    (batch, query_heads, kv_heads, dimension) = (16, 32, 8, 128)
    (length, page_size, splits) = (8192, 16, 8)
    pages_per_sequence = length // page_size
    pages = batch * pages_per_sequence
    from intent_bench.trees import to_device
    q, keys, values, offsets, indices, lengths, split_offsets, scale = to_device(context.inputs, 'cuda')
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/paged_decode/paged_gqa_decode_runtime.py', 'intent_bangc_paged_gqa_runtime')
    source = runtime.RUNTIME.load_source(context.project_root / 'corpus/gpu/baselines/triton/vllm/attention/paged_decode/triton_decode_attention.py', 'intent_bangc_paged_gqa_source')
    source_output = torch.empty_like(q)
    source_lse = torch.empty((batch, query_heads), device='cuda', dtype=torch.float32)
    workspace = torch.empty((batch, query_heads, splits, dimension + 1), device='cuda', dtype=torch.float32)
    page_table = indices.view(batch, pages_per_sequence)

    def reference():
        source.decode_attention_fwd(q, keys, values, source_output, source_lse, page_table, lengths, workspace, splits, scale, page_size=page_size)
        return source_output
    return checked_call(reference_launch(reference), Tolerance(atol=0.05, rtol=0.05), device_type='cuda', owners=locals())
