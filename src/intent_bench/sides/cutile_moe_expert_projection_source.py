from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import runtime_module
BLOCK_SIZE = 128

def prepare(context):
    (tokens, hidden, intermediate, experts, topk) = (4096, 4096, 14336, 8, 2)
    hidden_states = torch.randn((tokens, hidden), device='cuda', dtype=torch.bfloat16)
    weights = torch.randn((experts, intermediate, hidden), device='cuda', dtype=torch.bfloat16)
    topk_ids = torch.arange(tokens * topk, device='cuda', dtype=torch.long).remainder(experts).reshape(tokens, topk).contiguous()
    topk_weights = torch.full((tokens, topk), 1.0 / topk, device='cuda', dtype=torch.bfloat16)
    runtime = _runtime(context)
    (sorted_ids, expert_ids, padded_tokens, _, _) = initial_launch(lambda : runtime.align.moe_align_block_size(topk_ids, BLOCK_SIZE, experts), side='source')
    source_output = torch.empty((tokens, topk, intermediate), device='cuda', dtype=torch.bfloat16)
    config = {'TILE_SIZE_M': 128, 'TILE_SIZE_N': 128, 'TILE_SIZE_K': 64, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4}

    def source_launch():
        runtime.source.invoke_fused_moe_kernel(hidden_states, weights, source_output, None, None, topk_weights, topk_ids, sorted_ids, expert_ids, padded_tokens, False, topk, config, torch.bfloat16, False)
    initial_launch(source_launch, side='source')
    source = PreparedCall(source_launch, lambda : source_output)
    return checked_call(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, note='同算法；source 使用预计算 block ownership', owners=locals())

def _runtime(context):
    return runtime_module(context, 'corpus/gpu/baselines/cutile/tilegym/moe/fused/moe_runtime.py', 'intent_v2_cutile_moe_runtime')
