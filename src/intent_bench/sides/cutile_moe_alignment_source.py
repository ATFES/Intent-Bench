from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import runtime_module
BLOCK_SIZE = 128
EXPERTS = 64

def prepare(context):
    (tokens, topk) = (4096, 2)
    source_ids = torch.arange(tokens * topk, device='cuda', dtype=torch.int32).remainder(EXPERTS).reshape(tokens, topk).contiguous()
    runtime = _runtime(context)
    source_state: dict[str, tuple[torch.Tensor, ...]] = {}

    def source_launch():
        source_state['outputs'] = runtime.align.moe_align_block_size(source_ids, BLOCK_SIZE, EXPERTS)

    def source_outputs():
        (sorted_ids, expert_ids, total, _, _) = source_state['outputs']
        return _canonical_alignment(sorted_ids, expert_ids, total)
    initial_launch(source_launch, side='source')
    source = PreparedCall(source_launch, source_outputs)
    return checked_call(source, (Tolerance(atol=0.0), Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=False, note='同算法计数/前缀和/散射/标记；workspace初始化与辅助输出计入调用', owners=locals())

def _runtime(context):
    return runtime_module(context, 'corpus/gpu/baselines/cutile/tilegym/moe/fused/moe_runtime.py', 'intent_v2_cutile_moe_runtime')

def _canonical_alignment(sorted_ids, expert_ids, total):
    total_value = int(total[0].item())
    blocks = total_value // BLOCK_SIZE
    canonical_ids = sorted_ids[:total_value].clone()
    active_experts = expert_ids[:blocks]
    for expert in range(EXPERTS):
        expert_blocks = torch.nonzero(active_experts == expert).flatten()
        if expert_blocks.numel() == 0:
            continue
        begin = int(expert_blocks[0].item()) * BLOCK_SIZE
        end = (int(expert_blocks[-1].item()) + 1) * BLOCK_SIZE
        canonical_ids[begin:end] = torch.sort(canonical_ids[begin:end]).values
    return (canonical_ids, active_experts, total)
