from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
BLOCK_SIZE = 128
EXPERTS = 64

def prepare(context):
    (tokens, topk) = (4096, 2)
    source_ids = torch.arange(tokens * topk, dtype=torch.int32).remainder(EXPERTS).reshape(tokens, topk).contiguous()
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_moe_alignment')
    source_state: dict[str, tuple[torch.Tensor, ...]] = {}

    def source_launch():
        source_state['output'] = runtime.moe_align_block_size(source_ids, BLOCK_SIZE, EXPERTS)
    return checked_call(PreparedCall(source_launch, lambda : _canonical_alignment(*source_state['output'])), tuple((Tolerance(atol=0.0) for _ in range(3))), cuda_graph=False, device_type='cpu', note='既有MoE alignment；T4096、top-k2、E64、block128；完整计数/前缀/散射/标记调用，辅助workspace初始化计入调用，输出为canonical三元组。', owners=locals())

def _canonical_alignment(sorted_ids, expert_ids, total):
    total_value = int(total.reshape(-1)[0].item())
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
