from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.routing.moe_align import moe_count_routes
from kernels.routing.moe_align import moe_mark_expert_blocks
from kernels.routing.moe_align import moe_prefix_routes
from kernels.routing.moe_align import moe_scatter_routes
from programs.composition import MoEAlignment
from intent_bench.corpus._common.measurement import compile_kernel, prepare_program
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance
BLOCK_SIZE = 128
EXPERTS = 64

def prepare(context):
    (tokens, topk) = (4096, 2)
    source_ids = torch.arange(tokens * topk, device='cuda', dtype=torch.int32).remainder(EXPERTS).reshape(tokens, topk).contiguous()
    program = MoEAlignment(compile_kernel(context, moe_count_routes), compile_kernel(context, moe_prefix_routes), compile_kernel(context, moe_scatter_routes), compile_kernel(context, moe_mark_expert_blocks))
    call = prepare_program(program, (source_ids,))
    generated = PreparedLaunch(call.launch, lambda : _canonical_alignment(*call.outputs()))
    return Side(generated, (Tolerance(atol=0.0), Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=False, note='同算法计数/前缀和/散射/标记；workspace初始化与辅助输出计入调用', owners=locals())

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
