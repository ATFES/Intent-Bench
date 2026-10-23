from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_sequence
import intent
import torch
from intent.examples.programs.composition import MoEAlignment
from intent.examples.kernels.routing.moe_align import moe_count_routes
from intent.examples.kernels.routing.moe_align import moe_mark_expert_blocks
from intent.examples.kernels.routing.moe_align import moe_prefix_routes
from intent.examples.kernels.routing.moe_align import moe_scatter_routes
from intent_bench.implementations.context import Tolerance
BLOCK_SIZE = 128
EXPERTS = 64
PADDED_ROUTES = 16320
ROUTES = 8192

def prepare(context):
    (tokens, topk) = (4096, 2)
    source_ids = torch.arange(tokens * topk, dtype=torch.int32).remainder(EXPERTS).reshape(tokens, topk).contiguous()
    count = context.compile(moe_count_routes)
    prefix = context.compile(moe_prefix_routes)
    scatter = context.compile(moe_scatter_routes)
    mark = context.compile(moe_mark_expert_blocks)
    program = MoEAlignment(count, prefix, scatter, mark)
    call = program.prepare(source_ids)
    prepared = intent_sequence((call.count, call.prefix, call.scatter, call.mark),
                               lambda: _canonical_alignment(*call.result()),
                               initializations=call.initializations,
                               host_run=call.launch, host_result=lambda: _canonical_alignment(*call.result()))
    return checked_call(prepared, tuple((Tolerance(atol=0.0) for _ in range(3))), cuda_graph=False, device_type='cpu', note='既有MoE alignment；T4096、top-k2、E64、block128；完整计数/前缀/散射/标记native调用，辅助workspace初始化计入同一原生计时间隔，输出为canonical三元组。', owners=locals())

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
