from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from programs.composition import MoEAlignment
from kernels.routing.moe_align import moe_count_routes
from kernels.routing.moe_align import moe_mark_expert_blocks
from kernels.routing.moe_align import moe_prefix_routes
from kernels.routing.moe_align import moe_scatter_routes
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
BLOCK_SIZE = 128
EXPERTS = 64
PADDED_ROUTES = 16320
ROUTES = 8192

def prepare(context):
    (tokens, topk) = (4096, 2)
    source_ids = torch.arange(tokens * topk, dtype=torch.int32).remainder(EXPERTS).reshape(tokens, topk).contiguous()
    expert_counts = torch.zeros((EXPERTS,), dtype=torch.int32)
    expert_cursors = torch.zeros_like(expert_counts)
    sorted_routes = torch.full((PADDED_ROUTES,), ROUTES, dtype=torch.int32)
    count = intent.compile(moe_count_routes, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    prefix = intent.compile(moe_prefix_routes, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    scatter = intent.compile(moe_scatter_routes, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    mark = intent.compile(moe_mark_expert_blocks, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_state: dict[str, tuple[torch.Tensor, ...]] = {}
    program = MoEAlignment(count, prefix, scatter, mark)

    def generated_launch():
        generated_state['output'] = program.run_into(source_ids, expert_counts, expert_cursors, sorted_routes)
    return Side(PreparedLaunch(generated_launch, lambda : _canonical_alignment(*generated_state['output'])), tuple((Tolerance(atol=0.0) for _ in range(3))), cuda_graph=False, device_type='cpu', note='既有MoE alignment；T4096、top-k2、E64、block128；完整计数/前缀/散射/标记调用，辅助workspace初始化计入调用，输出为canonical三元组。', owners=locals())

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
