from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.ragged.grouped_gemm import routed_expert_projection_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (tokens, hidden, intermediate, experts, topk) = (4096, 4096, 14336, 8, 2)
    hidden_states = torch.randn((tokens, hidden), device='cuda', dtype=torch.bfloat16)
    weights = torch.randn((experts, intermediate, hidden), device='cuda', dtype=torch.bfloat16)
    topk_ids = torch.arange(tokens * topk, device='cuda', dtype=torch.long).remainder(experts).reshape(tokens, topk).contiguous()
    flat_experts = topk_ids.flatten()
    member_routes = torch.argsort(flat_experts, stable=True).to(torch.int32)
    expert_offsets = torch.empty((experts + 1,), device='cuda', dtype=torch.int32)
    expert_offsets[0] = 0
    expert_offsets[1:] = torch.cumsum(torch.bincount(flat_experts, minlength=experts), dim=0).to(torch.int32)
    (_, generated) = compile_single(context, routed_expert_projection_bf16, (hidden_states, expert_offsets, member_routes, weights), constexprs={'TOP_K': topk})
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, note='同算法；source 使用预计算 block ownership', owners=locals())
