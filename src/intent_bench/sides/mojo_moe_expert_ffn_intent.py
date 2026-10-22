from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import intent
import torch
from intent.examples.kernels.ragged.moe import moe_expert_ffn
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call
MOE_EXPERTS = 8
MOE_HIDDEN = 4096
MOE_INTERMEDIATE = 14336
MOE_TOKENS = 4096
MOE_TOP_K = 2

def prepare(context):
    (route_offsets, member_routes, route_token, route_weights) = _make_moe_routes()
    x = torch.randn((MOE_TOKENS, MOE_HIDDEN), dtype=torch.float16)
    x /= math.sqrt(MOE_HIDDEN)
    w1 = torch.randn((MOE_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE), dtype=torch.float16)
    w1 /= math.sqrt(MOE_HIDDEN)
    w2 = torch.randn((MOE_EXPERTS, MOE_INTERMEDIATE, MOE_HIDDEN), dtype=torch.float16)
    w2 /= math.sqrt(MOE_INTERMEDIATE)
    artifact = context.compile(moe_expert_ffn)
    generated_y = torch.zeros((MOE_TOKENS, MOE_HIDDEN), dtype=torch.float32)
    arguments = (x, route_offsets, member_routes, route_token, route_weights, w1, w2)

    generated = intent_call(artifact, (*arguments, generated_y), observe=lambda _: generated_y,
                            reset=lambda: generated_y.zero_())
    return checked_call(generated, Tolerance(atol=0.002), device_type='cpu', note='Prepared complete MoE expert FFN; f16 inputs/weights and f32 InOut. Restore y=0 outside timing.', owners=locals())

def _make_moe_routes():
    route_count = MOE_TOKENS * MOE_TOP_K
    route_token = torch.arange(MOE_TOKENS, dtype=torch.int32).repeat_interleave(MOE_TOP_K)
    route_weights = torch.full((route_count,), 1.0 / MOE_TOP_K, dtype=torch.float32)
    topk_ids = (torch.arange(route_count) % MOE_EXPERTS).reshape(MOE_TOKENS, MOE_TOP_K)
    flat_experts = topk_ids.reshape(-1)
    member_routes = torch.argsort(flat_experts, stable=True).to(torch.int32)
    counts = torch.bincount(flat_experts, minlength=MOE_EXPERTS)
    route_offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).to(torch.int32)
    return (route_offsets, member_routes, route_token, route_weights)
