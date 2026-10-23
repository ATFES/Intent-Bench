from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.variants.decomposition import moe_count_routes_product_domain
from intent_bench.implementations.context import Tolerance

def prepare(context):
    ids = torch.randint(0, 64, (4096, 2), dtype=torch.int32)
    artifact = context.compile(moe_count_routes_product_domain)

    def side():
        counts = torch.zeros((64,), dtype=torch.int32)
        return intent_call(artifact, (ids, counts), reset=lambda: counts.zero_(), observe=lambda _: counts)
    return checked_call(side(), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='原product-domain MoE count T4096/top2/E64、随机i32 IDs，relaxed i32 atomic add；独立counts每次清零且清零不计时，CPU bincount参考，精确比较；默认计prepared native调用。', owners=locals())
