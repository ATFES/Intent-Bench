from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.variants.decomposition import moe_count_routes_product_domain
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    ids = torch.randint(0, 64, (4096, 2), dtype=torch.int32)
    artifact = intent.compile(moe_count_routes_product_domain, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def side(function):
        counts = torch.zeros((64,), dtype=torch.int32)
        return PreparedLaunch(lambda : function(ids, counts), lambda : counts, prepare=lambda : counts.zero_())
    return Side(side(artifact.run), Tolerance(atol=0.0), cuda_graph=False, device_type='cpu', note='原product-domain MoE count T4096/top2/E64、随机i32 IDs，relaxed i32 atomic add；独立counts每次清零且清零不计时，CPU bincount参考，精确比较，完整host调用。', owners=locals())
