from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.streaming.attention import grouped_flash_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    q = torch.randn((8, 32, 1, 128), dtype=torch.bfloat16)
    k = torch.randn((8, 8, 8192, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    partials = intent.compile(grouped_flash_decode_partials, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'HEAD_GROUP': 4, 'P': 32, 'SPLIT_SIZE': 256})
    reduction = intent.compile(splitk_attention_reduce, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(q, k, v, 128 ** (-0.5))
        generated['output'] = reduction.run(partial, lse).unsqueeze(2)
    return Side(PreparedLaunch(launch, lambda : generated['output']), Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='原B8/HQ32/HK8/K8192/D128 BF16 grouped decode，split256/P32完整partial+reduce含workspace分配；生成LSEf32/partial BF16，CPU数学参考f32概率，最终BF16[B,H,1,D]，原容差。', owners=locals())
