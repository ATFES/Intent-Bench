from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.streaming.attention_specialized import attention_sink_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    q = torch.randn((32, 32, 128), dtype=torch.bfloat16)
    k = torch.randn((32, 8192, 8, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((32,), dtype=torch.bfloat16)
    start = torch.tensor([8191], dtype=torch.int32)
    partials = intent.compile(attention_sink_decode_partials, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'HEAD_GROUP': 4, 'WINDOW': 0, 'P': 32})
    reduction = intent.compile(splitk_attention_reduce, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(q, k, v, sinks, start, 128 ** (-0.5))
        generated['output'] = reduction.run(partial, lse).reshape(32, 1, 4096)
    return Side(PreparedLaunch(launch, lambda : generated['output']), Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='原BF16 sink decode B32/S8192/HQ32/HK8/D128、start8191、split256/P32；完整partial+reduce含workspace分配，生成LSEf32/partial BF16；CPU数学reference f32 softmax，最终BF16，原容差。', owners=locals())
