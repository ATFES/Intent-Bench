from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.streaming.mla import splitk_mla_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_reduce_f16
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    arguments = _absorbed_decode_inputs()
    partials = intent.compile(splitk_mla_decode_partials, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'SPLITS': 16, 'SPLIT_SIZE': 512})
    reduction = intent.compile(splitk_attention_reduce_f16, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(*arguments)
        generated['output'] = reduction.run(partial, lse)
    return Side(PreparedLaunch(launch, lambda : generated['output']), Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='原B8/H64/S8192/C512/R64 f16 split512/P16 MLA完整partial+reduce，含workspace分配；partial f16/LSE f32，CPU完整attention数学参考f32，最终f16，原容差。', owners=locals())

def _absorbed_decode_inputs():
    query = torch.randn((8, 64, 512), dtype=torch.float16)
    query_rope = torch.randn((8, 64, 64), dtype=torch.float16)
    cache = torch.randn((8, 8192, 512), dtype=torch.float16)
    cache_rope = torch.randn((8, 8192, 64), dtype=torch.float16)
    return (query, query_rope, cache, cache_rope, 576 ** (-0.5))
