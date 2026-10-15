from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.streaming.mla import paged_mla_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    query = torch.randn((8, 128, 512), dtype=torch.float16)
    query_rope = torch.randn((8, 128, 64), dtype=torch.float16)
    cache = torch.randn((4096, 16, 1, 512), dtype=torch.float16)
    cache_rope = torch.randn((4096, 16, 1, 64), dtype=torch.float16)
    offsets = torch.arange(0, 4097, 512, dtype=torch.int32)
    indices = torch.arange(4096, dtype=torch.int32)
    lengths = torch.full((8,), 8192, dtype=torch.int32)
    flat_arguments = (query, query_rope, cache.reshape(65536, 1, 512), cache_rope.reshape(65536, 1, 64), offsets, indices, lengths, 576 ** (-0.5))
    partials = intent.compile(paged_mla_decode_partials, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'PAGE_SIZE': 16, 'HEAD_GROUP': 128, 'SPLITS': 8})
    reduction = intent.compile(splitk_attention_f32_to_f16_reduce, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    (generated, source) = ({}, {})

    def launch():
        (lse, partial) = partials.run(*flat_arguments)
        generated['output'] = reduction.run(partial, lse)
    return Side(PreparedLaunch(launch, lambda : generated['output']), Tolerance(atol=0.05, rtol=0.05), cuda_graph=False, device_type='cpu', note='原B8/HQ128/HK1/K8192/C512/R64/page16/split8 f16 MLA完整partial+reduce，含workspace分配；生成flattened page cache与f32 partial/LSE，CPU完整page-table attention数学参考，最终f16，原容差。', owners=locals())
