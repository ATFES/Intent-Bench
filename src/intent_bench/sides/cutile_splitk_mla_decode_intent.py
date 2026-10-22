from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent.examples.kernels.streaming.mla import splitk_mla_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_reduce_f16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, heads, sequence, latent, rope, split_size) = (8, 64, 8192, 512, 64, 512)
    splits = sequence // split_size
    query = torch.randn((batch, heads, latent), device='cuda', dtype=torch.float16)
    query_rope = torch.randn((batch, heads, rope), device='cuda', dtype=torch.float16)
    cache = torch.randn((batch, sequence, latent), device='cuda', dtype=torch.float16)
    cache_rope = torch.randn((batch, sequence, rope), device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(latent + rope)
    (_, partials) = compile_single(context, splitk_mla_decode_partials, (query, query_rope, cache, cache_rope, scale), constexprs={'SPLITS': splits, 'SPLIT_SIZE': split_size})
    (partial_lse, partial_output) = partials.result()
    (_, reduction) = compile_single(context, splitk_attention_reduce_f16, (partial_output, partial_lse))

    def generated_launch():
        partials.launch()
        reduction.launch()
    generated = PreparedCall(generated_launch, reduction.result)
    return checked_call(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
