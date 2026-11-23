from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent.examples.kernels.streaming.attention import grouped_flash_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    from .attention_inputs import grouped
    q,k,v,scale = context.inputs if context.inputs is not None else grouped().arguments
    split_size = 256
    query_heads, key_heads, sequence = q.shape[1], k.shape[1], k.shape[2]
    splits = (sequence + split_size - 1) // split_size
    (_, partials) = compile_single(context, grouped_flash_decode_partials, (q, k, v, scale), constexprs={'HEAD_GROUP': query_heads // key_heads, 'P': splits, 'SPLIT_SIZE': split_size})
    (partial_lse, partial_output) = partials.result()
    (_, reduction) = compile_single(context, splitk_attention_reduce, (partial_output, partial_lse))

    def generated_launch():
        partials.launch()
        reduction.launch()
    generated = PreparedCall(generated_launch, lambda : reduction.result().unsqueeze(2))
    return checked_call(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
