from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    from .attention_inputs import grouped
    q,k,v,scale = context.inputs if context.inputs is not None else grouped().arguments
    split_size = 256
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/flash_decode/flash_decode.py', 'grouped_flash_decode', needs_utils=True, needs_splitk=True)
    source = functional_launch(lambda : source_module.fmha_decode(q, k, v, scale, kv_len_per_split=split_size))
    return checked_call(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
