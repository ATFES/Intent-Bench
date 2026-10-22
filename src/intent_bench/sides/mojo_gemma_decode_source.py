from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension, split_size) = (32, 32, 8, 8192, 128, 256)
    q = torch.randn((batch, query_heads, 1, dimension), dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_gemma_decode')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.gemma_decode(q, k, v, scale, window_size=1024, soft_cap=50.0, kv_len_per_split=split_size)
    return checked_call(PreparedCall(source_launch, lambda : source_state['output']), Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='完整 Gemma GQA decode partial+split-k pipeline；生成端输出及 partial 预分配，CPU reference 为 PyTorch eager，计入其临时分配；原输入、输出、容差不变。', owners=locals())
