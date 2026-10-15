from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.references.torch_ops import fp8_mqa_logits
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (queries, keys, heads, dimension) = (256, 4096, 32, 128)
    q = (torch.randn((queries, heads, dimension), device='cuda', dtype=torch.float16) * 0.25).to(torch.float8_e4m3fn)
    kv = (torch.randn((keys, dimension), device='cuda', dtype=torch.float16) * 0.25).to(torch.float8_e4m3fn)
    kv_scale = 0.5 + torch.rand((keys,), device='cuda', dtype=torch.float32)
    head_weight = torch.rand((queries, heads), device='cuda', dtype=torch.float32) * 0.1
    indices = torch.arange(queries, device='cuda', dtype=torch.int32)
    key_start = indices % 17
    key_end = keys - indices % 19
    arguments = (q, kv, kv_scale, head_weight, key_start, key_end)
    source = functional_launch(lambda : fp8_mqa_logits(*arguments))
    return Side(source, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, owners=locals())
