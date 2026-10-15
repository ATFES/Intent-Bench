from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    q = torch.randn((8, 32, 1, 128), dtype=torch.bfloat16)
    k = torch.randn((8, 8, 8192, 128), dtype=torch.bfloat16)
    v = torch.randn_like(k)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    (generated, source) = ({}, {})

    def reference():
        source['output'] = runtime.grouped_flash_decode(q, k, v, 128 ** (-0.5))
    return Side(PreparedLaunch(reference, lambda : source['output']), Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='原B8/HQ32/HK8/K8192/D128 BF16 grouped decode，split256/P32完整partial+reduce含workspace分配；生成LSEf32/partial BF16，CPU数学参考f32概率，最终BF16[B,H,1,D]，原容差。', owners=locals())
