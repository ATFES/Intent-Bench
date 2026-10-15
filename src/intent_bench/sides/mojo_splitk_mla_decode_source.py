from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    arguments = _absorbed_decode_inputs()
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    (generated, source) = ({}, {})

    def reference():
        source['output'] = runtime.absorbed_mla_decode(*arguments)
    return Side(PreparedLaunch(reference, lambda : source['output']), Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, device_type='cpu', note='原B8/H64/S8192/C512/R64 f16 split512/P16 MLA完整partial+reduce，含workspace分配；partial f16/LSE f32，CPU完整attention数学参考f32，最终f16，原容差。', owners=locals())

def _absorbed_decode_inputs():
    query = torch.randn((8, 64, 512), dtype=torch.float16)
    query_rope = torch.randn((8, 64, 64), dtype=torch.float16)
    cache = torch.randn((8, 8192, 512), dtype=torch.float16)
    cache_rope = torch.randn((8, 8192, 64), dtype=torch.float16)
    return (query, query_rope, cache, cache_rope, 576 ** (-0.5))
