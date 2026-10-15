from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import initial_launch
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (tokens, streams) = (8192, 4)
    logits = torch.randn((tokens, streams, streams), device='cuda', dtype=torch.float32)
    source_module = _source(context)
    packed = torch.zeros((tokens, streams * (streams + 2)), device='cuda', dtype=torch.float32)

    def prepare():
        packed[:, 2 * streams:].copy_(logits.view(tokens, -1))

    def launch():
        source_module.mhc_sinkhorn(packed, streams)
    prepare()
    initial_launch(launch, side='source')
    source = PreparedLaunch(launch=launch, outputs=lambda : packed[:, 2 * streams:].view(tokens, streams, streams), prepare=prepare)
    return Side(source, Tolerance(atol=0.0002, rtol=0.0001), cuda_graph=False, note='同算法；source 用 exp2(x * log2(e)) 实现 exp', owners=locals())

def _source(context):
    return tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/mhc/fused/mhc.py', 'mhc')
