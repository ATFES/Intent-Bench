from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.statistics.histogram import histogram_256
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    samples = torch.randint(0, 256, (8 * 1024 * 1024,), device='cuda', dtype=torch.int32).to(torch.float32)
    (_, generated_base) = compile_single(context, histogram_256, (samples,))
    generated = PreparedLaunch(launch=generated_base.launch, outputs=generated_base.outputs, prepare=generated_base.outputs().zero_)
    return Side(generated, Tolerance(atol=0.0), cuda_graph=False, owners=locals())
