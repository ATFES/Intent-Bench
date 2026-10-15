from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.indexing.relations import roll_rows_forward
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    (_, generated_flat) = compile_single(context, roll_rows_forward, (x.view(-1),), constexprs={'ROW_WIDTH': x.shape[1]})
    generated = PreparedLaunch(generated_flat.launch, lambda : generated_flat.outputs().view_as(x))
    return Side(generated, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
