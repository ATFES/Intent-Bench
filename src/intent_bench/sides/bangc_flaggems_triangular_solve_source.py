from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.references.torch_ops import triangular_solve
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import reference_launch

def prepare(context):
    (batch, size) = (4096, 16)
    lower = torch.tril(torch.randn((batch, size, size), device='cuda', dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((batch, size), device='cuda', dtype=torch.float32)
    return Side(reference_launch(lambda : triangular_solve(*(lower, initial.clone()))), Tolerance(atol=0.0001, rtol=0.0001), device_type='cuda', owners=locals())
