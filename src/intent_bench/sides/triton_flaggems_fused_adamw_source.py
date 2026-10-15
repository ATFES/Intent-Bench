from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.references.torch_ops import adamw
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    elements = 8 * 1024 * 1024
    gradient = torch.randn((elements,), device='cuda', dtype=torch.float32)
    initial_parameter = torch.randn_like(gradient)
    initial_first = torch.randn_like(gradient)
    initial_second = torch.rand_like(gradient) + 0.5
    scalars = (0.001, 0.9, 0.999, 1.0 - 0.9 ** 10, 1.0 - 0.999 ** 10, 1e-08, 0.01)
    source_parameter = initial_parameter.clone()
    source_first = initial_first.clone()
    source_second = initial_second.clone()
    source_arguments = (gradient, source_parameter, source_first, source_second, *scalars)
    source = PreparedLaunch(launch=lambda : adamw(*source_arguments), outputs=lambda : (source_parameter, source_first, source_second), prepare=lambda : (source_parameter.copy_(initial_parameter), source_first.copy_(initial_first), source_second.copy_(initial_second)))
    return Side(source, Tolerance(atol=1e-05, rtol=1e-05), cuda_graph=False, owners=locals())
