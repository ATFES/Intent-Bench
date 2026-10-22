from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.optimization.adamw import adamw_update
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    elements = 8 * 1024 * 1024
    gradient = torch.randn((elements,), device='cuda', dtype=torch.float32)
    initial_parameter = torch.randn_like(gradient)
    initial_first = torch.randn_like(gradient)
    initial_second = torch.rand_like(gradient) + 0.5
    generated_parameter = initial_parameter.clone()
    generated_first = initial_first.clone()
    generated_second = initial_second.clone()
    scalars = (0.001, 0.9, 0.999, 1.0 - 0.9 ** 10, 1.0 - 0.999 ** 10, 1e-08, 0.01)
    (_, generated_base) = compile_single(context, adamw_update, (gradient, generated_parameter, generated_first, generated_second, *scalars))
    generated = PreparedCall(launch=generated_base.launch, result=lambda : (generated_parameter, generated_first, generated_second), reset=lambda : (generated_parameter.copy_(initial_parameter), generated_first.copy_(initial_first), generated_second.copy_(initial_second)))
    return checked_call(generated, Tolerance(atol=1e-05, rtol=1e-05), cuda_graph=False, owners=locals())
