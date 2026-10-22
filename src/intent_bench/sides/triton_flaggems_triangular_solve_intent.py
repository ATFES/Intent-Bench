from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.factorization.triangular_solve import batched_lower_triangular_solve
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, size) = (4096, 16)
    lower = torch.tril(torch.randn((batch, size, size), device='cuda', dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((batch, size), device='cuda', dtype=torch.float32)
    generated_solution = initial.clone()
    (_, generated_base) = compile_single(context, batched_lower_triangular_solve, (lower, generated_solution))
    generated = PreparedCall(launch=generated_base.launch, result=lambda : generated_solution, reset=lambda : generated_solution.copy_(initial))
    return checked_call(generated, Tolerance(atol=0.0001, rtol=0.0001), cuda_graph=False, owners=locals())
