from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import importlib
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import reference_launch

def prepare(context):
    states = torch.randn((1, 8, 32, 8192), device='cuda', dtype=torch.float32) * 0.01
    decay = -torch.rand((1, 32, 8), device='cuda', dtype=torch.float32) * 0.1
    initial = torch.zeros((1, 32, 8192), device='cuda', dtype=torch.float32)
    source = _source(context, 'ssd_state_passing', '_state_passing_fwd')
    return checked_call(reference_launch(lambda : source(states, decay, initial)), Tolerance(atol=1e-05, rtol=1e-05), device_type='cuda', owners=locals())

def _source(context, module, function):
    load_module(context.project_root / ('corpus/gpu/baselines/triton/state-spaces-mamba/mamba_ssm/ops/triton/' + module + '_runtime.py'), 'intent_bangc_' + module)
    return getattr(importlib.import_module('mamba_ssm.ops.triton.' + module), function)
