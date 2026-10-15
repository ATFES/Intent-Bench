from __future__ import annotations
from intent_bench.sides import Side
import torch
from types import SimpleNamespace
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import reference_launch

def prepare(context):
    q = torch.randn((8, 128, 32), dtype=torch.float32)
    (k, v) = (torch.randn_like(q), torch.randn_like(q))
    scale = 32 ** (-0.5)
    reference = _mojo_reference(context, 'corpus/cpu/baselines/mojo/intentdsl/attention/causal/causal_runtime.py', (q, k, v, scale))
    return Side(reference_launch(reference), Tolerance(0.0002, 1e-05), device_type='cpu', owners=locals())

def _mojo_reference(context, relative_path, arguments):
    runtime = load_module(context.project_root / relative_path, 'intent_bangc_mojo_' + relative_path.split('/')[-2])
    source = runtime.prepare(SimpleNamespace(workers=8,executable=context.environment.options.get('mojo')), *arguments)

    def reference():
        source.launch()
        return source.result()
    return reference
