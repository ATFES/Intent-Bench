"""Connect authored corpus cases to the common execution and result interfaces."""
from __future__ import annotations

from dataclasses import replace

from .catalog import catalog
from .model import Case


def build(name):
    row = next((item for item in catalog() if item["name"] == name), None)
    if row is None:
        raise KeyError(name)
    provider, kernel = row["provider"], row["kernel"]
    if provider == 'weft':
        from .workloads.weft import build as weft_workload
        return weft_workload(row)
    if provider == 'bangc' and kernel in {'relu','fused_softmax','dense_gemm'}:
        from .workloads.mlu import build as mlu_workload
        operation={'relu':'relu','fused_softmax':'softmax','dense_gemm':'matmul'}[kernel]
        parameters={'relu':{'rows':8192,'columns':4096},'fused_softmax':{'rows':8192,'columns':8192},
                    'dense_gemm':{'rows':4096,'columns':4096,'output_columns':14336}}[kernel]
        return replace(mlu_workload(operation),name=name,cases=[Case(row['case'],parameters)],parameter_keys=())
    if provider == "mojo":
        from .side_workload import mappings,build as side_workload
        side = mappings().get(name, {})
        if "v2_symbol" in side and "inputs" in side:
            return side_workload(row, side)
        from .cpu_inputs import FACTORIES
        if kernel in FACTORIES:
            from .structured import cpu_workload
            return cpu_workload(row, FACTORIES[kernel])
    if provider in {"triton","cutile","mojo","bangc"}:
        from .side_workload import mappings,build as side_workload
        sides=mappings()
        if name in sides:
            return side_workload(row,sides[name])
    raise KeyError(f'no adapter registered for {name}')
