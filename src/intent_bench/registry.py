"""Declaration-only workload registration shared by discovery and execution."""
from dataclasses import dataclass
from functools import cache
from .catalog import catalog, ADVANCED_OPERATIONS


@dataclass(frozen=True)
class Registration:
    metadata: dict
    factory: str
    argument: str

    def build(self):
        from .loading import load
        return load(self.factory)(self.argument)


@cache
def workloads():
    from .implementations.mojo import NAMES as CPU_NAMES
    from .workloads.mlu import NAMES as MLU_NAMES
    from .workloads.rvv import NAMES as RVV_NAMES
    values = {}
    def add(metadata, factory, argument):
        name = metadata["name"]
        if name in values:
            raise ValueError(f"duplicate workload registration: {name}")
        values[name] = Registration(metadata, factory, argument)
    for name, family in (("softmax", "reduction"), ("relu", "pointwise")):
        add({"name":name, "family":family,
             "implementations":["numpy","torch","triton","cutile","intent:triton","intent:cutile","intent:mojo"]+(["intent-v2:triton","intent-v2:weft","intent-v2:mojo","intent:weft","ggml"] if name=="softmax" else [])},
            "intent_bench.workloads.elementwise:build", name)
    add({"name":"weighted_rms_norm","family":"normalization","implementations":["torch","intent:triton","intent-v2:triton","ggml","intent:weft","intent:mojo","intent-v2:weft","intent-v2:mojo"]},
        "intent_bench.workloads.weighted_rms:build", "weighted_rms_norm")
    for row in catalog():
        add(row, "intent_bench.corpus_adapter:build", row["name"])
    from .workloads.programs import BASELINES as PROGRAM_BASELINES, backends as program_backends
    providers = tuple("intent:" + backend for backend in program_backends())
    for name,baselines in PROGRAM_BASELINES.items():
        add({"name":"program/"+name, "family":"program", "case":"author",
             "description":"Installed author program with its original inputs and complete call",
             "implementation_descriptions":dict(baselines),
             "implementations":[*baselines,*providers]},
            "intent_bench.workloads.programs:build", name)
    for group, names, implementations in (
        ("advanced", ADVANCED_OPERATIONS, ["torch","mojo","intent:mojo"]),
        ("cpu", CPU_NAMES, ["numpy","mojo"]),
        ("mlu", MLU_NAMES, ["cnnl","intent:bangc"]),
        ("rvv", RVV_NAMES, ["ggml","intent:weft"]),
    ):
        for name in names:
            add({"name":group+"/"+name, "family":group, "implementations":implementations},
                f"intent_bench.workloads.{group}:build", name)
    return values
