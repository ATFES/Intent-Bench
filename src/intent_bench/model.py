from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


class BenchError(RuntimeError):
    status = "execution_failed"


class ExpressionGap(BenchError):
    status = "expression_gap"


class EnvironmentUnavailable(BenchError):
    status = "environment_missing"


class ImplementationMissing(BenchError):
    status = "implementation_missing"


class DeviceBusy(BenchError):
    status = "skipped_busy"


@dataclass(frozen=True)
class Case:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Environment:
    device: str = "cpu"
    intent_root: Path | None = None
    compiler: str | None = None
    threads: int = 1
    options: Mapping[str, Any] = field(default_factory=dict)
    transport: str | None = None
    host: str | None = None
    python: str | None = None
    name: str = "local"
    output_root: Path | None = None
    remote_output_root: str | None = None
    cpus: tuple[int, ...] | None = None
    launcher: tuple[str, ...] = ()


@dataclass(frozen=True)
class Measurement:
    method: str = "auto"
    samples: int | None = None
    warmup: int | None = None
    scope: str = "operator"
    enabled: bool = True
    cache: str = "flush"
    warmup_ms: float = 25
    rep_ms: float = 100
    throughput: str | None = None

    def __post_init__(self):
        if self.scope == "prepared_call":
            object.__setattr__(self, "scope", "operator")
        if (self.samples is not None and self.samples < 1) or (self.warmup is not None and self.warmup < 0):
            raise ValueError("samples must be positive and warmup nonnegative")
        if self.warmup_ms < 0 or self.rep_ms <= 0:
            raise ValueError("warmup_ms must be nonnegative and rep_ms positive")
        if self.cache not in {"flush", "warm"}:
            raise ValueError("cache must be flush or warm")
        if self.throughput not in {None, "tflops", "gbps"}:
            raise ValueError("throughput must be tflops or gbps")
        if self.method not in {"auto", "host", "cuda_event", "cuda_graph", "native"}:
            raise ValueError(f"unknown timing method: {self.method}")
        if self.scope not in {"operator", "host_call"}:
            raise ValueError("scope must be operator or host_call")
        if self.scope == "host_call" and self.method not in {"auto", "host"}:
            raise ValueError("host_call measures Python entry-to-completion time; use auto or host timing")


@dataclass
class PreparedCall:
    launch: Callable[[], Any]
    result: Callable[[], Any]
    reset: Callable[[], None] | None = None
    compile: Callable[[], None] | None = None
    synchronize: Callable[[], None] | None = None
    close: Callable[[], None] | None = None
    native_measure: Callable[[Measurement], Sequence[float]] | None = None
    device: str = "cpu"
    timing: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    scopes: tuple[str, ...] = ("operator",)
    host_run: Callable[[], Any] | None = None
    host_result: Callable[[], Any] | None = None
    tolerance: Any = None
    owners: Any = None
    compiler_observations: Callable[[], Sequence[Any]] | None = None

    def __post_init__(self):
        self.scopes = tuple("operator" if scope == "prepared_call" else scope for scope in self.scopes)


@dataclass(frozen=True)
class Implementation:
    name: str
    prepare: Callable[[Case, Any, Environment], PreparedCall]
    description: str = ""


@dataclass
class Workload:
    name: str
    cases: Sequence[Case]
    inputs: Callable[[Case, int], Any]
    implementations: Mapping[str, Implementation]
    oracle: Callable[[Case, Any], Any] | Implementation | None = None
    check: Callable[[Any, Any], None] | None = None
    atol: float = 1e-5
    rtol: float = 1e-5
    family: str = "custom"
    description: str = ""
    work: Callable[[Case], Mapping[str, float]] | None = None
    default_baseline: str | None = None
    parameter_keys: tuple[str, ...] | None = None
    materialized_inputs: bool = True
    prepare_inputs: Callable[[Case, Any, Environment], Any] | None = None

    def __post_init__(self):
        names = [case.name for case in self.cases]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate case names in {self.name}")


@dataclass
class Result:
    workload: str
    family: str
    case: str
    parameters: dict
    implementation: str
    status: str = "pending"
    correctness: str = "unchecked"
    stage: str = "inputs"
    device: str = ""
    method: str = ""
    scope: str = ""
    samples_ms: list[float] = field(default_factory=list)
    median_ms: float | None = None
    iqr_ms: float | None = None
    baseline: str | None = None
    baseline_ms: float | None = None
    speedup: float | None = None
    stages_seconds: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
