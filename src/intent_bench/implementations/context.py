from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import torch
    from intent import CompileOptions


TensorTree: TypeAlias = "torch.Tensor | tuple[TensorTree, ...]"


@dataclass(frozen=True)
class Entry:
    kernel: str
    case: str
    generated: str
    examples: tuple[str, ...]
    source_runtime: str
    deployment: str | None = None


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float = 0.0


@dataclass(frozen=True)
class IntegerTolerance:
    max_abs: int


@dataclass(frozen=True)
class SimilarityTolerance:
    max_error: float


NumericalTolerance: TypeAlias = Tolerance | IntegerTolerance | SimilarityTolerance


@dataclass(frozen=True)
class Context:
    compiler: str
    project_root: Path
    target: object
    provider: str
    compiler_timeout_seconds: int = 15
    tuning_config: Path | None = None
    compile_options: CompileOptions | None = None
    intent_root: Path | None = None
    kernels: list = field(default_factory=list)
    prepared_calls: list = field(default_factory=list)
    environment: object | None = None

    @property
    def artifacts(self):
        return [str(artifact.cache_directory) for kernel in self.kernels for artifact in kernel.artifacts]

    def compile(self, definition, *, constexprs=None):
        from intent_bench.implementations.intent import make_kernel
        kernel = make_kernel(definition, self.environment, self.provider,
                             constexprs=constexprs, tuning_config=self.tuning_config,
                             options=self.compile_options)
        self.kernels.append(kernel)
        return kernel


class ComparisonUnavailable(RuntimeError):
    """A migrated source reports a specific unavailable implementation status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
