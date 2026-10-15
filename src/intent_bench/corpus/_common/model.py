from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import torch

if TYPE_CHECKING:
    from intent import CompileOptions


TensorTree: TypeAlias = torch.Tensor | tuple["TensorTree", ...]


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
class PreparedLaunch:
    launch: Callable[[], object]
    outputs: Callable[[], TensorTree]
    prepare: Callable[[], object] | None = None
    native_benchmark: Callable[[], float] | None = None


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
    artifacts: list[str] = field(default_factory=list)
    environment: object | None = None


class ComparisonUnavailable(RuntimeError):
    """A migrated source reports a specific unavailable implementation status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
