"""Launch preparation for migrated adapters; measurement lives in intent_bench.measurement."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
import torch

from .model import Context, PreparedLaunch, TensorTree


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


CUTILE_TUNING_LAUNCH_TIMEOUT_SECONDS = 3

_stage_observer: ContextVar[Callable[[str], None] | None] = ContextVar("preparation_stage_observer", default=None)


@contextmanager
def observe_stages(observer: Callable[[str], None]):
    token = _stage_observer.set(observer)
    try:
        yield
    finally:
        _stage_observer.reset(token)


def report_stage(stage: str) -> None:
    observer = _stage_observer.get()
    if observer is not None:
        observer(stage)


def initial_launch(function, *, side: str):
    stage = f"{side}_provider_jit_or_initial_launch"
    report_stage(stage)
    try:
        result = function()
        torch.cuda.synchronize()
    except Exception as error:
        raise PipelineStageError(stage, str(error)) from error
    report_stage("adapter_preparation")
    return result


def compile_kernel(
    context: Context,
    definition,
    *,
    constexprs: dict[str, object] | None = None,
):
    import intent
    report_stage("generated_compilation")
    try:
        artifact = intent.compile(
            definition,
            target=context.target,
            compiler=context.compiler,
            constexprs=constexprs,
            tuning_config=context.tuning_config,
            options=context.compile_options,
        )
    except intent.CompilationStageError as error:
        raise PipelineStageError(f"generated_{error.stage}", str(error)) from error
    context.artifacts.append(str(artifact.cache_directory))
    return artifact


def prepare_program(program, arguments: tuple[object, ...]) -> PreparedLaunch:
    """Compile and warm one complete author callable with its owned workspace."""
    import intent
    report_stage("generated_launcher_preparation")
    try:
        call = program.prepare(*arguments)
    except Exception as error:
        raise PipelineStageError("generated_launcher_preparation", str(error)) from error
    report_stage("generated_native_compilation")
    try:
        call.compile()
    except intent.CompilationStageError as error:
        raise PipelineStageError(f"generated_{error.stage}", str(error)) from error
    initial_launch(call.launch, side="generated")
    report_stage("generated_launcher_preparation")
    try:
        call.launch()
    except Exception as error:
        raise PipelineStageError("generated_launcher_preparation", str(error)) from error
    report_stage("adapter_preparation")
    return PreparedLaunch(launch=call.launch, outputs=call.result)


def compile_single(
    context: Context,
    definition,
    arguments: tuple[object, ...],
    *,
    constexprs: dict[str, object] | None = None,
) -> tuple[object, PreparedLaunch]:
    artifact = compile_kernel(context, definition, constexprs=constexprs)
    return artifact, prepare_program(artifact, arguments)


def functional_launch(function) -> PreparedLaunch:
    state: dict[str, TensorTree] = {}

    def launch():
        state["output"] = function()
        return state["output"]

    initial_launch(launch, side="source")
    return PreparedLaunch(launch=launch, outputs=lambda: state["output"])
