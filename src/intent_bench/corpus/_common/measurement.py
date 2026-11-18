"""Launch preparation for migrated adapters; measurement lives in intent_bench.measurement."""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
import torch

from intent_bench.implementations.context import Context, TensorTree
from intent_bench.model import PreparedCall


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
        artifact = context.compile(definition, constexprs=constexprs)
    except intent.CompilationStageError as error:
        raise PipelineStageError(f"generated_{error.stage}", str(error)) from error
    return artifact


def prepare_program(program, arguments: tuple[object, ...], *, outputs: tuple | None = None) -> PreparedCall:
    """Bind a complete author callable without compiling, tuning or executing it."""
    report_stage("generated_launcher_preparation")
    try:
        call = (program.prepare(*arguments) if outputs is None
                else program.prepare(*arguments, outputs=outputs))
    except Exception as error:
        raise PipelineStageError("generated_launcher_preparation", str(error)) from error
    report_stage("adapter_preparation")
    return PreparedCall(launch=call.launch, result=call.result, compile=call.compile)


def compile_single(
    context: Context,
    definition,
    arguments: tuple[object, ...],
    *,
    constexprs: dict[str, object] | None = None,
    outputs: tuple | None = None,
) -> tuple[object, PreparedCall]:
    artifact = compile_kernel(context, definition, constexprs=constexprs)
    call = prepare_program(artifact, arguments, outputs=outputs)
    context.prepared_calls.append(call)
    return artifact, call


def functional_launch(function) -> PreparedCall:
    state: dict[str, TensorTree] = {}

    def launch():
        state["output"] = function()
        return state["output"]

    return PreparedCall(launch=launch, result=lambda: state["output"])
