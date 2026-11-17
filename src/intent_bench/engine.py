from __future__ import annotations

import math
import platform
import time
import traceback
from dataclasses import asdict, replace
from contextlib import nullcontext
import sys
from typing import Sequence

import numpy as np

from .model import Environment, Implementation, Measurement, Result, Workload
from .measurement import device_window, measure, reset, synchronize
from .trees import assert_close, clone


def benchmark(workload: Workload, implementations: Sequence[str | Implementation] | None = None,
              *, environment: Environment | None = None, measurement: Measurement | None = None,
              baseline: str | None = None, seed: int = 0, observer=None, oracle_output=None):
    env, settings = environment or Environment(), measurement or Measurement()
    if env.threads < 1:
        raise ValueError("threads must be positive")
    # Input construction and the oracle use the same CPU budget as candidates.
    torch_module=sys.modules.get('torch')
    if torch_module is not None:
        torch_module.set_num_threads(env.threads)
    selected = [workload.implementations[x] if isinstance(x, str) else x
                for x in (implementations if implementations is not None else workload.implementations)]
    if not selected or len({i.name for i in selected}) != len(selected):
        raise ValueError("select at least one implementation with unique names")
    baseline = baseline if baseline is not None else workload.default_baseline
    if baseline is not None and baseline not in {i.name for i in selected}:
        raise ValueError(f"baseline {baseline!r} must be selected")
    results = []
    for case_index, case in enumerate(workload.cases):
        inputs = expected = None
        input_error = oracle_error = None
        try:
            inputs = workload.inputs(case, seed)
            if oracle_output is not None:
                import json
                from .storage import write_tree
                captured = oracle_output / f"case-{case_index}"
                write_tree(inputs, captured / "inputs")
                (captured / "case.json").write_text(json.dumps({"name": case.name, "parameters": dict(case.parameters)}))
        except Exception as error:
            input_error = error
        if input_error is None and workload.oracle is not None and oracle_output is None:
            reference_call = None
            try:
                if isinstance(workload.oracle, Implementation):
                    reference_call = workload.oracle.prepare(case, clone(inputs), env)
                    if reference_call.compile:
                        reference_call.compile()
                    with device_window(reference_call.device):
                        reset(reference_call)
                        reference_call.launch()
                        synchronize(reference_call)
                        expected = clone(reference_call.result())
                else:
                    expected = workload.oracle(case, clone(inputs))
            except Exception as error:
                oracle_error = error
            finally:
                if reference_call is not None and reference_call.close:
                    try:reference_call.close()
                    except Exception as error:oracle_error=error
        case_results = []
        for impl_index, impl in enumerate(selected):
            row = Result(workload.name, workload.family, case.name, dict(case.parameters), impl.name,
                         scope=settings.scope, baseline=baseline,
                         metadata={"seed": seed, "machine": platform.node(), "platform": platform.platform(),
                                   "controller_machine": platform.node(),
                                   "threads": env.threads, "compiler": env.compiler})
            if oracle_error is not None:
                row.metadata["reference_error"] = f"{type(oracle_error).__name__}: {oracle_error}"
            call = None
            try:
                if input_error is not None:
                    raise input_error
                row.stage = "prepare"
                if observer: observer(row)
                started = time.perf_counter()
                call = impl.prepare(case, clone(inputs), env)
                if settings.scope == "host_call" and call.host_run is not None:
                    call=replace(call,launch=call.host_run,result=call.host_result or call.result,
                                 native_measure=None,scopes=("host_call",))
                row.stages_seconds["prepare"] = time.perf_counter() - started
                row.device = call.device
                row.metadata.update(call.metadata)
                row.metadata['measurement_contract'] = 'operator-v1' if settings.scope == 'operator' else 'host-call-v1'
                if call.device.startswith('cuda'):
                    import torch
                    device=torch.cuda.get_device_properties(call.device)
                    row.metadata.update(device_name=device.name,compute_capability=[device.major,device.minor],torch_version=torch.__version__)
                if settings.scope not in call.scopes:
                    raise ValueError("implementation does not support the selected call scope")
                row.stage = "native_compile"
                if observer: observer(row)
                started = time.perf_counter()
                if call.compile is not None:
                    call.compile()
                if call.compile is not None:
                    row.stages_seconds["native_compile"] = time.perf_counter() - started
                elif "native_compile_seconds" in call.metadata:
                    row.stages_seconds["native_compile"] = call.metadata["native_compile_seconds"]
                with device_window(call.device):
                    row.stage = "first_call"
                    if observer: observer(row)
                    reset(call)
                    started = time.perf_counter()
                    call.launch()
                    synchronize(call)
                    row.stages_seconds["first_call"] = time.perf_counter() - started
                    if settings.enabled:
                        row.stage = "measurement"
                        if observer: observer(row)
                        samples, row.method = measure(call, settings)
                        row.metadata.update(call.metadata)
                        if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                            raise ValueError("timer must return nonempty finite positive samples in milliseconds")
                        row.samples_ms = [float(x) for x in samples]
                        row.median_ms = float(np.median(samples))
                        row.iqr_ms = float(np.percentile(samples, 75) - np.percentile(samples, 25))
                        if workload.work is not None:
                            work = workload.work(case)
                            if settings.throughput == "tflops" and "flops" in work:
                                row.metrics["tflops"] = work["flops"] / row.median_ms / 1e9
                            if settings.throughput == "gbps" and "bytes" in work:
                                row.metrics["gbps"] = work["bytes"] / row.median_ms / 1e6
                    # Observe the execution actually measured (including graph replay),
                    # rather than validating only the eager warmup before capture.
                    row.stage = "correctness"
                    if oracle_output is not None:
                        from .storage import write_tree
                        write_tree(call.result(), captured / f"actual-{impl_index}")
                        row.metadata["oracle_capture"] = {"case": f"case-{case_index}", "actual": f"actual-{impl_index}"}
                    elif oracle_error is not None:
                        row.correctness = "reference_failed"
                        row.metadata["reference_error"] = f"{type(oracle_error).__name__}: {oracle_error}"
                    elif workload.oracle is not None:
                        try:
                            if workload.check:
                                workload.check(call.result(), expected)
                            else:
                                assert_close(call.result(), expected, atol=workload.atol, rtol=workload.rtol)
                            row.correctness = "passed"
                        except AssertionError as error:
                            row.correctness = "failed"
                            row.error = str(error)
                    row.status = {"passed": "passed", "failed": "numerical_failed",
                                  "unchecked": "unchecked", "reference_failed": "reference_failed"}[row.correctness]
                    if call.compiler_observations is not None:
                        row.stage = "observation"
                        row.metadata["compiler_observations"] = [
                            asdict(observation) for observation in call.compiler_observations()]
                    row.stage = "complete"
            except Exception as error:
                row.status = getattr(error, "status", {
                    "inputs": "input_failed", "prepare": "preparation_failed", "native_compile": "compilation_failed",
                    "first_call": "execution_failed", "correctness": "validation_failed", "measurement": "measurement_failed",
                    "observation": "observation_failed",
                }.get(row.stage, "execution_failed"))
                if isinstance(error,(ImportError,ModuleNotFoundError)):
                    row.status='environment_missing'
                # Preserve compiler-provided stages without classifying unsupported lowering as missing hardware.
                if getattr(error, "stage", None):
                    row.stage = str(error.stage)
                    row.status = ("execution_failed" if row.stage in {"provider_invocation", "provider_tuning"}
                                  else "compilation_failed")
                row.error = f"{type(error).__name__}: {error}"
                row.traceback = traceback.format_exc()
                observation = getattr(error, "observation", None)
                if observation is not None:
                    row.metadata["compiler_observation"] = asdict(observation)
            finally:
                if call is not None and call.close:
                    try:
                        call.close()
                    except Exception as error:
                        row.metadata["close_error"] = str(error)
                        if row.status in {"passed", "unchecked"}:
                            row.status, row.error = "cleanup_failed", str(error)
            case_results.append(row)
            if observer: observer(row)
        assign_speedups(case_results, baseline)
        results.extend(case_results)
        if observer:
            for row in case_results:
                observer(row)
    return results


def assign_speedups(rows, baseline):
    """Use only a numerically checked baseline from the same measured device."""
    for row in rows:
        anchors = [anchor for anchor in rows if anchor.case == row.case and anchor.implementation == baseline
                   and anchor.status == "passed" and anchor.correctness == "passed" and anchor.median_ms]
        if not anchors:
            continue
        anchor = anchors[0]
        if (row.status == "passed" and row.correctness == "passed" and row.median_ms and row.device == anchor.device
                and row.method == anchor.method and row.scope == anchor.scope
                and row.metadata.get("cache_policy") == anchor.metadata.get("cache_policy")
                and row.metadata.get("timing_boundary") == anchor.metadata.get("timing_boundary")):
            row.baseline_ms = anchor.median_ms
            row.speedup = anchor.median_ms / row.median_ms
