"""MLU row competitors with a common CNRT notifier interval."""

from dataclasses import asdict
from functools import partial
import ctypes as C
import numpy as np
from ..model import Implementation, PreparedCall, EnvironmentUnavailable

NAMES = (
    "cnnl",
    "intent:bangc",
    "intent-v2:bangc",
    "intent-v2:bangc-direct",
    "intent-v2:bangc-double",
)


def register(name):
    return {k: Implementation(k, partial(prepare, name, k)) for k in NAMES}


def timer(launch, stream, settings, reset=None):
    lib = C.CDLL("libcnrt.so")

    def invoke(name, *args):
        status = getattr(lib, name)(*args)
        if status:
            raise RuntimeError(f"{name}: CNRT status {status}")

    marks = []
    try:
        for _ in range(2):
            ev = C.c_void_p()
            invoke("cnrtNotifierCreate", C.byref(ev))
            marks.append(ev)
        for _ in range(settings.warmup):
            if reset: reset()
            launch()
        samples = []
        for _ in range(settings.samples):
            if reset: reset()
            invoke("cnrtPlaceNotifier", marks[0], C.c_void_p(stream.mlu_stream))
            launch()
            invoke("cnrtPlaceNotifier", marks[1], C.c_void_p(stream.mlu_stream))
            stream.synchronize()
            us = C.c_float()
            invoke("cnrtNotifierDuration", *marks, C.byref(us))
            samples.append(us.value / 1000)
        return samples
    finally:
        for mark in marks:
            invoke("cnrtNotifierDestroy", mark)


def prepare(name, impl, case, inputs, env):
    if not env.device.startswith("mlu"):
        raise EnvironmentUnavailable(f"{impl} requires a local MLU")
    import torch, torch_mlu
    from .outputs import native_result

    stream = torch.mlu.current_stream(env.device)
    metadata = {}
    selection = None
    if impl == "intent:bangc":
        import intent
        from .intent import make_kernel
        from intent.examples.kernels.normalization.softmax import (
            stable_softmax,
            stable_softmax_f16,
        )
        from intent.examples.kernels.normalization.rms_norm import weighted_rms_norm

        definition = (
            (stable_softmax_f16 if inputs[0].dtype == np.float16 else stable_softmax)
            if name == "softmax"
            else weighted_rms_norm
        )
        kernel = make_kernel(definition, env, "bangc")
        buffers = intent.HostBuffers(
            "bangc", env.device, env.options.get("intent", {}).get("bangc", {})
        )
        try:
            args = tuple(
                buffers.input(v) if isinstance(v, np.ndarray) else v for v in inputs
            )
            call = kernel.prepare(*args)
            buffers.retain_outputs(call.result())
        except BaseException:
            buffers.close()
            kernel.close()
            raise

        def compile():
            call.compile()
            snapshot = call.observation
            rows = [asdict(c) for c in snapshot.candidates if c.status == "compiled"]
            if env.options.get("fresh_compile_cache"):
                from pathlib import Path
                import os, re

                by_entry = {dict(r["configuration"])["entry"]: r for r in rows}
                for trace in (Path(os.environ["INTENT_CACHE_DIR"]) / "bangc").glob(
                    "*/attempts/*/native_compilation.seconds"
                ):
                    src = (trace.parent / "kernel.mlu").read_text()
                    m = re.search(r'extern "C" int (\w+)\(', src)
                    if m and m.group(1) in by_entry:
                        by_entry[m.group(1)].update(
                            seconds=float(trace.read_text()), cache_hit=False
                        )
            metadata["cold_compile"] = dict(candidate_count=len(rows), rows=rows)

        def launch():
            call.enqueue(C.c_void_p(stream.mlu_stream))

        def result():
            return native_result(call.result())

        from .prepared import native_selection
        selection = lambda: native_selection(call, kernel)

        def close():
            kernel.close()
            buffers.close()

    else:
        args = tuple(
            torch.from_numpy(v).to(env.device) if isinstance(v, np.ndarray) else v
            for v in inputs
        )
        if impl == "cnnl":
            from .cnnl_rows import RowOperator

            operation = RowOperator(name, args, stream)
            launch, close, compile = operation.launch, operation.close, None
            result = lambda: operation.out.detach().cpu().numpy()
            metadata["cold_compile"] = dict(
                candidate_count=0, rows=[], library=operation.api
            )
        else:
            import intentdsl
            from intentdsl.examples import softmax, softmax_f32, rms_norm

            fn = (
                (softmax if inputs[0].dtype == np.float16 else softmax_f32)
                if name == "softmax"
                else rms_norm
            )
            spec = env.options.get("compilerinfra", {})
            cfg = spec.get("configuration")
            if impl.endswith(("-direct", "-double")):
                mode = "direct" if impl.endswith("-direct") else "double-buffer"
                from copy import deepcopy

                cfg = deepcopy(spec.get("configuration") or {})
                cfg.setdefault("nodes", {}).setdefault("dsa", {}).setdefault(
                    "decisions", {}
                )["levels/storage/dsa::supply"] = {
                    "strategies": ["levels/storage/dsa::" + mode],
                    "realizer": "levels/storage/dsa::stage-transfers",
                }
            kernel = intentdsl.jit(
                fn, backend="bangc", library=spec.get("library"), configuration=cfg
            )
            call = kernel.prepare(*args, artifact_file=spec.get("artifact_file"))

            def compile():
                call.compile()
                metadata["cold_compile"] = call.compilation_report()

            if "row_id" in spec:
                call = call.bind_row(spec["row_id"])
            selection = call.tuning_report
            launch, close = call.launch, None
            result = lambda: call.result().detach().cpu().numpy()
    return PreparedCall(
        launch,
        result,
        compile=compile,
        close=close,
        synchronize=stream.synchronize,
        native_measure=lambda s: timer(launch, stream, s),
        tuning_report=selection,
        timing="cnrt_notifier",
        device=env.device,
        metadata=metadata,
    )
