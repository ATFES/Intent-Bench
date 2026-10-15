"""Full-call latency; cache eviction and input restoration are outside the timer."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import math
import os
from pathlib import Path
import re
import statistics
import time

from .model import Measurement, PreparedCall


@contextmanager
def device_window(device):
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "intent-bench/locks"
    root.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", device)
    with (root / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def reset(call):
    if call.reset is not None:
        call.reset()


def synchronize(call):
    if call.synchronize:
        call.synchronize()
    elif call.device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(call.device)


def method_for(call, settings):
    if settings.method != "auto":
        return settings.method
    if settings.scope == "host_call":
        return "host"
    if call.native_measure is not None:
        return "native"
    return "cuda_event" if call.device.startswith("cuda") else "host"


def _counts(settings, estimate_ms):
    # Bound event allocation for very short kernels. Explicit counts stay exact.
    estimate_ms = max(estimate_ms, 0.001)
    warmup = settings.warmup if settings.warmup is not None else min(10000, max(1, math.ceil(settings.warmup_ms / estimate_ms)))
    samples = settings.samples if settings.samples is not None else min(10000, max(1, math.ceil(settings.rep_ms / estimate_ms)))
    return warmup, samples


def _cuda_flush(torch, call, settings):
    if settings.cache == "warm":
        call.metadata["cache_policy"] = "warm"
        return lambda: None
    # TileLang uses a 256 MiB int32 fill. Exceed the L2 capacity as well when
    # PyTorch exposes a larger device cache. Allocation is outside measurement.
    properties = torch.cuda.get_device_properties(call.device)
    size = max(256 * 1024 * 1024, 2 * getattr(properties, "L2_cache_size", 0))
    cache = torch.empty((size + 3) // 4, dtype=torch.int32, device=call.device)
    call.metadata.update(cache_policy="flush_l2", cache_bytes=cache.numel() * 4)
    return cache.zero_


def measure(call: PreparedCall, settings: Measurement):
    method = method_for(call, settings)
    if method == "cuda_graph" and call.reset is not None:
        raise ValueError("CUDA Graph replay cannot restore mutable inputs outside the graph; use cuda_event")
    if method.startswith("cuda") and not call.device.startswith("cuda"):
        raise ValueError(f"{method} requires a CUDA implementation")
    if method == "native":
        if call.native_measure is None:
            raise ValueError("implementation does not supply native timing")
        estimate = 1.0
        if settings.samples is None or settings.warmup is None:
            calibration = list(call.native_measure(replace(settings, samples=5, warmup=1)))
            if not calibration or any(not math.isfinite(x) or x <= 0 for x in calibration):
                raise ValueError("native calibration requires finite positive latency samples")
            estimate = statistics.median(calibration)
        warmup, samples = _counts(settings, estimate)
        call.metadata.setdefault('cache_policy','native_default')
        call.metadata.update(warmup_iterations=warmup, sample_count=samples)
        return list(call.native_measure(replace(settings, samples=samples, warmup=warmup))), call.timing or "native"
    if call.device.startswith("cuda"):
        import torch
        with torch.cuda.device(call.device):
            flush = _cuda_flush(torch, call, settings)
            if method == "cuda_graph":
                # Initialize before capture; one full call per replay. The L2
                # flush is outside the graph and outside its event interval.
                call.run()
                synchronize(call)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    call.run()
                replay = replace(call, run=graph.replay)
                values = _measure_python(replay, replace(settings, method="cuda_event"), flush, torch)
            else:
                values = _measure_python(call, replace(settings, method=method), flush, torch)
        return values, method
    call.metadata["cache_policy"] = "warm"
    return _measure_python(call, replace(settings, method=method), lambda: None), method


def _measure_python(call, settings, flush, torch=None):
    def before():
        reset(call)
        flush()

    if settings.method == "cuda_event":
        def batch(count):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
            for begin, end in zip(starts, ends):
                before()
                begin.record()
                call.run()
                end.record()
            synchronize(call)
            return [begin.elapsed_time(end) for begin, end in zip(starts, ends)]
    else:
        def batch(count):
            values = []
            for _ in range(count):
                before()
                synchronize(call)
                start = time.perf_counter_ns()
                call.run()
                synchronize(call)
                values.append((time.perf_counter_ns() - start) / 1e6)
            return values
    estimate = 1.0
    if settings.samples is None or settings.warmup is None:
        # Count reset/flush overhead when sizing the budget, while keeping it
        # outside every reported latency interval.
        before()
        call.run()
        synchronize(call)
        started = time.perf_counter_ns()
        batch(5)
        estimate = (time.perf_counter_ns() - started) / 5e6
    warmup, samples = _counts(settings, estimate)
    for _ in range(warmup):
        before()
        call.run()
    synchronize(call)
    call.metadata.update(warmup_iterations=warmup, sample_count=samples)
    return batch(samples)
