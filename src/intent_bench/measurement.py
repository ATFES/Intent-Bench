"""Prepared operator latency; Python entry-to-completion timing is opt-in."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import gc
import math
import os
from pathlib import Path
import re
import statistics
import time

from .model import Measurement, PreparedCall


@contextmanager
def device_window(device):
    from .output import output_root
    from .model import DeviceBusy
    root = output_root()/"work"/"locks"
    root.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", device)
    with (root / (name + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DeviceBusy(f"another Bench invocation owns device {device}") from None
        # SDK dispatchers can own CUDA libraries through Python cycles. Their
        # destructors must not unload a library inside another provider's graph
        # capture (including capture performed by its first-call autotuner).
        collecting = device.startswith("cuda") and gc.isenabled()
        if collecting:
            gc.collect()
            gc.disable()
        try:
            yield
        finally:
            if collecting:
                gc.enable()


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
    if call.device.startswith("cuda"):
        return "cuda_graph"
    if call.device != "cpu":
        raise NotImplementedError(f"{call.device} requires a registered device or native timing implementation")
    return "host"


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
    if call.device != "cpu" and method == "host" and settings.scope != "host_call":
        raise ValueError("Python wall-clock timing requires --scope host_call")
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
                values = _measure_graph(call, settings, flush, torch)
            else:
                call.metadata['timing_boundary'] = 'python_entry_to_completion' if method == 'host' else 'stream_events'
                values = _measure_python(call, replace(settings, method=method), flush, torch)
        return values, method
    call.metadata["cache_policy"] = "warm"
    return _measure_python(call, replace(settings, method=method), lambda: None), method


def _measure_graph(call, settings, flush, torch):
    """One complete invocation per replay, with real timing event nodes in the graph.

    Restore existing buffers before replay, never inside the measured graph. Event
    nodes exclude Python's graph submission and reset/eviction from device latency.
    Read each event pair before the next replay overwrites its timestamps.
    """
    stream = torch.cuda.Stream(device=call.device)
    stream.wait_stream(torch.cuda.current_stream(call.device))
    with torch.cuda.stream(stream):
        reset(call)
        call.launch()
    stream.synchronize()
    begin = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.stream(stream):
            reset(call)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            begin.record()
            call.launch()
            end.record()
    except Exception as error:
        raise RuntimeError("CUDA Graph capture failed; use a capture-compatible prepared operator, "
                           "or explicitly select --timing cuda_event for stream-event timing. "
                           f"{type(error).__name__}: {error}") from error

    call.metadata.update(timing_boundary='graph_event_nodes', graph_calls_per_replay=1)
    def replay():
        reset(call)
        flush()
        graph.replay()
        end.synchronize()

    # Graph upload and the first replay are preparation, not reported samples.
    replay()
    estimate = 1.0
    if settings.samples is None or settings.warmup is None:
        started = time.perf_counter_ns()
        for _ in range(5):
            replay()
        estimate = (time.perf_counter_ns() - started) / 5e6
    warmup, samples = _counts(settings, estimate)
    for _ in range(warmup):
        replay()
    values = []
    for _ in range(samples):
        replay()
        values.append(begin.elapsed_time(end))
    call.metadata.update(warmup_iterations=warmup, sample_count=samples)
    return values


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
                call.launch()
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
                call.launch()
                synchronize(call)
                values.append((time.perf_counter_ns() - start) / 1e6)
            return values
    estimate = 1.0
    if settings.samples is None or settings.warmup is None:
        # Count reset/flush overhead when sizing the budget, while keeping it
        # outside every reported latency interval.
        before()
        call.launch()
        synchronize(call)
        started = time.perf_counter_ns()
        batch(5)
        estimate = (time.perf_counter_ns() - started) / 5e6
    warmup, samples = _counts(settings, estimate)
    for _ in range(warmup):
        before()
        call.launch()
    synchronize(call)
    call.metadata.update(warmup_iterations=warmup, sample_count=samples)
    return batch(samples)
