from types import SimpleNamespace
import pytest

from intent_bench import Measurement, PreparedCall
from intent_bench.measurement import _measure_python, measure
from intent_bench.results import compare, table


def test_cuda_reset_and_flush_are_outside_each_event_interval():
    log = []
    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            log.append("event")
        def elapsed_time(self, end):
            return 0.25
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event))
    call = PreparedCall(lambda: log.append("run"), lambda: None,
                        reset=lambda: log.append("reset"), synchronize=lambda: log.append("sync"))
    samples = _measure_python(call, Measurement("cuda_event", 2, 1), lambda: log.append("flush"), torch)
    assert samples == [0.25, 0.25]
    assert log == ["reset", "flush", "run", "sync",
                   "reset", "flush", "event", "run", "event",
                   "reset", "flush", "event", "run", "event", "sync"]


def test_native_calibration_stays_in_native_worker():
    counts = []
    def native(settings):
        counts.append((settings.warmup, settings.samples))
        return [2.0] * settings.samples
    def forbidden():
        raise AssertionError("do not time transport by calling run")
    call = PreparedCall(forbidden, lambda: None, native_measure=native, timing="cnrt_notifier")
    samples, method = measure(call, Measurement(warmup_ms=4, rep_ms=10))
    assert counts == [(0, 1), (1, 5), (2, 5)] and len(samples) == 5 and method == "cnrt_notifier"


def test_native_sample_budget_includes_eviction_without_reporting_it(monkeypatch):
    ticks = iter((0, 50_000_000))
    monkeypatch.setattr("intent_bench.measurement.time.perf_counter_ns", lambda: next(ticks))
    counts = []
    def native(settings):
        counts.append((settings.warmup, settings.samples))
        return [0.001] * settings.samples
    call = PreparedCall(lambda: None, lambda: None, native_measure=native)
    samples, _ = measure(call, Measurement(warmup_ms=25, rep_ms=100))
    assert counts == [(0, 1), (1, 5), (3, 10)]
    assert samples == [0.001] * 10
    assert call.metadata['sampling_cycle_ms'] == 10


def test_explicit_native_counts_remain_exact():
    counts = []
    def native(settings):
        counts.append((settings.warmup, settings.samples))
        return [0.001] * settings.samples
    samples, _ = measure(PreparedCall(lambda: None, lambda: None, native_measure=native),
                         Measurement(samples=11, warmup=2))
    assert counts == [(2, 11)] and samples == [0.001] * 11


def test_slow_native_kernel_uses_requested_budget_after_pilot():
    counts = []
    def native(settings):
        counts.append((settings.warmup, settings.samples))
        return [23000.0] * settings.samples
    samples, _ = measure(PreparedCall(lambda: None, lambda: None, native_measure=native),
                         Measurement(samples=3))
    assert counts == [(0, 1), (1, 3)]
    assert samples == [23000.0] * 3


def test_operator_does_not_fall_back_to_python_cpu_timing():
    def forbidden():
        raise AssertionError("unsupported timing must fail before calling Python")
    call = PreparedCall(forbidden, lambda: None)
    with pytest.raises(NotImplementedError, match="native timer"):
        measure(call, Measurement())
    with pytest.raises(ValueError, match="host_call"):
        measure(call, Measurement(method="host"))


def test_default_output_is_latency_and_optional_single_throughput():
    row = dict(workload="add", case="large", implementation="mine", status="passed", correctness="passed",
               device="cuda:0", method="cuda_event", scope="prepared_call", median_ms=0.25, parameters={},
               metrics={}, metadata={"cache_policy": "flush_l2"})
    output = table([row])
    assert "latency ms" in output and "speedup" not in output and "throughput" not in output
    row["metrics"] = {"gbps": 128}
    assert "128 GB/s" in table([row])
    warm = {**row, "metadata": {"cache_policy": "warm"}}
    assert {r["change"] for r in compare([row], [warm])} == {"added", "removed"}
