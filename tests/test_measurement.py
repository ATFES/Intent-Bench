from types import SimpleNamespace

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
    assert counts == [(1, 5), (2, 5)] and len(samples) == 5 and method == "cnrt_notifier"


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
