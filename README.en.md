# IntentDSL Bench

Bench 0.3.0 compares the dispersion of complete calls. For IntentDSL 2.1.0, 244 matched calls were faster, 85 tied and 7 were slower across five exits. [Scoreboard](https://github.com/Tianchen-ai/Intent/blob/main/doc/plan/v0.6-semantics.md)

Independent operator correctness and performance comparisons. [中文](README.md) · [Attribution](THIRD_PARTY.md)

Intent owns local JIT, target discovery, compilation, ABI binding and execution.
Bench owns inputs, implementations, oracles, measurement and SSH transport.
V2 programs live in the installed `intentdsl.examples` package. `intent:*` selects the historical V1 implementation.

## Use

Start in this checkout, using the GPU environment from the [IntentDSL source installation](https://github.com/Tianchen-ai/Intent#build-from-source).

```sh
python -m pip install -e .
export INTENT_BENCH_OUTPUT_ROOT="$PWD/output"
intent-bench describe softmax
intent-bench run softmax --env examples/environments/gpu.json --output-root "$PWD/output" --case 64x1024-float16 --impl intent-v2:triton,triton --baseline triton --check-only
intent-bench run softmax --env examples/environments/gpu.json --output-root "$PWD/output" --case 64x1024-float16 --impl intent-v2:triton,triton --baseline triton
```

Alternatively set output_root in ~/.config/intent-bench/config.json. Implementations
are explicit. A baseline must be selected and is independent of the numerical oracle.
Both Intent GPU backends can share one native baseline. Torch and framework wrappers
are used only when selected.

list/describe query declarations without compilation or device startup. run performs
numerical checks and timing. check uses the installed author programs at source/native/run
stages; executed alone is not numerical validation. experiment selects an explicit paper
or Agent protocol. compare/report read existing results.

## Execution and output

Copy a template from `examples/environments/` to the configuration directory or to the ignored `local-config/` directory and pass its path to `--env`.
For archived Weft runs, copy the profiles in `archive/legacy-tools/cpu/providers/weft/` to `~/.config/intent-bench/legacy-weft/` and set the SSH destination and toolchain, or set `INTENT_WEFT_PROFILE` to your deployment file.

Profiles in ~/.config/intent-bench/environments select transport, host, python, device,
threads and toolchain paths. SSH profiles also require remote_output_root. The selected
Python must already contain Bench, Intent when selected, and provider dependencies.
The worker executes ordinary local JIT on that host. It never substitutes controller-side
generation or deployment. Explicit compiler paths belong to the execution host.
intent_root verifies an installed editable checkout without modifying imports.
CPU cpus selects the execution set; threads is a budget; Intent discovers ISA/VLEN.
MLU uses device=mlu:0 and NeuWare paths in options.

CLI, Python API, MCP and suites share RunRequest and the execution service. Requests have
timeouts and preserve failure stages/logs. Overlapping device ownership reports skipped_busy.
Check external device/process load before accepting performance measurements.

Use `--oracle-env local-cpu` when the execution host lacks the registered oracle's
dependencies. Intent still compiles and executes locally on the selected target.
Bench transfers actual inputs, initial state and outputs, preserving dtype, strides
and shared storage. The oracle environment runs the original oracle and checker;
it does not recreate inputs from a seed or supply a cross-device timing baseline.
Python's `RunRequest.oracle_environment` and the MCP parameter use the same service.

For a remote target, pass its environment file and an oracle environment to `run --check-only`.

```text
output/
  latest/<environment>/<workload>/
  daily/YYYY-MM-DD/<environment>/
  experiments/<experiment>/<run>/
  cache/<environment>/
  build/
  work/
```

A new failure replaces the previous latest result. Shared compile caches remain separate.
The service controls provider caches and temporary paths, and removes worker staging after
result retrieval. Historical archive contents are never replaced with current results.

## Measurement

Default scope=operator measures prepared computation. CUDA replays the complete operator
with timing events inside the graph. Input restoration, L2 eviction and Python submission
are outside. Every required kernel and state update remains inside. Cache eviction uses
at least 256 MiB or a larger-than-L2 allocation; cache=warm is explicit. Capture failure
does not silently change methods; timing=cuda_event is an explicit alternative.

CPU native calls bind and allocate first, then time all kernels in one native C ABI interval.
The sequence timer needs CXX or c++. MLU measures the complete queue using CNRT notifiers.
Pure Python/eager CPU implementations are accurately labeled host timing.
Only scope=host_call enables optional Python/framework wrappers.

Compilation, initial invocation/tuning and warm measurements are separate. Default budgets
are 25 ms warmup and 100 ms sampling; samples/warmup override counts. JSON retains raw
milliseconds and diagnostics; CSV/HTML are summaries. Speedup is baseline_ms/implementation_ms
only for numerically valid, matching devices, scopes, boundaries and cache policies.
With three or more samples on each side, the comparison spans candidate Q1 / baseline Q3 through candidate Q3 / baseline Q1. An interval wholly below 0.95 or above 1.05 counts as faster or slower. JSON and CSV retain the interval and the interquartile range. For repeated measurements, `intent_bench.comparison.compare_latency` takes matched rounds and requires agreement across all rounds.

## Python and custom implementations

```python
from intent_bench import RunRequest, run, load_environment
outcome = run(RunRequest(
    "softmax", ("intent-v2:triton", "triton"),
    environment=load_environment("examples/environments/gpu.json"),
    cases=("64x1024-float16",), baseline="triton",
))
```

A user file exports Workload. Implementation.prepare returns PreparedCall with launch,
result and optional reset, compile, synchronize, close and native_measure. Prepare binds
and allocates; compile builds; launch includes all required kernels. See
[mutable CPU](examples/custom.py) and [GPU composition](examples/custom_softmax.py) examples.
Run the Python callable example with `intent-bench run examples/custom.py:bench --impl mine --scope host_call`.

## Experiments and plugins

experiment paper --protocol protocol.json consumes a name and complete RunRequest records
in requests. Each declares method, scope, cache, samples and warmup; daily defaults do not
rewrite the protocol. Results go to experiments; historical paper tables remain intact.
experiment agent generate and evaluate remain separate; first submissions are never revised
using test feedback. Agent outputs belong under output/experiments.

intent-bench plugin --output /path/to/output/build/plugin exports the installed plugin.
Its manual/compiler MCP servers start the Intent package directly. Bench MCP submits the
same requests as CLI/API and exposes background status. See [plugin installation](docs/codex-plugin.md).
