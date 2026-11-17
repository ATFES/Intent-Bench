---
name: intentdsl
description: Use IntentDSL Bench to compare operators, user implementations and compiler backends, choose input cases and local or SSH environments, and inspect latency or throughput. Use for IntentDSL development, GPU PK, baseline comparisons and parameter sweeps; also provides Intent language and compiler tools.
---

IntentDSL Bench serves IntentDSL development through a reusable CLI, Python API
and MCP interface. Use the existing tool for operator comparisons and extensions.

For language/API questions use `intent_manual` search, api and read. For a supplied
program use `intent_compiler` describe, compile, materialize and read_artifact as
needed. The compiler owns lowering and native runtime; benchmark tasks do not
require changes to those components.

## Choose a comparison

- Discover the relevant operator with `intent_bench.list_workloads(query=...)`,
  then `describe_workload`. Its `implementations`, `implementation_details`,
  `cases` and `parameter_keys` describe the choices for that particular workload.
  Match the mathematical operation, dtype, layout and state behavior; similar
  names can describe different computations.
- Select participants explicitly. `intent:triton` and `intent:cutile` are two
  compilations of Intent; `triton` and `cutile` are native implementations.
  `mojo`, `ggml`, `cnnl` and `torch` are also independent implementations.
  A `reference:*` entry is a numerical reference on another execution stack,
  not a same-device performance opponent.
- Several Intent backends may share ONE selected baseline. For example, a request
  to compare both Intent GPU backends against native Triton becomes
  `implementations=["intent:triton", "intent:cutile", "triton"], baseline="triton"`.
  There is no automatic backend-to-baseline pairing. Do not add a native cuTile
  participant just because Intent also targets cuTile. If the user requests more
  native participants, include them explicitly.
- For "GPU 都比一下" or "GPU PK", use context to identify the operator and intended
  Intent backends. Choose one suitable available baseline when the user leaves
  that choice to you, using the workload's semantics and prior comparison context;
  state the choice briefly and proceed. Ask only if a real ambiguity changes the
  requested comparison. Never invent an implementation absent from the workload.
- A baseline must be in the participant list. It is independent of the numerical
  oracle; a single implementation or baseline-only run is also valid. FlagGems is
  not an active baseline; historical identifiers may still contain its name.

## Choose inputs and execution

- Preserve supplied shapes, dtype and layouts. For a quick look, choose an existing
  small case; for scaling, start from one case and override the exposed parameters.
  Scalar parameters override values, while list values request a Cartesian sweep.
  Do not multiply every preset by a sweep unless the user wants that combination.
- `parameter_keys=[]` means the retained input profile is fixed. Use a suitable
  parameterized workload or a custom file for different data. `null` means there
  is no declared key whitelist; use the case parameters or the provider's API,
  not guessed fields. Small presets and original production inputs serve different
  requests. Do not substitute one for the other without saying so.
- Discover saved environments with `list_environments`; preserve an environment
  already selected by the user. The implementation chooses computation; the
  environment chooses Python, device, host and toolchains. SSH is transport,
  not a compiler backend. Avoid copying machine settings into operator code.
- Use the default `scope="operator"`: prepared complete computation, with CUDA
  Graph replay and timing event nodes inside the graph, or native CPU/worker timing.
  Reset and cache flushing are outside the measured graph; mutable inputs are restored
  in their existing buffers. All algorithm-required kernels and state updates remain
  inside. Graph submission is outside the device timing events.
- Only use `scope="host_call"` when the user explicitly asks to measure or optimize
  Python/API/framework overhead. It selects the optional host entry and wall-clock
  timing; ordinary comparisons neither construct extra framework wrappers nor add
  host measurements. `torch.compile(fullgraph=True)` does not itself imply CUDA Graph
  replay. Do not propose Python micro-optimizations from operator timing alone.
- `timing="cuda_event"` explicitly chooses ordinary stream events; capture failures
  do not silently fall back to them. `cache="warm"` selects warm-cache runs. Clearing
  cache never means deleting compiler caches. SSH/startup are outside operator timing.
- If the execution host lacks oracle dependencies, select `oracle_environment`
  (CLI `--oracle-env`) explicitly. Bench transfers actual inputs, initial state and
  outputs with dtype/stride/alias information, then runs the existing oracle and
  checker there. Intent JIT and performance baselines remain on the execution host.
  Never count this remote oracle as a same-device performance opponent.

## Run and explain

- Call `run_benchmark` with explicit participants, baseline, selected cases,
  parameter overrides and environment. It uses the same CLI and engine as humans.
- The call returns immediately. Use `run_status` to retrieve its result; do not
  start another run merely to obtain progress. Use `compare_runs` for before/after.
- Present latency and, only when requested and defined, one throughput metric.
  Include actual implementation names, shapes and the useful result/report link.
  Inspect the returned error and artifact path when a requested implementation fails.
- Speedup means `baseline_ms / implementation_ms`. Keep incompatible devices,
  timing scopes and cache policies separate. Numerical failures can carry diagnostic
  latency; describe that status rather than treating them as performance wins.
- Old `prepared_call` records may include framework integration. New `operator`
  records have a separate measurement contract. A method/scope change is not a
  compiler performance improvement. Do not subtract graph time from host time and
  present the difference as an exact runtime-overhead measurement.

For exact MCP/CLI examples, see [comparisons](references/comparisons.md).
To add user code or custom inputs, read [extensions](references/extensions.md).
Extend a workload in the user's file and pass its absolute `file.py:bench` path;
there is no need to modify the runner or create an experiment-specific executor.

Use `check_mainline` only when the user wants the mainline authored programs.
It is not the workload catalog. Candidate re-evaluation uses existing code;
calling a model to generate new code is a separate user action.

Environment profiles live in `~/.config/intent-bench/environments/*.json` or the
directory set by `INTENT_BENCH_ENVIRONMENTS`. They select Python, device, host and
toolchain paths. Both Bench and Intent must be installed in the execution Python.
Remote runs start the ordinary local JIT there; never substitute controller-side
generation/export or hand-filled ISA, VLEN, memory limits or shape specialization.
Keep SSH aliases/credentials in the user's SSH configuration.
Do not put machine addresses into workload definitions or compiler targets.

All runs use the configured output root (INTENT_BENCH_OUTPUT_ROOT or config.json).
Latest results are grouped by environment/workload; experiments and daily results
are separate. The service redirects compiler/provider caches and temporary files.
SSH profiles require remote_output_root. Do not add output locations in the Intent
checkout or ad-hoc /tmp trees. check_mainline selects its backend explicitly.

When MCP tools are not connected, the same capabilities are available through
`intent-bench list`, `describe`, `env list`, `run`, `check`, `compare` and `report`.
The plugin's MCP configuration records its absolute Python interpreter. It can
also run `-m intent_bench.cli` if `intent-bench` is absent from PATH. Read the
command's `--help` for parameters. Do not turn a request to build or use the tool
into added test files, an unrelated test suite or a full-catalog benchmark run.
