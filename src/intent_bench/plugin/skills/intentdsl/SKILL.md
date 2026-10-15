---
name: intentdsl
description: Write and compile IntentDSL operators, benchmark user implementations against native baselines, and inspect latency across shapes and backends. Use for IntentDSL development or operator comparison through IntentDSL Bench.
---

Use the language, compiler and bench tools to complete the user's requested operator work.

For language/API questions use `intent_manual` search, api and read. For a supplied
program use `intent_compiler` describe, compile, materialize and read_artifact as
needed. The compiler owns lowering and native runtime; benchmark tasks do not
require changes to those components.

For an operator comparison:

- Discover its name and implementations with `intent_bench.list_workloads` and
  `describe_workload`. A new user Python file exports a `Workload` as `bench`;
  pass its absolute `file.py:bench` path without editing the runner.
- Discover saved machines with `list_environments`. Select the implementation
  separately: `triton` and `cutile` are native implementations;
  `intent:triton` and `intent:cutile` compile Intent to the named backend.
  `mojo`, `ggml` and `cnnl` are independent implementations.
- Call `run_benchmark` with the requested implementations and environment.
  Scalar parameters override a case; list-valued parameters request a sweep.
  A comparison baseline is optional and independent of the numerical oracle.
- The call returns immediately. Use `run_status` to retrieve its result; do not
  start another run merely to obtain progress. Use `compare_runs` for before/after.
- Present latency and, only when requested and defined, one throughput metric.
  Include actual implementation names, shapes and the useful result/report link.
  Inspect the returned error and artifact path when a requested implementation fails.

Use `check_mainline` only when the user wants the mainline authored programs.
It is not the workload catalog. Candidate re-evaluation uses existing code;
calling a model to generate new code is a separate user action.

Environment profiles live in `~/.config/intent-bench/environments/*.json` or the
directory set by `INTENT_BENCH_ENVIRONMENTS`. They select Python, device, host and
toolchain paths. Keep SSH aliases/credentials in the user's SSH configuration.
Do not put machine addresses into workload definitions or compiler targets.

When MCP tools are not connected, the same capabilities are available through
`intent-bench list`, `describe`, `env list`, `run`, `check`, `compare` and `report`.
Read the command's `--help` for its parameters. Do not turn a request to build or
use the tool into an unrelated test-suite or full-catalog benchmark run.
