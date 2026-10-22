# IntentDSL Bench

- Build usable operator benchmarks; separate workloads, implementations, execution environments and measurement.
- A baseline is a selected implementation, not an oracle or a compiler target. Keep workloads which expose Intent expression, compilation, correctness or performance gaps.
- Do not modify compiler algorithms/lowering to make benchmark outcomes pass. Do not relax tolerances or silently replace implementations/timing methods.
- No hash, SHA or checksum based artifact/source verification. All run products belong to the configured output root: latest, daily, experiments, cache, build and work. Credentials remain outside source and output.
- Use apply_patch for semantic changes. Preserve unrelated work in sibling repositories. Do not push without explicit authorization.
- Existing upstream code retains its attribution. Historical results are observations, not current validation.
- Deliver reusable interfaces, implementations and plugin capabilities. Do not add test files or expand test runs unless the user requests them; use only necessary direct invocations to confirm the requested tool behavior.
- FlagGems is not an active baseline or oracle. Historical source and identifiers may be retained in the archive.
- Intent owns local target discovery, compilation, ABI binding and native launch. Bench adapters call intent.jit with an explicit backend; they never fill hardware facts or export from a controller to impersonate native JIT. Baseline adapters own their independent library bindings. Local/SSH transport belongs to execution.py; machine settings belong to external profiles. CLI, API, MCP and suites submit RunRequest through service.run.
- Author kernels and 30 complete programs are imported from the installed intent.examples package. Declaration queries do not compile or probe devices. Experiments keep explicit protocols and historical results; daily defaults do not rewrite paper timing or Agent first submissions.
- Default measurements cover prepared operator execution. GPU uses graph replay with device timing events; input reset/cache eviction stay outside the measured graph. Python/framework wrappers are lazy, opt-in through host_call, and are not normal benchmark output. Keep all algorithm-required kernels and state updates inside the operator.
