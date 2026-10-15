# IntentDSL Bench

- Build usable operator benchmarks; separate workloads, implementations, execution environments and measurement.
- A baseline is a selected implementation, not an oracle or a compiler target. Keep workloads which expose Intent expression, compilation, correctness or performance gaps.
- Do not modify compiler algorithms/lowering to make benchmark outcomes pass. Do not relax tolerances or silently replace implementations/timing methods.
- No hash, SHA or checksum based artifact/source verification. Keep generated code, caches and credentials outside this repository.
- Use apply_patch for semantic changes. Preserve unrelated work in sibling repositories. Do not push without explicit authorization.
- Existing upstream code retains its attribution. Historical results are observations, not current validation.
- Deliver reusable interfaces, implementations and plugin capabilities. Do not add test files or expand test runs unless the user requests them; use only necessary direct invocations to confirm the requested tool behavior.
- FlagGems is not an active baseline or oracle. Historical source and identifiers may be retained in the archive.
- Backend implementations own computation and ABI binding. Local/SSH transport belongs to execution.py; machine settings belong to external environment profiles. CLI, API and MCP share the execution engine.
