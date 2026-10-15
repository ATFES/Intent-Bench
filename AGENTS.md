# IntentDSL Bench

- Build usable operator benchmarks; separate workloads, implementations, execution environments and measurement.
- A baseline is a selected implementation, not an oracle or a compiler target. Keep workloads which expose Intent expression, compilation, correctness or performance gaps.
- Do not modify compiler algorithms/lowering to make benchmark outcomes pass. Do not relax tolerances or silently replace implementations/timing methods.
- No hash, SHA or checksum based artifact/source verification. Keep generated code, caches and credentials outside this repository.
- Use apply_patch for semantic changes. Preserve unrelated work in sibling repositories. Do not push without explicit authorization.
- Existing upstream code retains its attribution. Historical results are observations, not current validation.
