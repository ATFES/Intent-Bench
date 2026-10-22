# Comparison recipes

Names below are examples for the built-in `softmax`. Discover the actual workload
and saved environment before running; an environment name is local configuration.
Always respect a user's explicitly chosen implementations.

## Both Intent GPU backends, one native baseline

Call `run_benchmark` with:

```json
{
  "workload": "softmax",
  "implementations": ["intent:triton", "intent:cutile", "triton"],
  "baseline": "triton",
  "environment": "local-gpu",
  "cases": ["64x1024-float16"]
}
```

The equivalent CLI is:

```sh
intent-bench run softmax --impl intent:triton,intent:cutile,triton \
  --baseline triton --env local-gpu --case 64x1024-float16
```

With a cuTile baseline, replace the native `triton` entry and `baseline` with
`cutile`; retain both Intent entries. For a baseline-only run, select only the
native implementation. Neither operation requires changing the bench runner.

## Input scaling

Keep the same participants and one starting case, then pass:

```json
{"parameters": {"rows": 64, "columns": [128, 1024, 4096]}}
```

CLI: `--case 64x1024-float16 --param rows=64 --param 'columns=[128,1024,4096]'`.
Add `throughput="gbps"` only when useful and supported by the workload's work
formula. Omit it for latency alone. Use the supplied production case when the
user wants production performance; do not shrink it to the example above.

## Existing results and compiler builds

`run_status(run=...)` reads an existing job. `compare_runs(before=..., after=...)`
compares saved runs without executing operators. To compare compiler builds,
make two explicit runs with the same implementations, cases, environment and
measurement policy, changing `compiler`, then compare their result directories.
`check_mainline` is for authored mainline programs, not the general operator catalog.

## Optional Python/framework overhead

Ordinary requests use `scope="operator"` and CUDA Graph device timing by default.
Only when asked about Python or framework invocation, add `scope="host_call"`
to the same MCP request, or `--scope host_call` to the CLI. This uses a host timer
and waits for completion. BatchNorm, RoPE and AdamW construct their `torch.compile`
integration only in this mode. Keep these results separate from operator latency.
Explicit `timing="cuda_event"` measures a stream-event interval, which can include
GPU waits for host submission. Graph timing event nodes exclude graph submission.

## Environment scope

Use `list_environments` to find names, devices and interpreter/toolchain paths.
Profiles live under `~/.config/intent-bench/environments/` (or
`INTENT_BENCH_ENVIRONMENTS`). A local or SSH profile is independent of the selected
Intent target. Native workers time inside their target process. Ordinary remote
Python runs need dependencies and custom imported modules in that interpreter.
The CLI accepts an explicit profile JSON path as well as a saved name.
