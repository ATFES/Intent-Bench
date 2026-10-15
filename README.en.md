# IntentDSL Bench

[Validation and known issues](docs/validation.md) · [Mutable GPU extension example](examples/custom_softmax.py)

An installable operator benchmark with one default metric: median latency.
Add `--throughput tflops` or `--throughput gbps` for one optional throughput metric.
Correctness is a status. Compilation details and raw samples remain in JSON.

```sh
python -m pip install -e .
intent-bench list
intent-bench run examples/custom.py:bench
intent-bench run softmax --impl torch,triton,cutile --device cuda:0
intent-bench run softmax --impl triton,intent:triton,intent:cutile --intent-root ../intentdsl --device cuda:0
intent-bench check --intent-root ../intentdsl --suite mainline --target triton
intent-bench compare runs/before runs/after
```

The core requires NumPy. Implementations load their SDKs only when selected.
PyTorch, native Mojo, GGML and CNNL baselines can run without Intent installed.
Python files export a `Workload` and `Implementation` objects; adding one does not
require changing the runner. See [the extension example](examples/custom.py).
`benchmark(workload, implementations, ...)` and the CLI use the same engine.

CUDA defaults to events around the complete call, with a 256 MiB or larger L2
eviction fill before each sample. Reset and cache eviction are outside the event
interval. `--cache warm` is explicit; compiler caches are not deleted. Warmup and
sampling counts are calibrated from 25 ms / 100 ms budgets, or overridden with
`--warmup` / `--samples`. Native workers time inside the target process, excluding
SSH and startup. Reports keep devices, timing methods and cache policies separate.
Numerical failures retain diagnostic latency but are not performance winners.

The 309 original registrations retain their production input profiles. New small
cases are named separately. The nine basic Mojo workloads are under `cpu/`, the
eleven formerly unchecked families under `advanced/`, and native remote examples
under `rvv/` and `mlu/`. Remote workers use the caller's SSH configuration and
explicit options such as `host`, `python`, `ggml_source`, `ggml_libraries`,
`weft_compiler` and `deployment`. Intent remote workers need Python 3.10+; all native
workers need NumPy. Missing toolchains and compilation failures remain results.

Agent generation and evaluation are separate commands. The historical single-shot
preset retains its import policy. Ordinary custom workloads have no experimental
import restrictions. Historical candidates and results live under `archive/`;
credentials, compiler outputs and runtime caches are excluded.

Run `pytest` for core checks. The regular CI builds and installs the package without
device SDKs; the manually triggered device workflow uses a configured self-hosted
runner. See [the Chinese guide](README.md) for API, remote and Agent examples,
and [source attribution](THIRD_PARTY.md) for vendored implementation licenses.
