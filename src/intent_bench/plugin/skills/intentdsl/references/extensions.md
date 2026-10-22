# Add an implementation or inputs in a user file

Use the installed `intent_bench` API. A file exports `bench`, a `Workload`.
Its `inputs(case, seed)` constructs the public inputs; implementation preparation
receives those inputs and the selected environment. Preserve their values and
the operation's semantics when converting to a device representation.

## Reuse a workload and its native baseline

This example adds a user Torch softmax to the existing workload. Replace the
body of `run` with the user's implementation. Preparation allocates inputs once.

```python
from dataclasses import replace
from intent_bench import Implementation, PreparedCall, load_workload

def prepare(case, inputs, environment):
    import torch
    device = "cuda:0" if environment.device == "auto" else environment.device
    x = torch.as_tensor(inputs[0], device=device)
    state = {}
    def run():
        state["y"] = torch.softmax(x, dim=-1)
    return PreparedCall(run, lambda: state["y"], device=str(x.device))

base = load_workload("softmax")
bench = replace(base, implementations={
    **base.implementations,
    "mine": Implementation("mine", prepare),
})
```

Pass `/absolute/path/my_bench.py:bench` as `workload`, select `mine` and `triton`,
and set `baseline="triton"`. Add Intent implementations only when requested.

## Custom data and state

- Replace `base.cases` with `Case(name, parameters)` entries and/or `base.inputs`
  with a custom input function. Keep the input structure expected by the selected
  implementations. Arbitrary new semantics use a new `Workload` and its own
  implementations, rather than silently reusing incompatible baselines.
- For a simple callable, `callable_implementation(name, fn, mutable=False, ...)`
  expands tuple inputs as positional arguments or dictionaries as keyword arguments.
- For device code use `Implementation(name, prepare)`. Return a `PreparedCall`
  with `launch`, `result` and, when needed, `reset`, `compile`, `synchronize` and `close`.
  `reset` restores mutable initial data outside timing. The input buffers must stay
  alive until all queued work is done. `result` observes outputs, including updated
  state when that is part of the workload's output contract.
- GPU `launch` must be capture-compatible for default graph timing. Keep stable buffer
  addresses; supply `reset` to restore initial values outside the graph. Add optional
  `host_run` and `host_result` only for the Python API/framework path; construct costly
  wrappers lazily inside `host_run`, which is never invoked in normal operator mode.
- Put all kernels and algorithm-required intermediate work in `launch`; compilation
  and one-time setup belong in preparation. A native worker provides
  `native_measure(settings)` returning millisecond samples measured in that worker.
- An oracle is separate from the performance baseline. Custom numerical checks use
  `check(actual, expected)`; custom state/output structures should match that contract.

The Python API `run(RunRequest("/path/to/file.py:bench", ("mine", "triton"),
environment=env, baseline="triton"))` uses the same worker, local/SSH transport and
output handling as CLI/MCP. Import run and RunRequest from intent_bench. User files
have no experimental candidate-import restrictions.
