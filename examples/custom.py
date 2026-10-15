"""A user workload: two-step mutable normalization versus an independent oracle."""
import numpy as np
from intent_bench import Case, Workload, callable_implementation


def two_steps(x, state):
    state += x.mean(axis=0)
    x -= state
    return x, state


def reference(case, inputs):
    x,state=inputs
    updated=state+x.mean(axis=0)
    return x-updated,updated


bench=Workload(
    "mutable_normalize",[Case("small",{"rows":16,"columns":32}),Case("wide",{"rows":32,"columns":1024})],
    inputs=lambda case,seed:(np.random.default_rng(seed).normal(size=(case.parameters["rows"],case.parameters["columns"])).astype("float32"),
                             np.zeros(case.parameters["columns"],dtype="float32")),
    implementations={"mine":callable_implementation("mine",two_steps,mutable=True)},
    oracle=reference,family="stateful",atol=1e-6,rtol=1e-6)
