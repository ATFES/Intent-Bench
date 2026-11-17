"""Agent tools backed by the public bench discovery, CLI and result APIs."""
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from . import discovery, jobs, results
from .environments import list_environments as environments

server=FastMCP('IntentDSL Bench')


@server.tool()
def list_workloads(query: str = '', family: str = '') -> list[dict]:
    """Find operator workloads and implementation names without importing SDKs."""
    return [row for row in discovery.list_workloads()
            if query.lower() in row['name'].lower() and (not family or family in row['family'])]


@server.tool()
def describe_workload(workload: str) -> dict:
    """Describe implementation roles, named cases and parameter keys for an operator or absolute file.py:bench.

    Empty parameter_keys means a fixed original profile; null means no declared key
    whitelist. Use actual case parameters for sweeps. Does not prepare or run operators.
    """
    return discovery.describe_workload(workload)


@server.tool()
def list_environments() -> list[dict]:
    """List saved execution environments: host, Python, device and toolchain settings."""
    return environments()


@server.tool()
def run_benchmark(workload: str, implementations: list[str], environment: str | None = None,
                  cases: list[str] | None = None, parameters: dict | None = None,
                  baseline: str | None = None, check_only: bool = False,
                  throughput: str | None = None, cache: str = 'flush',
                  device: str | None = None, intent_root: str | None = None,
                  compiler: str | None = None, options: dict | None = None,
                  timing: str = 'auto', scope: str = 'operator',
                  oracle_environment: str | None = None) -> dict:
    """Start an operator comparison using the ordinary bench CLI. Returns a run path immediately.

    Choose participants explicitly. Both Intent GPU backends can share one baseline:
    implementations=["intent:triton", "intent:cutile", "triton"], baseline="triton".
    The baseline must be a participant; no default pairing or extra opponent is added.
    Environment selects execution settings independently. Select one named case before
    a parameter sweep; list parameter values form a Cartesian product. CUDA defaults to
    graph replay with device timing event nodes and cache flushing outside the graph.
    scope="host_call" explicitly measures Python/framework entry-to-completion time;
    use only when requested. Use run_status to retrieve the result without rerunning.
    A custom workload file executes user Python code just as the CLI does.
    oracle_environment runs only the existing correctness check on actual transferred
    inputs and results. Performance baselines remain on the execution device.
    """
    if not implementations:raise ValueError('select at least one implementation')
    if baseline is not None and baseline not in implementations:
        raise ValueError('baseline must be included in implementations')
    from .model import Measurement
    from .request import RunRequest
    from .environments import load_environment
    env = load_environment(environment, device=device, intent_root=Path(intent_root) if intent_root else None,
                           compiler=compiler, options=options)
    measurement = Measurement(method=timing, scope=scope, cache=cache, throughput=throughput,
                              enabled=not check_only)
    return jobs.start(RunRequest(workload, tuple(implementations), env, measurement,
                                baseline, tuple(cases or ()), parameters or {},
                                oracle_environment=load_environment(oracle_environment) if oracle_environment else None))


@server.tool()
def run_status(run: str) -> dict:
    """Read a background run's progress and operator results. Does not rerun anything."""
    return jobs.status(run)


@server.tool()
def compare_runs(before: str, after: str) -> list[dict]:
    """Compare saved result directories, preserving device and timing distinctions."""
    def resolve(value):
        path=Path(value).expanduser()
        return path/'results' if (path/'results/results.json').exists() else path
    return results.compare(results.read(resolve(before)),results.read(resolve(after)))


@server.tool()
def check_mainline(backend: str, intent_root: str | None = None, programs: list[str] | None = None,
                   environment: str | None = None, compiler: str | None = None,
                   stage: str = 'run') -> dict:
    """Run the mainline author's programs through intent-bench check; returns a background run."""
    from intent.examples import PROGRAMS
    from .request import RunRequest
    from .model import Measurement
    from .environments import load_environment
    env = load_environment(environment, compiler=compiler,
                           intent_root=Path(intent_root) if intent_root else None)
    names = programs or tuple(PROGRAMS)
    if set(names)-PROGRAMS.keys():
        raise ValueError("unknown author program")
    return jobs.start([RunRequest(f"mainline/{backend}/{name}", ("intent:"+backend,),
                                 env, Measurement(enabled=False), kind="program", stage=stage)
                       for name in names])


def main():
    server.run(transport='stdio')


if __name__=='__main__':main()
