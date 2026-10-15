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
    """Show available implementations and input cases for an operator or absolute file.py:bench."""
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
                  compiler: str | None = None, options: dict | None = None) -> dict:
    """Start an operator comparison using the ordinary bench CLI. Returns a run path immediately.

    Choose implementations explicitly, e.g. torch,triton,intent:triton. Environment
    selects execution settings independently. Use run_status to retrieve the result.
    A custom workload file executes user Python code just as the CLI does.
    """
    if not implementations:raise ValueError('select at least one implementation')
    arguments=['run',workload,'--impl',','.join(implementations),'--cache',cache]
    for flag,value in [('--env',environment),('--baseline',baseline),('--throughput',throughput),
                       ('--device',device),('--intent-root',intent_root),('--compiler',compiler)]:
        if value is not None:arguments.extend([flag,value])
    if check_only:arguments.append('--check-only')
    for case in cases or []:arguments.extend(['--case',case])
    for key,value in (parameters or {}).items():arguments.extend(['--param',key+'='+json.dumps(value)])
    for key,value in (options or {}).items():arguments.extend(['--option',key+'='+json.dumps(value)])
    return jobs.start(arguments)


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
def check_mainline(intent_root: str, target: str, programs: list[str] | None = None,
                   environment: str | None = None, compiler: str | None = None,
                   stage: str = 'run') -> dict:
    """Run the mainline author's programs through intent-bench check; returns a background run."""
    arguments=['check','--intent-root',intent_root,'--target',target,'--stage',stage]
    if environment:arguments.extend(['--env',environment])
    if compiler:arguments.extend(['--compiler',compiler])
    for program in programs or []:arguments.extend(['--program',program])
    return jobs.start(arguments)


def main():
    import argparse
    import sys
    parser=argparse.ArgumentParser(prog='intent-bench-mcp')
    parser.add_argument('--component',choices=('bench','manual','compiler'),default='bench')
    parser.add_argument('--intent-root',type=Path)
    args,remaining=parser.parse_known_args()
    if args.component=='bench':
        if remaining:parser.error('unrecognized arguments: '+' '.join(remaining))
        server.run(transport='stdio')
    else:
        from .loading import intent_environment
        from .model import Environment
        from importlib import import_module
        intent_environment(Environment(intent_root=args.intent_root))
        sys.argv=[sys.argv[0],*remaining]
        if args.component=='manual' and args.intent_root is not None and '--corpus' not in remaining:
            import tempfile
            from intent.tools.manual import snapshot
            with tempfile.TemporaryDirectory(prefix='intent-bench-manual-') as directory:
                corpus=Path(directory)/'manual.json'
                corpus.write_text(json.dumps(snapshot(args.intent_root)))
                sys.argv.extend(['--corpus',str(corpus)])
                import_module('intent.mcp.manual').main()
            return
        import_module('intent.mcp.'+args.component).main()


if __name__=='__main__':main()
