"""Independently prepared implementations of the migrated operator workloads."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
import json
import time

from .loading import intent_environment,load
from .model import Case,ExpressionGap,Implementation,PreparedCall,Workload


def mappings():
    return json.loads((Path(__file__).parent/'sides/catalog.json').read_text())


def prepare_input_oracle(specification):
    """Use a registered input factory and independent CPU numerical function."""
    from .corpus.cpu.baselines.pytorch import cpu_runtime
    data = load(specification['inputs'])()
    function = getattr(cpu_runtime, specification.get('reference', data.reference))
    state = {}
    def launch():
        state['output'] = function(*data.arguments)
    return PreparedCall(launch, lambda: state['output'], device='cpu',
                        tolerance=data.tolerance,
                        metadata={'numerical_reference': function.__name__})


def build(row,entries):
    provider=row['provider'];tolerances={}
    targets=('triton','cutile') if provider in ('triton','cutile') else (provider,)
    def prepare(side,target,case,seed,env):
        import torch
        torch.set_num_threads(env.threads)
        from .corpus import PROJECT_ROOT
        from .implementations.context import Context
        from .corpus._common.measurement import observe_stages
        from .measurement import device_window
        index=int(env.device.split(':')[-1]) if env.device.startswith('cuda:') else 0
        if side=='intent':
            intent_environment(env)
            selected_target=SimpleNamespace(device=index,workers=env.threads)
        else:
            selected_target=SimpleNamespace(device=index,workers=env.threads)
        context=Context(env.compiler,PROJECT_ROOT,selected_target,target,compiler_timeout_seconds=60,
                        intent_root=Path(env.intent_root).resolve() if env.intent_root else None,environment=env)
        phases={};stage='input_preparation';started=time.perf_counter()
        def observe(next_stage):
            nonlocal stage,started
            now=time.perf_counter();phases[stage]=phases.get(stage,0)+now-started
            stage,started=next_stage,now
        with ExitStack() as stack:
            if provider in ('triton','cutile'):
                stack.enter_context(device_window(f'cuda:{index}'))
                stack.enter_context(torch.cuda.device(index))
            stack.enter_context(torch.random.fork_rng(devices=[index] if provider in ('triton','cutile') else []))
            stack.enter_context(observe_stages(observe))
            torch.manual_seed(seed)
            entry=entries[side]
            prepared=(prepare_input_oracle(entry) if isinstance(entry,dict)
                      else load(entry)(context))
        observe('ready')
        if not isinstance(prepared,PreparedCall):
            raise TypeError('implementation must return the public PreparedCall contract')
        compilers = list(dict.fromkeys(call.compile for call in context.prepared_calls if call.compile))
        if prepared.compile is not None and prepared.compile not in compilers:
            compilers.append(prepared.compile)
        if compilers:
            def compile_calls():
                for compile_call in compilers:
                    compile_call()
            prepared.compile = compile_calls
        artifacts = list(context.artifacts)
        for owner in (prepared.owners or {}).values():
            directory = getattr(owner, 'cache_directory', None)
            if directory is not None and str(directory) not in artifacts:
                artifacts.append(str(directory))
        tolerances[case.name]=prepared.tolerance
        if prepared.device == 'cuda':
            prepared.device=f'cuda:{index}'
        prepared.metadata.update(source=entries[side],input_seed=seed,preparation_stages_seconds=phases,
                                 compiler_artifacts=artifacts)
        if context.kernels:
            def compiler_observations():
                return [artifact.observation for kernel in context.kernels for artifact in kernel.artifacts
                        if artifact.observation is not None]
            prepared.compiler_observations = compiler_observations
        return prepared
    source_name='torch' if provider=='mojo' else 'reference:cuda' if provider=='bangc' else provider
    if provider=='bangc' and row['kernel'] in {'causal_attention_f32','causal_linear_attention_f32'}:
        source_name='reference:mojo'
    source_name=row.get('source_implementation',source_name)
    impls={source_name:Implementation(source_name,lambda case,seed,env:prepare('source',provider,case,seed,env))}
    impls.update({'intent:'+target:Implementation('intent:'+target,
        lambda case,seed,env,target=target:prepare('intent',target,case,seed,env)) for target in targets})
    def check(actual,expected):
        from .numeric import compare_outputs
        if provider=='bangc':
            def cpu(tree):return tuple(cpu(v) for v in tree) if isinstance(tree,tuple) else tree.cpu()
            actual,expected=cpu(actual),cpu(expected)
        compare_outputs(actual,expected,next(iter(tolerances.values())))
    return Workload(row['name'],[Case(row['case'],{'profile':'original'})],lambda case,seed:seed,impls,
                    oracle=Implementation('oracle',lambda case,seed,env:prepare(
                        'oracle' if 'oracle' in entries else 'source',provider,case,seed,env)),
                    check=check,family=row['family'],parameter_keys=())
