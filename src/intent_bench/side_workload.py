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


def build(row,entries):
    provider=row['provider'];tolerances={}
    targets=('triton','cutile') if provider in ('triton','cutile') else (provider,)
    def prepare(side,target,case,seed,env):
        import torch
        torch.set_num_threads(env.threads)
        from .corpus import PROJECT_ROOT
        from .corpus._common.model import Context
        from .corpus._common.measurement import observe_stages
        from .measurement import device_window
        index=int(env.device.split(':')[-1]) if env.device.startswith('cuda:') else 0
        if side=='intent':
            intent_environment(env)
            from intent.tools.backends import make_target
            selected_target=make_target(target,{'device':index} if target in ('triton','cutile','bangc') else {'workers':env.threads})
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
            if provider in ('triton','cutile','bangc'):
                stack.enter_context(device_window(f'cuda:{index}'))
                stack.enter_context(torch.cuda.device(index))
            stack.enter_context(torch.random.fork_rng(devices=[index] if provider in ('triton','cutile','bangc') else []))
            stack.enter_context(observe_stages(observe))
            factory=load(entries[side])
            torch.manual_seed(seed)
            prepared=factory(context)
        observe('ready')
        if prepared.launch is None:raise ExpressionGap(prepared.note or 'no implementation supplied')
        artifacts = list(context.artifacts)
        for owner in (prepared.owners or {}).values():
            directory = getattr(owner, 'cache_directory', None)
            if directory is not None and str(directory) not in artifacts:
                artifacts.append(str(directory))
        tolerances[case.name]=prepared.tolerance
        launch=prepared.launch
        if isinstance(launch,PreparedCall):
            launch.metadata.update(source=entries[side],input_seed=seed,preparation_stages_seconds=phases,
                                   compiler_artifacts=artifacts)
            return launch
        # Some upstream calls hold pointer tables rather than Tensor owners.
        # Keep the whole prepared side alive until the final synchronization.
        return PreparedCall(lambda:prepared.launch.launch(),lambda:prepared.launch.outputs(),reset=launch.prepare,
                            device=f'cuda:{index}' if prepared.device_type=='cuda' else 'cpu',
                            scopes=('prepared_call','host_call'),
                            metadata={'note':prepared.note,'preparation_stages_seconds':phases,
                                      'input_seed':seed,'source':entries[side],'artifacts':artifacts})
    source_name='torch' if provider=='mojo' else 'reference:cuda' if provider=='bangc' else provider
    if provider=='bangc' and row['kernel'] in {'causal_attention_f32','causal_linear_attention_f32'}:
        source_name='reference:mojo'
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
                    oracle=Implementation('oracle',lambda case,seed,env:prepare('source',provider,case,seed,env)),
                    check=check,family=row['family'],parameter_keys=())
