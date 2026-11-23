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


def prepare_input_oracle(specification, data=None):
    """Use a registered input factory and independent CPU numerical function."""
    from .corpus.cpu.baselines.pytorch import cpu_runtime
    if data is None:
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
    factory = entries.get('inputs')
    targets=('triton','cutile') if provider in ('triton','cutile') else (provider,)
    def prepare(side,target,case,payload,env,entry=None):
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
                        intent_root=Path(env.intent_root).resolve() if env.intent_root else None,environment=env,
                        inputs=payload.arguments if factory is not None else None)
        phases={};stage='input_preparation';started=time.perf_counter()
        def observe(next_stage):
            nonlocal stage,started
            now=time.perf_counter();phases[stage]=phases.get(stage,0)+now-started
            stage,started=next_stage,now
        with ExitStack() as stack:
            use_cuda = provider in ('triton', 'cutile') and (
                side != 'oracle' or factory is None or env.device.startswith('cuda'))
            if use_cuda:
                stack.enter_context(device_window(f'cuda:{index}'))
                stack.enter_context(torch.cuda.device(index))
            stack.enter_context(torch.random.fork_rng(devices=[index] if use_cuda else []))
            stack.enter_context(observe_stages(observe))
            if factory is None:
                torch.manual_seed(payload)
            entry=entries[side] if entry is None else entry
            prepared=(prepare_input_oracle(entry, payload if factory is not None else None) if isinstance(entry,dict)
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
        owners=prepared.owners
        for owner in (owners.values() if isinstance(owners,dict) else (owners,)):
            directory = getattr(owner, 'cache_directory', None)
            if directory is not None and str(directory) not in artifacts:
                artifacts.append(str(directory))
        tolerances[case.name]=prepared.tolerance
        if prepared.device == 'cuda':
            prepared.device=f'cuda:{index}'
        prepared.metadata.update(source=entry,preparation_stages_seconds=phases,
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
    for name,entry in entries.get('alternatives',{}).items():
        if name in impls or name.startswith('intent:'):
            raise ValueError(f'duplicate or reserved implementation name: {name}')
        impls[name]=Implementation(name,lambda case,seed,env,name=name,entry=entry:
            prepare(name,provider,case,seed,env,entry))
    def check(actual,expected):
        from .numeric import compare_outputs
        if provider=='bangc' or entries.get("v2_symbol"):
            def cpu(tree):
                if isinstance(tree,(list,tuple)):return tuple(cpu(v) for v in tree)
                return tree.cpu() if hasattr(tree,"cpu") else tree
            actual,expected=cpu(actual),cpu(expected)
        compare_outputs(actual,expected,next(iter(tolerances.values())))
    if entries.get("v2_symbol"):
        from .implementations.v2_program import bind
        from .workloads.programs import _device
        def v2(case, payload, env, backend):
            device = _device(env, backend)
            if device.startswith("mlu:"):
                import torch_mlu
            args = [v.to(device) if hasattr(v, "to") else v for v in payload.arguments]
            if adapter := entries.get("v2_arguments"):
                args = load(adapter)(*args)
            return bind(entries["v2_symbol"], backend, args, env, device)
        for backend in ("triton","cutile","mojo","weft","bangc"):
            name = "intent-v2:"+backend
            impls[name] = Implementation(name,lambda case,payload,env,b=backend:v2(case,payload,env,b))
            legacy="intent:"+backend
            if legacy not in impls and entries["v2_symbol"] in {"fused_add_rms_norm","dropout_residual_rms_norm","layer_norm_backward"}:
                from .implementations.legacy_norms import prepare as legacy_call
                from functools import partial
                impls[legacy]=Implementation(legacy,partial(legacy_call,entries["v2_symbol"],backend))
    def inputs(case, seed):
        if factory is None:
            return seed
        import torch
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if provider in ("triton","cutile") else []):
            torch.manual_seed(seed)
            return load(factory)()
    return Workload(row['name'],[Case(row['case'],{'profile':'original'})],inputs,impls,
                    oracle=Implementation('oracle',lambda case,seed,env:prepare(
                        'oracle' if 'oracle' in entries else 'source',provider,case,seed,env)),
                    check=check,family=row['family'],parameter_keys=(),materialized_inputs=factory is not None)
