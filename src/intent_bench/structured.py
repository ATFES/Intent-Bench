from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

from .model import Case, Implementation, PreparedCall, Workload
from .trees import clone, restore
from .loading import intent_environment, load


@dataclass
class KernelSymbol:
    reference: str
    member: str

    def resolve(self):
        return getattr(load(self.reference), self.member)


@dataclass
class InputBundle:
    arguments: tuple
    kernel: str
    reference: str
    tolerance: object
    constexprs: dict | None = None
    mutable_positions: tuple[int, ...] = ()


def cpu_workload(row,factory):
    import torch
    from .corpus.cpu.baselines.pytorch import cpu_runtime
    last={}
    def inputs(case,seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            data=factory()
        last['tolerance']=data.tolerance
        return data
    def oracle(case,data):
        output = getattr(cpu_runtime,data.reference)(*data.arguments)
        return observed(output, data)
    def observed(output, data):
        return (output, tuple(data.arguments[i] for i in data.mutable_positions)) if data.mutable_positions else output
    def source(case,data,env):
        torch.set_num_threads(env.threads)
        if data.reference.endswith('.py'):
            from .corpus import PROJECT_ROOT
            prepare=load(str(PROJECT_ROOT/data.reference)+':prepare')
            call=prepare(SimpleNamespace(workers=env.threads,executable=env.options.get('mojo')), *data.arguments)
            return PreparedCall(call.launch,call.result,
                native_measure=lambda settings:call.samples(settings.samples,settings.warmup),timing='cpu_native_monotonic',
                metadata={'reference_source':data.reference})
        state={};initial=clone(data.arguments)
        def run():state['output']=oracle(case,data)
        return PreparedCall(run,lambda:state['output'],reset=lambda:restore(data.arguments,initial),
                            metadata={'reference_function':data.reference},scopes=("prepared_call","host_call"))
    def generated(case,data,env):
        if row['kernel'] in {'fused_linear_cross_entropy','paged_gqa_f16_reduction'}:
            from .implementations.advanced_intent import prepare
            return prepare(row['kernel'],data.arguments,env)
        intent=intent_environment(env)
        definition=load(data.kernel)
        constexprs={key:value.resolve() if isinstance(value,KernelSymbol) else value for key,value in (data.constexprs or {}).items()}
        from .implementations.intent import make_kernel
        artifact=make_kernel(definition,env,'mojo',constexprs=constexprs or None)
        call=artifact.prepare(*data.arguments)
        initial=clone(data.arguments)
        host_state={}
        from .implementations.native_sampling import sample
        reset=lambda:restore(data.arguments,initial)
        def host_run():host_state['output']=artifact.run(*data.arguments)
        return PreparedCall(call.launch,lambda:observed(call.result(),data),compile=call.compile,reset=reset,
                            native_measure=lambda settings:sample(call,settings,reset),timing='cpu_native_monotonic',
                            metadata={'artifacts':[str(a.cache_directory) for a in artifact.artifacts]},host_run=host_run,
                            host_result=lambda:observed(host_state['output'],data))
    def native_advanced(case,data,env):
        from .implementations.advanced_mojo import prepare
        call=prepare(row['kernel'],data.arguments,env)
        result=call.result
        call.result=lambda:observed(result(),data)
        return call
    def check(actual,expected):
        from .numeric import compare_outputs, NumericalComparisonError
        try:compare_outputs(actual,expected,last['tolerance'])
        except NumericalComparisonError as error:raise AssertionError(str(error)) from error
    native_source='/baselines/mojo/' in (row.get('source_runtime') or '')
    reference=Implementation('oracle',source) if native_source else oracle
    source_name='mojo' if native_source else 'torch'
    implementations={source_name:Implementation(source_name,source),'intent:mojo':Implementation('intent:mojo',generated)}
    if row['kernel'] in {'scaled_fp8_gemm','flash_attention_bias','flash_varlen_attention','chunk_retention',
                         'flash_attention_inline','flash_attention_select','flash_attention_full_causal_stream',
                         'mamba3_siso_step','mamba3_siso_forward','fused_linear_cross_entropy','paged_gqa_f16_reduction'}:
        implementations['mojo']=Implementation('mojo',native_advanced)
    return Workload(row['name'],[Case(row['case'],{'profile':'original'})],inputs,implementations,
                    oracle=reference,check=check,family=row['family'],parameter_keys=())
