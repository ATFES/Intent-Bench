"""Compile the hand-written Weft baseline through Weft's own public artifact ABI."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import shlex

from ..loading import load
from ..model import BenchError,EnvironmentUnavailable,PreparedCall
from ..execution import WorkerSession


class SourceCompilationError(BenchError):
    status='compilation_failed'


def deployment(env,ime=False):
    from weft.runtime.driver import Toolchain
    from weft.runtime.target import NativeTarget
    compiler = env.options.get('weft_compiler') or os.environ.get('INTENT_WEFT_COMPILER')
    native = NativeTarget.discover(Toolchain(compiler=compiler))
    extensions = [name for name,cpus in native.matrix_extensions if native.cpus.issubset(cpus)]
    if ime and len(extensions) != 1:
        raise EnvironmentUnavailable('this workload requires one available native Weft matrix extension')
    command = env.options.get('cc') or shlex.split(os.environ.get('CC','cc'))
    if isinstance(command,str):
        command = shlex.split(command)
    return {**native.build_target.to_dict(), 'cpus':sorted(native.cpus), 'cc':command,
            'cflags':env.options.get('cflags',[]), 'matrix_extension':extensions[0] if ime else None}


def prepare(reference,arguments,outputs,bindings,dimensions,env,*,ime=False,meta=(),matrix_extension=True):
    try:import weft
    except ImportError as error:raise EnvironmentUnavailable('install Weft in the execution environment') from error
    compiler=env.options.get('weft_compiler') or os.environ.get('INTENT_WEFT_COMPILER') or shutil.which('weft-compile')
    if not compiler:raise EnvironmentUnavailable('select --option weft_compiler="/path/to/weft-compile"')
    profile=deployment(env,ime)
    definition=load(str(Path(__file__).parents[1]/reference[0])+':'+reference[1])
    canonical=weft.lower_to_mlir(definition)
    command=[compiler,'--emit=artifact','--march='+profile['march'],'--abi='+profile['abi'],
             '--vlen-bits='+str(profile['vlen_bits']),'--private-stack-bytes=32768',*(f'--meta={k}={v}' for k,v in meta)]
    if matrix_extension and profile.get('matrix_extension'):command.append('--matrix-extension='+profile['matrix_extension'])
    started=time.perf_counter();proc=subprocess.run(command,input=canonical,text=True,capture_output=True)
    if proc.returncode:raise SourceCompilationError(proc.stderr)
    lowering=time.perf_counter()-started;artifact=json.loads(proc.stdout)
    native=Path(__file__).parents[1]/'native'
    request={'profile':profile,'artifact':artifact,'arity':len(arguments),'outputs':outputs,'bindings':bindings,'dimensions':dimensions}
    session=WorkerSession(env,'weft_source_worker.py',[native/'weft_source_worker.py',native/'cnnl_worker.py'],request,arguments)
    (session.local/'canonical.mlir').write_text(canonical)
    metadata={**session.metadata(),'source_compile_seconds':lowering,'weft_compiler':compiler}
    def measure(settings):
        metadata['cache_policy']='flush_cpu_64m' if settings.cache=='flush' else 'warm'
        return session.call('measure',samples=settings.samples,warmup=settings.warmup,cache=settings.cache)
    return PreparedCall(lambda:session.call('run'),session.result,close=session.close,native_measure=measure,
                        device=session.device('cpu'),timing='cpu_native_monotonic',metadata=metadata)
