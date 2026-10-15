import os
from pathlib import Path
import shutil
import tempfile
import numpy as np

from ..loading import intent_environment,load
from ..model import EnvironmentUnavailable,PreparedCall
from .remote import NativeSession
from .weft_source import deployment


def prepare(reference,args,env,*,ime=False):
    intent=intent_environment(env)
    from intent.runtime.weft import TargetProfile,export_artifact
    profile=deployment(env,ime)
    target=intent.WeftTarget(vector_bits=profile['vlen_bits'],workers=env.threads,
                             matrix_extension=profile.get('matrix_extension'))
    program=intent.generate(load(reference),target=target,compiler=env.compiler)
    compiler=env.options.get('weft_compiler') or os.environ.get('INTENT_WEFT_COMPILER') or shutil.which('weft-compile')
    if not compiler:raise EnvironmentUnavailable('select --option weft_compiler="/path/to/weft-compile"')
    root=Path(tempfile.mkdtemp(prefix='intent-bench-weft-'));export=root/'program'
    try:
        export_artifact(program,export,compiler=compiler,profile=TargetProfile.from_deployment(profile))
        arrays=[];arguments=[]
        for value in args:
            if isinstance(value,np.ndarray):arguments.append({'input':len(arrays)});arrays.append(value)
            else:arguments.append(value)
        native=Path(__file__).parents[1]/'native'
        session=NativeSession(profile['host'],'weft_intent_worker.py',[native/'weft_intent_worker.py',native/'cnnl_worker.py',
            export,Path(intent.__file__).parent],{'profile':profile,'arguments':arguments},arrays,env.options.get('python','python3'))
    finally:shutil.rmtree(root)
    metadata=session.metadata()
    def measure(settings):
        metadata['cache_policy']='flush_cpu_64m' if settings.cache=='flush' else 'warm'
        return session.call('measure',samples=settings.samples,warmup=settings.warmup,cache=settings.cache)
    return PreparedCall(lambda:session.call('run'),session.result,close=session.close,native_measure=measure,
                        device=profile['host']+'/cpu',timing='cpu_native_monotonic',metadata=metadata)
