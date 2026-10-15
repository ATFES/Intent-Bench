"""An authored Intent sequence, exported to one persistent remote worker."""
from pathlib import Path
import json
import tempfile

from ..model import EnvironmentUnavailable, PreparedCall
from .remote import NativeSession


def prepare_sequence(sequence, outputs):
    import intent
    import torch
    from intent.runtime.bangc import export_artifact
    from .bangc_sequence import DTYPES
    env=sequence.context.environment
    host=env.options.get('host')
    if not host:raise EnvironmentUnavailable('Intent BANG C requires --option host="SSH_ALIAS"')
    root=Path(tempfile.mkdtemp(prefix='intent-bench-bangc-'))
    package=root/'program';package.mkdir()
    buffers={};storage_ids={};steps=[]
    def describe(value):
        if not isinstance(value,torch.Tensor):return value
        storage=value.untyped_storage();key=(str(value.device),storage.data_ptr(),storage.nbytes())
        if key not in storage_ids:
            name='buffer'+str(len(storage_ids));storage_ids[key]=name
            info={'dtype':DTYPES[value.dtype],'elements':storage.nbytes()//value.element_size()}
            if key not in sequence.allocated_outputs:
                raw=torch.empty(0,dtype=torch.uint8,device=value.device).set_(storage,0,(storage.nbytes(),),(1,))
                (package/(name+'.bin')).write_bytes(raw.cpu().numpy().tobytes());info['file']=name+'.bin'
            buffers[name]=info
        return {'buffer':storage_ids[key],'shape':list(value.shape),'strides':list(value.stride()),'offset':value.storage_offset()}
    for index,(program,arguments,interface) in enumerate(sequence.steps):
        name='kernel'+str(index);export_artifact(program,package/name)
        steps.append({'artifact':name,'arguments':[describe(arguments[p.name]) for p in interface.parameters]})
    def output_tree(value):
        return {'items':[output_tree(item) for item in value]} if isinstance(value,tuple) else describe(value)
    manifest={'buffers':buffers,'steps':steps,'outputs':output_tree(outputs)}
    (package/'manifest.json').write_text(json.dumps(manifest))
    native=Path(__file__).parents[1]/'native'
    request={'device':env.options.get('device',0),'neuware':env.options.get('neuware','/usr/local/neuware')}
    try:
        session=NativeSession(host,'bangc_worker.py',[native/'bangc_worker.py',native/'cnnl_worker.py',package,Path(intent.__file__).parent],request,(),env.options.get('python','python3'))
    finally:
        import shutil
        shutil.rmtree(root)
    return PreparedCall(lambda:session.call('run'),session.result,close=session.close,
        native_measure=lambda settings:session.call('measure',samples=settings.samples,warmup=settings.warmup),
        device=f"{host}/mlu:{request['device']}",timing='cnrt_notifier',
        metadata={**session.metadata(),
                  'scope_note':'Complete authored sequence; transfers and state restoration excluded'})
