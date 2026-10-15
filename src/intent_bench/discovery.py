"""The discovery surface shared by the terminal and Agent tools."""
from .catalog import catalog, ADVANCED_OPERATIONS


def list_workloads():
    from .workloads.elementwise import build
    from .implementations.mojo import NAMES as CPU_NAMES
    from .workloads.mlu import NAMES as MLU_NAMES
    from .workloads.rvv import NAMES as RVV_NAMES
    rows=[{"name":name,"family":build(name).family,"implementations":list(build(name).implementations)}
          for name in ("softmax","relu")]
    rows.extend(catalog())
    rows.extend({'name':'advanced/'+name,'family':'advanced','implementations':['torch','mojo','intent:mojo']}
                for name in ADVANCED_OPERATIONS)
    rows.extend({'name':'cpu/'+name,'family':'cpu','implementations':['numpy','mojo']} for name in CPU_NAMES)
    rows.extend({'name':'mlu/'+name,'family':'mlu','implementations':['cnnl','intent:bangc']} for name in MLU_NAMES)
    rows.extend({'name':'rvv/'+name,'family':'rvv','implementations':['ggml','intent:weft']} for name in RVV_NAMES)
    return rows


def describe_workload(name):
    if name.startswith('corpus/'):
        return next(row for row in catalog() if row['name']==name)
    from .loading import load_workload
    value=load_workload(name)
    return {'name':value.name,'family':value.family,'description':value.description,
            'implementations':list(value.implementations),'parameter_keys':value.parameter_keys,
            'cases':[{'name':case.name,'parameters':dict(case.parameters)} for case in value.cases]}
