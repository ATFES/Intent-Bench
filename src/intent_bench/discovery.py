"""The discovery surface shared by the terminal and Agent tools."""
from .catalog import catalog, ADVANCED_OPERATIONS


def list_workloads():
    from .registry import workloads
    return [dict(item.metadata) for item in workloads().values()]


def describe_workload(name):
    from .registry import workloads
    registration = workloads().get(name)
    if registration is not None and "case" in registration.metadata:
        row = dict(registration.metadata)
        row.update(parameter_keys=[], cases=[{'name': row['case'], 'parameters': {}}],
                   input_description='Retained production profile; dimensions are fixed by this authored case.')
    else:
        from .loading import load_workload
        value = load_workload(name)
        row = {'name': value.name, 'family': value.family, 'description': value.description,
               'implementations': list(value.implementations), 'parameter_keys': value.parameter_keys,
               'cases': [{'name': case.name, 'parameters': dict(case.parameters)} for case in value.cases]}
    row['implementation_details'] = [describe_implementation(item) for item in row['implementations']]
    return row


def describe_implementation(name):
    """Explain project implementation names without selecting a default opponent."""
    if name.startswith('intent:'):
        target = name.partition(':')[2]
        return {'name': name, 'kind': 'intent', 'target': target,
                'description': f'Intent compiled for {target}; choose the execution environment separately.'}
    if name.startswith('reference:'):
        return {'name': name, 'kind': 'reference',
                'description': 'Numerical reference on another execution stack; not a same-device performance baseline.'}
    descriptions = {
        'triton': 'Native Triton implementation.', 'cutile': 'Native cuTile implementation.',
        'mojo': 'Native Mojo implementation.', 'torch': 'Independent PyTorch implementation.',
        'numpy': 'Independent NumPy implementation.', 'cnnl': 'Native CNNL implementation on MLU.',
        'ggml': 'Independent GGML implementation.', 'weft': 'Authored Weft source implementation.',
        'cub': 'NVIDIA CUB device primitives, compiled for the executing GPU.',
    }
    return {'name': name, 'kind': 'library' if name in descriptions else 'custom',
            'description': descriptions.get(name, 'User-provided implementation; inspect its workload contract.')}
