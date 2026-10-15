import ast
from pathlib import Path

def validate_program(path: Path, *, language: str, generated_source: bool=False) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed = {'torch', 'triton', 'intent', 'math', 'functools', 'typing', 'collections', 'dataclasses', '__future__'}
    if language == 'intent':
        allowed.remove('triton')
    intent_names = set()
    torch_names = set()
    torch_result_types = {'max', 'min'}
    torch_api = {'Tensor', 'Size', 'dtype', 'device', 'empty', 'empty_like', 'empty_strided', 'zeros', 'ones', 'ones_like', 'zeros_like', 'std', 'finfo', 'iinfo', 'is_tensor', 'numel', 'broadcast_to', 'promote_types', 'result_type', 'broadcast_shapes', 'view_as_real', 'view_as_complex', 'bool', 'int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'long', 'short', 'float', 'float16', 'float32', 'float64', 'bfloat16', 'half', 'double', 'complex64', 'complex128', 'strided', 'contiguous_format', 'preserve_format', 'channels_last'}
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in tree.body:
        if isinstance(node, ast.Import):
            intent_names.update((item.asname or item.name for item in node.names if item.name == 'intent'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            torch_names.update((item.asname or item.name for item in node.names if item.name == 'torch'))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [item.name.split('.')[0] for item in node.names]
            if any((item.name.startswith('torch.') for item in node.names)):
                raise ValueError('Torch imports are limited to allocation, tensor metadata and dtype APIs')
        elif isinstance(node, ast.ImportFrom):
            names = [node.module.split('.')[0]] if node.module else []
            if node.level:
                raise ValueError('candidate imports must not reach other programs or the evaluator')
            runtime_hooks = generated_source and node.module == 'intent.runtime.triton' and all((item.name == 'TuningHooks' for item in node.names))
            if node.module is not None and node.module.startswith('intent.') and (node.module != 'intent.language') and (not runtime_hooks):
                raise ValueError('candidate author code may import only the public Intent language API')
            if node.module is not None and node.module.split('.')[0] == 'torch':
                result_types = node.module == 'torch.return_types' and all((item.name in torch_result_types for item in node.names))
                if not result_types and (node.module != 'torch' or any((item.name not in torch_api for item in node.names))):
                    raise ValueError('Torch imports are limited to allocation, tensor metadata and dtype APIs')
            if node.module == 'intent' and any((item.name in {'compile', 'compile_ir', 'generate', 'compile_shared_gpu'} for item in node.names)):
                raise ValueError('compiler calls belong to context.compile(), not the candidate host')
        else:
            names = []
        if set(names) - allowed:
            raise ValueError(f'candidate import outside the language/host API allowlist: {names}')
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and (node.id in torch_names):
            parent = parents.get(node)
            container = parents.get(parent)
            result_type = isinstance(parent, ast.Attribute) and parent.attr == 'return_types' and isinstance(container, ast.Attribute) and (container.value is parent) and (container.attr in torch_result_types)
            if not result_type and (not isinstance(parent, ast.Attribute) or parent.value is not node or parent.attr not in torch_api):
                raise ValueError('Torch is available only through allocation, tensor metadata and dtype APIs')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('__') or node.attr in {'open', 'from_file', 'fromfile', 'tofile', 'read_text', 'read_bytes', 'write_text', 'write_bytes', 'unlink', 'rename', 'mkdir', 'rmdir', 'iterdir', 'glob'}):
            raise ValueError(f'candidate may not access files or runtime internals through {node.attr}')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id in {'open', 'exec', 'eval', 'compile', '__import__', 'globals', 'locals', 'breakpoint', 'getattr', 'setattr', 'delattr', 'vars'}):
            raise ValueError(f'candidate may not use {node.func.id}()')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id in intent_names and (node.func.attr in {'compile', 'compile_ir', 'generate', 'compile_shared_gpu'}):
                raise ValueError('compiler calls belong to context.compile(), not the candidate host')
