"""Run the retained Weft algorithms with Triton's CPU JIT and public native SDK."""
import ctypes as ct
from pathlib import Path
import os
import shlex
import subprocess
import tempfile

import numpy as np

from ..loading import load
from ..model import PreparedCall
from .native_sampling import sample_timer


class View(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('bytes', ct.c_size_t), ('encoding', ct.c_char_p),
                ('rank', ct.c_size_t), ('shape', ct.POINTER(ct.c_size_t)),
                ('strides', ct.POINTER(ct.c_size_t)), ('readonly', ct.c_bool)]


class Error(ct.Structure):
    _fields_ = [('status', ct.c_int), ('message', ct.c_char * 1024)]


def prepare(reference, arguments, outputs, bindings, env, *, meta=(), use_matrix=False):
    from triton.runtime.weft.buffer import Buffer
    from triton.backends.weft.driver import WeftDriver
    from weft_core import sdk_path

    target = WeftDriver().get_current_target()
    definition = load(str(Path(__file__).parents[1] / reference[0]) + ':' + reference[1])
    output = [np.empty(item['shape'], dtype=item['dtype']) for item in outputs]
    values = [*arguments, *output]
    buffers, owners = [], [values]
    for binding in bindings:
        data = values[binding['buffer']] if 'buffer' in binding else np.empty(binding['workspace'], dtype=np.uint8)
        buffers.append(Buffer(data, shape=binding.get('shape'), encoding=binding.get('encoding')))
        owners.append(data)
    saved = [value.copy() for value in arguments]
    state = {}
    metadata = {'producer': 'triton-weft', 'source': reference[0],
                'scope_note': 'Prepared native single-instance operator; JIT, SDK binding and worker checks outside timing'}

    def compile_call():
        if state:
            return
        root = Path(os.environ['INTENT_BENCH_RUN_DIR']) / 'artifacts'
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix='weft-source-', dir=root))
        metadata['native_directory'] = str(directory)
        options = {'num_cpu_threads': env.threads}
        if target.arch == 'riscv64':
            options.update(private_stack_bytes=32768,
                           matrix_extension=target.matrix_extension if use_matrix else 'none')
        compiled = definition.warmup(*buffers, **dict(meta), grid=(1,), backend_options=options)
        compiled.export(directory / 'artifact')
        sdk = Path(sdk_path())
        library_path = directory / 'binding.so'
        command = [*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17', '-O2', '-shared', '-fPIC',
                   str(Path(__file__).parents[1] / 'native/weft_prepared.cpp'),
                   '-I' + str(sdk / 'include'), '-L' + str(sdk / 'lib'),
                   '-Wl,-rpath,' + str(sdk / 'lib'), '-lweft_runtime', '-ldl', '-o', str(library_path)]
        with (directory / 'compiler.log').open('w') as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if status.returncode:
            raise RuntimeError('Weft native timer build failed: ' + (directory / 'compiler.log').read_text())
        library = ct.CDLL(str(library_path))
        library.prepare.argtypes = [ct.c_char_p, ct.POINTER(View), ct.c_size_t, ct.POINTER(Error)]
        library.prepare.restype = ct.c_void_p
        for name in ('launch', 'release', 'measure'):
            getattr(library, name).argtypes = [ct.c_void_p]
            getattr(library, name).restype = ct.c_double if name == 'measure' else None
        library.validate.argtypes = [ct.c_void_p, ct.POINTER(Error)]
        library.validate.restype = ct.c_int
        views = []
        for buffer in buffers:
            shape = (ct.c_size_t * len(buffer.shape))(*buffer.shape)
            strides = (ct.c_size_t * len(buffer.strides))(*buffer.strides)
            encoding = buffer.encoding.encode()
            owners.extend((shape, strides, encoding))
            views.append(View(buffer.address, buffer.nbytes, encoding, len(buffer.shape), shape, strides, buffer.readonly))
        descriptors = (View * len(views))(*views)
        error = Error()
        handle = library.prepare(os.fsencode(directory / 'artifact'), descriptors, len(views), ct.byref(error))
        if not handle:
            raise RuntimeError(bytes(error.message).decode())
        state.update(library=library, handle=handle, compiled=compiled, descriptors=descriptors)

    def validate():
        error = Error()
        if state['library'].validate(state['handle'], ct.byref(error)):
            raise RuntimeError(bytes(error.message).decode())

    def launch():
        validate()
        state['library'].launch(state['handle'])

    def reset():
        for data, initial in zip(arguments, saved, strict=True):
            np.copyto(data, initial)

    class Timer:
        def run(self):
            validate()
            return state['library'].measure(state['handle'])

    def close():
        if state:
            state['library'].release(state['handle'])
            state.clear()

    return PreparedCall(launch, lambda: output[0] if len(output) == 1 else tuple(output),
                        compile=compile_call, reset=reset, close=close, device='cpu',
                        native_measure=lambda settings: sample_timer(Timer(), settings, reset, metadata=metadata),
                        timing='cpu_native_monotonic', metadata=metadata, owners=(buffers, owners))
