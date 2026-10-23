"""Time a prepared sequence of C ABI kernels in one native interval."""
import ctypes as ct
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import shutil


def _type(value):
    names = {ct.c_void_p: 'void*', ct.c_bool: 'bool', ct.c_float: 'float', ct.c_double: 'double',
             ct.c_int8: 'int8_t', ct.c_uint8: 'uint8_t', ct.c_int16: 'int16_t', ct.c_uint16: 'uint16_t',
             ct.c_int32: 'int32_t', ct.c_uint32: 'uint32_t', ct.c_int64: 'int64_t', ct.c_uint64: 'uint64_t'}
    if value not in names:
        raise TypeError(f'unsupported native sequence ABI type: {value}')
    return names[value]


class _Call(ct.Structure):
    _fields_ = [('function', ct.c_void_p), ('arguments', ct.POINTER(ct.c_void_p)), ('signature', ct.c_size_t)]


class _Fill(ct.Structure):
    _fields_ = [('data', ct.c_void_p), ('count', ct.c_size_t),
                ('width', ct.c_size_t), ('value', ct.c_ubyte * 8)]


class NativeSequence:
    def __init__(self, calls, *, initializations=()):
        import torch
        self.initializations = tuple(initializations)
        fills = []
        for buffer, value in self.initializations:
            if buffer.device.type != 'cpu' or not buffer.is_contiguous():
                raise NotImplementedError('native sequence initialization requires a contiguous CPU buffer')
            width = buffer.element_size()
            if width not in (1, 2, 4, 8):
                raise NotImplementedError('native sequence initialization requires a 1, 2, 4 or 8 byte element')
            encoded = torch.tensor([value], dtype=buffer.dtype).view(torch.uint8).tolist()
            fills.append(_Fill(buffer.data_ptr(), buffer.numel(), width, (ct.c_ubyte * 8)(*encoded)))
        self.fills = (_Fill * len(fills))(*fills)
        signatures, entries, owners = [], [], []
        for call in calls:
            binding = call.native_call()
            function = binding.function
            types = binding.argument_types
            signature = tuple(_type(t) for t in types)
            if signature not in signatures:
                signatures.append(signature)
            values = binding.arguments
            pointers = (ct.c_void_p * len(values))(*(ct.addressof(v) for v in values))
            owners.append((binding, values, pointers))
            entries.append(_Call(ct.cast(function, ct.c_void_p), pointers, signatures.index(signature)))
        self.owners = owners
        self.calls = (_Call * len(entries))(*entries)
        cases = []
        for index, types in enumerate(signatures):
            arguments = ', '.join(f'*static_cast<{t}*>(call.arguments[{i}])' for i, t in enumerate(types))
            cases.append(f'case {index}: reinterpret_cast<void (*)({", ".join(types)})>(call.function)({arguments}); break;')
        source = '''#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
struct Call { void* function; void** arguments; size_t signature; };
struct Fill { void* data; size_t count; size_t width; unsigned char value[8]; };
template<size_t Width> void initialize(const Fill& fill) {
    auto* data = static_cast<unsigned char*>(fill.data);
    for (size_t i = 0; i < fill.count; ++i)
        std::memcpy(data + i * Width, fill.value, Width);
}
extern "C" double measure(Call* calls, size_t count, Fill* fills, size_t fill_count) {
    auto begin = std::chrono::steady_clock::now();
    for (size_t i = 0; i < fill_count; ++i) {
        const auto& fill = fills[i];
        switch (fill.width) {
        case 1: initialize<1>(fill); break;
        case 2: initialize<2>(fill); break;
        case 4: initialize<4>(fill); break;
        case 8: initialize<8>(fill); break;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const auto& call = calls[i];
        switch (call.signature) {
''' + '\n'.join(cases) + '''
        }
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
}
'''
        from ..output import worker_root
        root = worker_root() / 'native'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = path = Path(tempfile.mkdtemp(prefix='sequence-', dir=root))
        (path / 'timer.cpp').write_text(source)
        result = subprocess.run([*shlex.split(os.environ.get('CXX', 'c++')), '-std=c++17', '-O2', '-shared', '-fPIC',
                                 str(path / 'timer.cpp'), '-o', str(path / 'timer.so')], capture_output=True, text=True)
        if result.returncode:
            (path/'compiler.log').write_text(result.stdout+result.stderr)
            raise RuntimeError(f'native sequence timer compilation failed; see {path}: ' + result.stderr)
        self.library = ct.CDLL(str(path / 'timer.so'))
        self.measure = self.library.measure
        self.measure.argtypes = [ct.POINTER(_Call), ct.c_size_t, ct.POINTER(_Fill), ct.c_size_t]
        self.measure.restype = ct.c_double

    def run(self):
        for binding, _, _ in self.owners:
            binding.validate()
        return self.measure(self.calls, len(self.calls), self.fills, len(self.fills))

    def close(self):
        self.measure = None
        self.library = None
        shutil.rmtree(self.directory)
