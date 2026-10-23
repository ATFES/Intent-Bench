"""Sample one or more public product native invocations in a single interval."""
from .native_sequence import NativeSequence


def sample(call, settings, restore, *, metadata=None):
    timer = NativeSequence((call,))
    try:
        return sample_timer(timer, settings, restore, metadata=metadata)
    finally:
        timer.close()


def sample_timer(timer, settings, restore, *, metadata=None):
    import numpy as np
    if metadata is not None:
        metadata['cache_policy'] = 'flush_cpu_64m' if settings.cache == 'flush' else 'warm'
    cache = np.empty(64*1024*1024, dtype=np.uint8) if settings.cache == "flush" else None
    for _ in range(settings.warmup):
        restore()
        timer.run()
    values = []
    for _ in range(settings.samples):
        restore()
        if cache is not None:
            cache.fill(0)
        values.append(timer.run())
    return values
