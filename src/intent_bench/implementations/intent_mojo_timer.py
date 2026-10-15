"""Use the selected Mojo provider entry's in-process monotonic timer."""


def sample(call,settings,restore):
    selected=call.choose()
    timer=call.program.measurements[selected]
    for _ in range(settings.warmup):
        restore();call.launch()
    samples=[]
    for _ in range(settings.samples):
        restore()
        samples.append(timer(*call.native_arguments,1))
    return samples
