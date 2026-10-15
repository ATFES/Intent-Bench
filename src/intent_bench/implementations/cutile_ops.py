import cuda.tile as ct


@ct.kernel
def softmax(x, y, block: ct.Constant[int]):
    row = ct.bid(0)
    values = ct.load(x, index=(row, 0), shape=(1, block), padding_mode=ct.PaddingMode.NEG_INF).astype(ct.float32)
    numer = ct.exp(values - ct.max(values, axis=1, keepdims=True))
    output = numer / ct.sum(numer, axis=1, keepdims=True)
    ct.store(y, index=(row, 0), tile=output.astype(y.dtype))


@ct.kernel
def relu(x, y, block: ct.Constant[int]):
    values = ct.load(x, index=(ct.bid(0), 0), shape=(1, block), padding_mode=ct.PaddingMode.ZERO)
    ct.store(y, index=(ct.bid(0), 0), tile=ct.maximum(values, 0))
