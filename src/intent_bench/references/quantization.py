"""Numerical fields of the Q8_K activation record."""
import numpy as np


def q8_fields(blocks):
    rows = blocks.reshape(-1, 256)
    extreme = rows[np.arange(len(rows)), np.abs(rows).argmax(axis=1)]
    inverse = np.divide(np.float32(-127), extreme, out=np.zeros_like(extreme),
                        where=extreme != 0)
    scale = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse),
                      where=inverse != 0)
    codes = np.clip(np.rint(rows * inverse[:, None]), -128, 127).astype(np.int8)
    return scale, codes


def q8_records(blocks):
    """Little-endian GGML records, including all sixteen sums of stored codes."""
    scale, codes = q8_fields(blocks)
    sums = codes.reshape(-1, 16, 16).sum(axis=2, dtype=np.int16)
    return np.concatenate((scale.astype("<f4").view(np.uint8).reshape(-1, 4),
                           codes.view(np.uint8), sums.astype("<i2").view(np.uint8)), axis=1)
