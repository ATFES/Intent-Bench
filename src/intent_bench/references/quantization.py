"""Numerical fields of the Q8_K activation record."""
import numpy as np


def q8_fields(blocks):
    rows = blocks.reshape(-1, 256)
    maximum, minimum = rows.max(axis=1), rows.min(axis=1)
    extreme = np.where(np.abs(maximum) > np.abs(minimum), maximum, minimum)
    inverse = np.divide(np.float32(-127), extreme, out=np.zeros_like(extreme),
                        where=extreme != 0)
    scale = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse),
                      where=inverse != 0)
    codes = np.clip(np.rint(rows * inverse[:, None]), -128, 127).astype(np.int8)
    return scale, codes
