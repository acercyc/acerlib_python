import numpy as np
from numpy import array as npa


def symRange(v, centre=0):
    v -= centre
    if np.abs(np.max(v)) > np.abs(np.min(v)):
        lim = npa([-np.abs(np.max(v)), np.abs(np.max(v))])
    else:
        lim = npa([-np.abs(np.min(v)), np.abs(np.min(v))])

    lim += centre
    return lim


def norm_range(X, dataRange=(0, 1), axis=None):
    # 1.0 - Acer 2017/05/18 14:28
    X_std = (X - np.amin(X, axis, keepdims=True)) / (np.amax(X, axis, keepdims=True) - np.amin(X, axis, keepdims=True))
    X_scaled = X_std * (dataRange[1] - dataRange[0]) + dataRange[0]
    return X_scaled