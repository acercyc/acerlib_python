import numpy as np
from numpy import array as npa
import pandas as pd


def c2b(d, members):
    """
    categorical variable to binary coding
    :param d: 
    :param members:  categorical set members
    :return: data, look up table
    """
    lookup = pd.get_dummies(members)
    d_ = npa(lookup.loc[:, d]).transpose()
    return d_, lookup


def b2c(d, members):
    """
    binary coding to  categorical variable
    :param d:  data
    :param members:  categorical set members
    :return: data, look up table
    """
    lookup = pd.get_dummies(members)
    d_ = np.argmax(d, axis=1)
    d_ = npa(list(lookup))[d_]
    return d_, lookup
