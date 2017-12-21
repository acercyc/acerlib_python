# ============================================================================ #
#                          1.0 - Acer 2017/04/24 15:44                         #
# ============================================================================ #
import numpy as np
from numpy import random


def timeSeriesBatchGenerator(d, wSize, iRow=None, isForward=1, nPredictTime=1, isShuffle=True):
    """
        :param d: 1st dimention is time
        :param wSize: window size
        :param iRow: row (time) index
        :param isForward: extract forward or backword
        :param nPredictTime: # of prediction row
        :param isShuffle
        :return: 2 lists
        # 1.0 - Acer 2017/12/14 19:36
        """

    if iRow is None:
        iRow = range(d.shape[0] - nPredictTime - wSize + 1)

    if isShuffle:
        np.random.shuffle(iRow)

    dcat = []
    prediction = []
    for i in iRow:
        if isForward:
            sheet = np.take(d, np.arange(i, i + wSize), axis=0)
            dcat.append(sheet)
            prediction.append(np.take(d, np.arange(i + wSize, i + wSize + nPredictTime), axis=0))
        else:
            sheet = np.take(d, np.arange(i - wSize, i), axis=0)
            dcat.append(sheet)
            prediction.append(np.take(d, np.arange(i, i + nPredictTime), axis=0))

    dcat = np.array(dcat)
    prediction = np.array(prediction)

    return dcat, prediction

# ============================================================================ #
#                                    Backup                                    #
# ============================================================================ #
# def timeSeriesBatchGen(d, wSize, iRow=None, isForward=1, nPredictTime=1, shuffle=None):
#     """
# 
#         :param d: 1st dimention is time
#         :param wSize: window size
#         :param iRow: row index
#         :param isForward: extract forward or backword
#         :param nPredictTime: # of prediction row
#         :param shuffle
#         :return: 2 lists
# 
#         1.0 - Acer 2017/01/26 15:12
#         2.0 - Acer 2017/04/24 15:47
#         3.0 - Acer 2017/05/15 20:32
#         3.1 - Acer 2017/05/16 20:42
#         """
# 
#     if iRow is None:
#         iRow = range(d.shape[0] - nPredictTime - wSize + 1)
#     dcat = []
#     prediction = []
#     for i in iRow:
#         if isForward:
#             sheet = np.take(d, np.arange(i, i + wSize), axis=0)
#             dcat.append(sheet)
#             prediction.append(np.take(d, np.arange(i + wSize, i + wSize + nPredictTime), axis=0))
#         else:
#             sheet = np.take(d, np.arange(i - wSize, i), axis=0)
#             dcat.append(sheet)
#             prediction.append(np.take(d, np.arange(i, i + nPredictTime), axis=0))
#     dcat = np.array(dcat)
#     prediction = np.array(prediction)
# 
#     if shuffle is not None:
#         iShuffle = np.arange(dcat.shape[0])
#         random.shuffle(iShuffle)
#         dcat = dcat[iShuffle]
#         prediction = prediction[iShuffle]
#     return dcat, prediction
