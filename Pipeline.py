# ============================================================================ #
#                     A class for keras pipeline processing                    #
# ============================================================================ #
# Need to specify path when initialising a new pipeline object

# 1.0 - Acer 2017/02/14 18:58
# 2.0 - Acer 2017/02/15 19:04
# 3.0 - Acer 2017/02/17 15:58
# 3.1 - Acer 2017/03/01 16:22
# 3.2 - Acer 2017/04/26 16:02
# 3.3 - Acer 2017/04/28 16:56
# 3.4 - Acer 2017/05/01 20:41
# 3.5 - Acer 2017/05/15 12:16


import os
from subprocess import Popen
import time
import inspect
import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.utils.vis_utils import plot_model

import acerlib.shelve_ext as she
from acerlib import sequence


class Pipeline:

    # pipeline info
    path  = None
    paras = None
    id    = 'p' + time.strftime("%Y%d%d_%H%M%S")

    # data
    d_train    = None  # should be a list [X, Y]
    d_test     = None  # should be a list [X, Y]
    d_valid    = None  # should be a list [X, Y]
    d_pred     = None  # only Y

    # data reading functions
    # ---------------------------------------------------------------------------- #
    # reading function needs to return [X, Y]
    f_read_d_train = None
    f_read_d_test  = None
    f_read_d_valid = None

    # data processing
    f_preprocessing    = None
    f_revPreprocessing = None

    # model
    m     = None
    f_m   = None  # function which create a model and need output (m, locals())
    m_env = None

    # fitting
    paras_fit = {'epochs': 10,
                 'batch_size': 32,
                 'callbacks': None}
    history   = None

    # prediction
    paras_pred = {}

    # evaluation
    result = None
    f_eval = None  # result = f_eval(y_true, y_pred)

    # =================================== init =================================== #
    def __init__(self, path='pipeline_temp'):
        # create data folder
        self.path = path
        self.check_and_create_path()

    # ============================================================================ #
    #                              Top Level Functions                             #
    # ============================================================================ #
    def run(self):
        self.load()
        self.genModel()
        self.fit()
        self.save_m()
        self.predict()
        self.save_d_pred()
        self.evaluation()

    # ============================================================================ #
    #                             High Level Functions                             #
    # ============================================================================ #
    def preproessing(self, d):
        if self.f_preprocessing is None:
            raise Exception('Not define preprocessing function yet')
        return self.f_preprocessing(d)

    def revPreprocessing(self, d):
        if self.f_revPreprocessing is None:
            raise Exception('Not define preprocessing function yet')
        return self.f_revPreprocessing(d)

    # Create model --------------------------------------------------------------- #
    def genModel(self, m_function=None):
        if m_function is not None:
            self.f_m = m_function
        self.m, self.m_env = self.f_m()

    # Training ------------------------------------------------------------------- #
    def _fitPreparation(self, use_d_valid=True, addDefaultCallback=True):
        # create folder to save all data
        self.check_and_create_path()

        # arrange fit() parameters
        paras_fit = self.paras_fit.copy()
        if use_d_valid:
            paras_fit['validation_data'] = self.d_valid

        # add callbacks
        if addDefaultCallback:
            if paras_fit['callbacks'] is None:
                paras_fit['callbacks'] = self.defaultCallbacks()
            else:
                paras_fit['callbacks'].append(self.defaultCallbacks())
        return paras_fit

    def fit(self, use_d_valid=True, addDefaultCallback=True):
        paras_fit = self._fitPreparation(use_d_valid, addDefaultCallback)
        self.history = self.m.fit(self.d_train[0], self.d_train[1], **paras_fit)

    def fit_generator(self, generator, steps_per_epoch, use_d_valid=True, addDefaultCallback=True):
        paras_fit = self._fitPreparation(use_d_valid, addDefaultCallback)
        paras_fit.pop('batch_size')
        self.history = self.m.fit_generator(generator, steps_per_epoch, **paras_fit)

    # Predict and evaluation ----------------------------------------------------- #
    def predict(self):
        self.d_pred = self.m.predict(self.d_test[0], **self.paras_pred)

    def evaluation(self):
        if self.f_eval is None:
            raise Exception('Not define evaluation function yet')
        self.result = self.f_eval(self.d_test[1], self.d_pred)
        return self.result

    # ============================================================================ #
    #                                     Plot                                     #
    # ============================================================================ #
    def plot_m(self):
        fName = os.path.join(self.path, '%s_plot_m.png' % self.id)
        plot_model(self.m, show_shapes=True, to_file=fName)
        img = mpimg.imread(fName)
        try:
            Popen(['eog', fName])
        except:
            plt.imshow(img)

    def plot_history(self):
        fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.id)
        history = np.loadtxt(fName, skiprows=1, delimiter=',')
        plt.ion()
        plt.figure(figsize=(13, 5))
        plt.plot(history[:, 1], label="training")
        plt.plot(history[:, 2], label="validation")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # --- back up ---
        # h = self.history.history
        # plt.figure()
        # measures = list(h.keys())
        # hData = np.array(list(h.values())).T
        # plt.plot(hData)
        # plt.legend(measures)

    # ============================================================================ #
    #                             Pipeline Information                             #
    # ============================================================================ #
    def print_m_function(self):
        print(inspect.getsource(self.f_m))

    @staticmethod
    def print_function_source(fun):
        f_str = inspect.getsource(fun)
        print(f_str)
        return f_str

    # ============================================================================ #
    #                                   callback                                   #
    # ============================================================================ #
    def defaultCallbacks(self):

        # save the best model
        fName = os.path.join(self.path, '%s_CModelCheckpoint_best.hdf5' % self.id)
        cb_ModelCheckpoint_best = ModelCheckpoint(fName, monitor='val_loss', save_best_only=True)

        # log history
        fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.id)
        cb_CSVLogger = CSVLogger(fName)

        # save all model history
        pathName = os.path.join(self.path, 'model_history_%s' % self.id)
        self.check_and_create_path(pathName)
        fName = os.path.join(pathName, '%s_CModelCheckpoint_{epoch:04d}.hdf5' % self.id)
        cb_ModelCheckpoint = ModelCheckpoint(fName, monitor='val_loss')

        return [cb_ModelCheckpoint_best, cb_CSVLogger, cb_ModelCheckpoint]

    # ============================================================================ #
    #                                   File I/O                                   #
    # ============================================================================ #
    def load(self, m=True, d_train=True, d_test=True, d_pred=True, d_valid=True):
        funMappting = {'m':       [m,       self.load_m],
                       'd_train': [d_train, self.load_d_train],
                       'd_test':  [d_test,  self.load_d_test],
                       'd_pred':  [d_pred,  self.load_d_pred],
                       'd_valid': [d_valid, self.load_d_valid]}
        for key, fun in funMappting.items():
            try:
                if fun[0]:
                    fun[1]()
                    print('')
            except Exception as e:
                print(e)
                print(key + ': not loaded\n')
        print('')

    def save(self, m=True, d_train=True, d_test=True, d_pred=True, d_valid=True):
        funMappting = {'m':       [m,       self.save_m],
                       'd_train': [d_train, self.save_d_train],
                       'd_test':  [d_test,  self.save_d_test],
                       'd_pred':  [d_pred,  self.save_d_pred],
                       'd_valid': [d_valid, self.save_d_valid]}
        for key, fun in funMappting.items():
            try:
                if fun[0]:  # if data exist, then save
                    if getattr(self, key) is not None:
                        fun[1]()  # run save funciton
                    print('')
            except Exception as e:
                print(e)
                print(key + ': not saved\n')
        print('')
        self.save_pipeline()

    def read_d(self, d_train=True, d_test=True, d_valid=True):
        funMappting = {'d_train': [d_train, self.read_d_train],
                       'd_test':  [d_test,  self.read_d_test],
                       'd_valid': [d_valid, self.read_d_valid]}
        for key, fun in funMappting.items():
            try:
                if fun[0]:
                    fun[1]()
                    print('')
            except Exception as e:
                print(e)
                print(key + ': not read\n')
        print('')

    # ============================================================================ #
    #                              Low-level File I/O                              #
    # ============================================================================ #

    # save ----------------------------------------------------------------------- #
    def save_d_train(self):
        fName = os.path.join(self.path, '%s_d_train.npz' % self.id)
        np.savez(fName, *self.d_train)
        print('training data saved')

    def save_d_test(self):
        fName = os.path.join(self.path, '%s_d_test.npz' % self.id)
        np.savez(fName, *self.d_test)
        print('testing data saved')

    def save_d_pred(self):
        fName = os.path.join(self.path, '%s_d_pred.npz' % self.id)
        np.savez(fName, *self.d_pred)
        print('predicted data saved')

    def save_d_valid(self):
        fName = os.path.join(self.path, '%s_d_valid.npz' % self.id)
        np.savez(fName, *self.d_valid)
        print('validation data saved')

    def save_m(self):
        fName = os.path.join(self.path, '%s_m' % self.id)
        self.m.save(fName)
        print('model saved')

    def save_pipeline(self):
        ps = copy.copy(self)
        ps.d_train = []
        ps.d_train = []
        ps.d_pred = []
        ps.d_valid = []
        ps.m = []

        fName = os.path.join(self.path, '%s_pipeline' % self.id)
        she.save(fName, 'pipeline', ps)
        print('Pipeline saved')

    # load ----------------------------------------------------------------------- #
    def load_d_train(self):
        fName = os.path.join(self.path, '%s_d_train.npz' % self.id)
        d = np.load(fName)
        d.files.sort()
        self.d_train = [d[vName] for vName in d.files]
        print('trainig data loaded')

    def load_d_test(self):
        fName = os.path.join(self.path, '%s_d_test.npz' % self.id)
        d = np.load(fName)
        d.files.sort()
        self.d_test = [d[vName] for vName in d.files]
        print('testing data loaded')

    def load_d_pred(self):
        fName = os.path.join(self.path, '%s_d_pred.npz' % self.id)
        d = np.load(fName)
        d.files.sort()
        self.d_pred = [d[vName] for vName in d.files]
        print('predicted data loaded')

    def load_d_valid(self):
        fName = os.path.join(self.path, '%s_d_valid.npz' % self.id)
        d = np.load(fName)
        d.files.sort()
        self.d_valid = [d[vName] for vName in d.files]
        print('validation data loaded')

    def load_m(self):
        fName = os.path.join(self.path, '%s_m' % self.id)
        load_model(fName)
        print('model loaded')

    # read ----------------------------------------------------------------------- #
    def read_d_train(self):
        """
        Execute training set reading function
        Training set reading function should return [X, Y]
        """
        if self.f_read_d_train is None:
            raise Exception('Not define traing set reading function yet')
        self.d_train = self.f_read_d_train()
        print('trainig data: read')

    def read_d_test(self):
        """
        Execute testing set reading function
        Testing set reading function should return [X, Y]
        """
        if self.f_read_d_test is None:
            raise Exception('Not define testing set reading function yet')
        self.d_test = self.f_read_d_test()
        print('testing data: read')

    def read_d_valid(self):
        """
        Execute validation set reading function
        Validation set reading function should return [X, Y]
        """
        if self.f_read_d_valid is None:
            raise Exception('Not define validation set reading function yet')
        self.d_valid = self.f_read_d_valid()
        print('validation data: read')

    # ============================================================================ #
    #                                   Utilities                                  #
    # ============================================================================ #
    def check_and_create_path(self, path=None):
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)


def load_pipeline(fName):
    p = she.load(fName, 'pipeline')
    p.load()
    return p


def load_pipeline_withBestModel(fName):
    p = she.load(fName, 'pipeline')
    p.load()
    fName = os.path.join(p.path, '%s_CModelCheckpoint_best.hdf5' % p.id)
    p.m = load_model(fName)
    return p


def toBatchGenerator(f):
    """
    This decorator wraps a adata processing function that process a subset of the data 
    :param f: wrappered function in which the first input is the to be processed subset data
    :return: wrapper: a wrappered generator function gf(dataset, batchSize, ...)
        dataset: whole dataset
        batchSize: 
    Example: 
        @toBatchGenerator
        def f(d_batch, para):
            return xsub + para
        x = np.arange(4)
        for i in f(d,  2, 1):
            print(i)        

    2.0 - Acer 2017/05/14 14:22
    """
    def wrapper(d, batch_size, *args, **kwargs):
        nSample = d[0].shape[0] if isinstance(d, (list, tuple)) else d.shape[0]
        c1 = 0
        while True:
            if isinstance(d, (list, tuple)):
                d_batch = [np.take(x, range(c1, c1+batch_size), axis=0) for x in d]
            else:
                d_batch = np.take(d, range(c1, c1+batch_size), axis=0)
            c1 = 0 if (c1 + 2*batch_size) > nSample else c1 + batch_size
            yield f(d_batch,  *args, **kwargs)

    return wrapper


# ---------------------------------- Backup ---------------------------------- #
# # input
# batch_size = 32
# epochs = 2
# d = d_train
# d_valid
#
#
# def f_train(d_batch, m):
#     x, y = d_batch
#     loss = m.train_on_batch(x, y)
#     return loss
#
#
# # ----------------------------------------------------------------------------
# def fit_constomCallLoop(c, *args, **kwargs):
#     """
#     :param c:  a callable object e.g. function or class which execute training code when it is called
#     !!! The first input of c should be input data !!!
#     :return: loss
#     """
#     nSample = d[0].shape[0] if isinstance(d, (list, tuple)) else d.shape[0]
#     indexes = list(sequence.movingWindow(0, nSample, batch_size))
#     for iEpoch in range(epochs):
#         for iBatch, iSample in enumerate(indexes):
#             if isinstance(d, (list, tuple)):
#                 d_batch = [np.take(x, iSample, axis=0) for x in d]
#             else:
#                 d_batch = np.take(d, iSample, axis=0)
#
#             # train and update
#             loss = c(d_batch, *args, **kwargs)
#             print('Batch =', iBatch, 'loss =', loss, sep='\t')
#
#         # evaluation
#         loss = c(d_batch, *args, **kwargs)
#
#         # callback

# ---------------------------------------------------------------------------- #

class modelController:
    # pipeline info
    path  = None
    id    = None
    loss  = None
    loss_best = np.inf

    def __init__(self, m, path='pipeline_temp', ID=None):
        # create data folder
        self.m = m
        self.path = path
        self.check_and_create_path()
        if ID is None:
            self.id = 'p' + time.strftime("%Y%d%d_%H%M%S")
        else:
            self.id = ID

    def check_and_create_path(self, path=None):
        if path is None:
            path = self.path
        if not os.path.exists(path):
            os.makedirs(path)

    def save_m(self):
        fName = os.path.join(self.path, '%s_m' % self.id)
        self.m.save(fName)
        print('model saved')

    def save_m_best(self, newLoss):
        if newLoss < self.loss_best:
            self.loss_best = newLoss
            fName = os.path.join(self.path, '%s_m_best' % self.id)
            self.m.save(fName)
            print('model saved')

    # def wrtie_loss(self, newLoss):
    #     fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.id)
    #     with open(fName, "a") as myfile:
    #         myfile.write("appended text", )

    def plot_m(self):
        fName = os.path.join(self.path, '%s_plot_m.png' % self.id)
        plot_model(self.m, show_shapes=True, to_file=fName)
        img = mpimg.imread(fName)
        try:
            Popen(['eog', fName])
        except:
            plt.imshow(img)