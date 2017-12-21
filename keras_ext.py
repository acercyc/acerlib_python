# 1.0 - Acer 2017/05/17 17:06
# 2.0 - Acer 2017/11/24 15:51
import copy
import os
import time
from collections import Iterable
from functools import reduce
from subprocess import Popen

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import load_model
from keras.utils import vis_utils

import acerlib.shelve_ext as she
from acerlib import print_ext


# ============================================================================ #
#                               Common Functions                               #
# ============================================================================ #
def check_and_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ============================================================================ #
#                              TenserFlow Control                              #
# ============================================================================ #
def setTensorFlowGpuMemory(ratio):
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=ratio)

    if num_threads:
        sessionInfo = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        sessionInfo = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    KTF.set_session(sessionInfo)


# ============================================================================ #
#                                     Plot                                     #
# ============================================================================ #
def plot_model(m, filename='temp_modelplot.png'):
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    vis_utils.plot_model(m, show_shapes=True, to_file=filename)
    filename = 'temp_modelplot.png'

    img = mpimg.imread(filename)
    try:
        Popen(['eog', filename])
    except:
        plt.imshow(img)

        # --- back up ---
        # img = Image.open(fName)
        # img.show()
        # --------------


# ============================================================================ #
#                             Command line display                             #
# ============================================================================ #
def print_trainProgress(iEpoch, iBatch, loss):
    print('Epoch', iEpoch, 'Batch', iBatch, 'loss =', loss, sep='\t')


def print_evalProgress(iEpoch, iBatch, loss):
    print('Epoch', iEpoch, 'Batch', iBatch, 'val_loss =', loss, sep='\t')


def print_loss(iEpoch, iBatch, loss, numFormat='.3e'):
    """
    :param iEpoch: 
    :param iBatch: 
    :param loss: can be number or dict
    :param numFormat: 
    :return: 
    """
    print('Epoch %d  Batch %d' % (iEpoch, iBatch), end="")
    print('\t', end="")
    if isinstance(loss, dict):
        print_ext.print_dict(loss, numFormat)
        print('')
    elif isinstance(loss, Iterable):
        print('loss: ', end='')

        for x in loss:
            print('%.3e\t' % x, end='')
        print('')
    else:
        print(('loss: %' + '.3e') % loss)


# ============================================================================ #
#                                Data Processing                               #
# ============================================================================ #
def batchGenerator(d, batch_size, isLooping=True):
    """
    Make batch generator
    :param d: multidimantional data
    :param batch_size: 
    :param isLooping: 
    """
    nSample = d[0].shape[0] if isinstance(d, (list, tuple)) else d.shape[0]
    c1 = 0
    while True:
        if isinstance(d, (list, tuple)):
            d_batch = [np.take(x, range(c1, c1 + batch_size), axis=0) for x in d]
        else:
            d_batch = np.take(d, range(c1, c1 + batch_size), axis=0)
        yield d_batch, range(c1, c1 + batch_size)

        if (c1 + 2 * batch_size) > nSample:
            if isLooping:
                c1 = 0
            else:
                break
        else:
            c1 += batch_size


# ============================================================================ #
#                              Model Construction                              #
# ============================================================================ #
def stackLayers(layers):
    return reduce((lambda x, y: y(x)), layers)


# ============================================================================ #
#                                  Controller                                  #
# ============================================================================ #
class ModelController:
    def __init__(self, m, path='pipeline_temp', ID=None):
        # create data folder
        self.m = m
        self.path = path
        self.history = {'history': [], 'loss_train': [], 'loss_valid': []}
        self.loss_best = np.inf

        if ID is None:
            self.id = 'p' + time.strftime("%Y%d%d_%H%M%S")
        else:
            self.id = ID

        check_and_create_path(path)

    def save_m(self, fName=None):
        if fName is None:
            fName = os.path.join(self.path, '%s_m' % self.id)
        self.m.save(fName)
        print('model saved')

    def save_m_best(self, newLoss):
        if newLoss < self.loss_best:
            self.loss_best = newLoss
            fName = os.path.join(self.path, '%s_m_best' % self.id)
            self.m.save(fName)
            print('New model saved')
            isSaveNew = 1
        else:
            isSaveNew = 0
        return isSaveNew

    def plot_m(self):
        filename = os.path.join(self.path, '%s_plot_m.png' % self.id)
        plot_model(self.m, filename)

    def genDefaultCallbacks(self, isSaveBestModel=True, isLogCSV=True, isSaveModel=True):
        callbacks = []

        # save the best model
        if isSaveBestModel:
            fName = os.path.join(self.path, '%s_CModelCheckpoint_best.hdf5' % self.id)
            cb_ModelCheckpoint_best = ModelCheckpoint(fName, monitor='val_loss', save_best_only=True)
            callbacks.append(cb_ModelCheckpoint_best)

        # log history
        if isLogCSV:
            fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.id)
            cb_CSVLogger = CSVLogger(fName)
            callbacks.append(cb_CSVLogger)

        # save all model history
        if isSaveModel:
            pathName = os.path.join(self.path, 'model_history_%s' % self.id)
            check_and_create_path(pathName)
            fName = os.path.join(pathName, '%s_CModelCheckpoint_{epoch:04d}.hdf5' % self.id)
            cb_ModelCheckpoint = ModelCheckpoint(fName, monitor='val_loss')
            callbacks.append(cb_ModelCheckpoint)

        return callbacks


class Pipeline:
    # ============================================================================ #
    #                                  Initialise                                  #
    # ============================================================================ #
    def __init__(self, path='pipeline_temp', ID=None):
        if ID is None:
            ID = 'p' + time.strftime("%Y%d%d_%H%M%S")
        self.ID = ID
        # create data folder
        self.path = path
        self.check_and_create_path()

        # data
        self.d_train = None  # should be a list [X, Y]
        self.d_test = None  # should be a list [X, Y]
        self.d_valid = None  # should be a list [X, Y]

        # model
        self.m = None

        # fitting
        self.history = None

    # ============================================================================ #
    #                             High Level Functions                             #
    # ============================================================================ #
    # Training ------------------------------------------------------------------- #
    def fit(self, batch_size=32, epochs=10, callbacks=None, validation_split=0.0, useValidation_data=True):
        if callbacks is None:
            callbacks = self.defaultCallbacks()
        if useValidation_data:
            d_valid = self.d_valid
        else:
            d_valid = None
        self.history = self.m.fit(self.d_train[0], self.d_train[1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  callbacks=callbacks,
                                  validation_split=validation_split,
                                  validation_data=d_valid)

    # ============================================================================ #
    #                                     Plot                                     #
    # ============================================================================ #
    def plot_m(self):
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        fName = os.path.join(self.path, '%s_plot_m.png' % self.ID)
        vis_utils.plot_model(self.m, show_shapes=True, to_file=fName)
        img = mpimg.imread(fName)
        try:
            Popen(['eog', fName])
        except:
            plt.imshow(img)

    def plot_history(self):
        import matplotlib.pyplot as plt

        fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.ID)
        history = np.loadtxt(fName, skiprows=1, delimiter=',')
        plt.ion()
        plt.figure(figsize=(13, 5))
        plt.plot(history[:, 1], label="training")
        plt.plot(history[:, 2], label="validation")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    # ============================================================================ #
    #                                   callback                                   #
    # ============================================================================ #
    def defaultCallbacks(self):

        # save the best model
        fName = os.path.join(self.path, '%s_CModelCheckpoint_best.hdf5' % self.ID)
        cb_ModelCheckpoint_best = ModelCheckpoint(fName, monitor='val_loss', save_best_only=True)

        # log history
        fName = os.path.join(self.path, '%s_CSVLogger.csv' % self.ID)
        cb_CSVLogger = CSVLogger(fName)

        # save all model history
        pathName = os.path.join(self.path, 'model_history_%s' % self.ID)
        self.check_and_create_path(pathName)

        fName = os.path.join(pathName, '%s_CModelCheckpoint_{epoch:04d}.hdf5' % self.ID)
        cb_ModelCheckpoint = ModelCheckpoint(fName, monitor='val_loss')

        return [cb_ModelCheckpoint_best, cb_CSVLogger, cb_ModelCheckpoint]

    # ============================================================================ #
    #                                   File I/O                                   #
    # ============================================================================ #
    def load(self, m=True, d_train=True, d_test=True, d_valid=True):
        funMappting = {'m': [m, self.load_m],
                       'd_train': [d_train, self.load_d_train],
                       'd_test': [d_test, self.load_d_test],
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

    def save(self, m=True, d_train=True, d_test=True, d_valid=True):
        funMappting = {'m': [m, self.save_m],
                       'd_train': [d_train, self.save_d_train],
                       'd_test': [d_test, self.save_d_test],
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
                       'd_test': [d_test, self.read_d_test],
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
        fName = os.path.join(self.path, '%s_d_train.npz' % self.ID)
        np.savez(fName, *self.d_train)
        print('training data saved')

    def save_d_valid(self):
        fName = os.path.join(self.path, '%s_d_valid.npz' % self.ID)
        np.savez(fName, *self.d_valid)
        print('validation data saved')

    def save_d_test(self):
        fName = os.path.join(self.path, '%s_d_test.npz' % self.ID)
        np.savez(fName, *self.d_test)
        print('testing data saved')

    def save_m(self):
        fName = os.path.join(self.path, '%s_m' % self.ID)
        self.m.save(fName)
        print('model saved')

    def save_pipeline(self):
        ps = copy.copy(self)
        ps.d_train = []
        ps.d_test = []
        ps.d_valid = []
        ps.m = []

        fName = os.path.join(self.path, '%s_pipeline' % self.ID)
        she.save(fName, 'pipeline', ps)
        print('Pipeline saved')

    # load ----------------------------------------------------------------------- #
    def load_d_train(self):
        fName = os.path.join(self.path, '%s_d_train.npz' % self.ID)
        d = np.load(fName)
        d.files.sort()
        self.d_train = [d[vName] for vName in d.files]
        print('trainig data loaded')

    def load_d_test(self):
        fName = os.path.join(self.path, '%s_d_test.npz' % self.ID)
        d = np.load(fName)
        d.files.sort()
        self.d_test = [d[vName] for vName in d.files]
        print('testing data loaded')

    def load_d_valid(self):
        fName = os.path.join(self.path, '%s_d_valid.npz' % self.ID)
        d = np.load(fName)
        d.files.sort()
        self.d_valid = [d[vName] for vName in d.files]
        print('validation data loaded')

    def load_m(self):
        fName = os.path.join(self.path, '%s_m' % self.ID)
        load_model(fName)
        print('model loaded')

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
