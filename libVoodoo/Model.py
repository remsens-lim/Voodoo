"""
This module contains functions for generating deep learning models with Tensorflow and Keras.

"""

import copy
import os
import sys
import numpy as np

# neural network imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.activations import softmax, relu, sigmoid
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Model as kmodel
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, LSTM, Conv2D, MaxPool2D, Flatten, LeakyReLU, ReLU, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K

from tensorflow.python import debug as tf_debug

tf.compat.v1.keras.backend.set_session(
    tf_debug.TensorBoardDebugWrapperSession(tf.compat.v1.Session(), "sdig-workstation:6006"))

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'

#sys.path.append('../../larda/')
#sys.path.append('.')

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('INFO')

__author__      = "Willi Schimmel"
__copyright__   = "Copyright 2019, The Voodoo Project"
__credits__     = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__     = "MIT"
__version__     = "0.0.1"
__maintainer__  = "Willi Schimmel"
__email__       = "willi.schimmel@uni-leipzig.de"
__status__      = "Prototype"


########################################################################################################################
########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
########################################################################################################################
########################################################################################################################
class LogEpochScores(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LogEpochScores, self).__init__()

    def on_train_begin(self, logs=None):
        self.model.epoch_log = []

    def on_epoch_end(self, epoch, logs=None):
        self.model.epoch_log.append(logs)


def define_autoencoder(n_input, hyper_params):
    ACTIVATION = hyper_params['activations']
    LOSSES = hyper_params['loss_fcns']
    model_path = hyper_params['model_path']

    if os.path.exists(model_path):
        # load model
        model = keras.models.load_model(model_path)
        print(f'Loaded model from disk {model_path}')
    else:
        # create the model and add the input layer
        model = Sequential()
        model.add(Dense(128, input_shape=(n_input,)))
        # model.add(BatchNormalization())
        model.add(Activation(ACTIVATION))
        # model.add(CuDNNLSTM(layer_size, input_shape=(n_input,), return_sequences=True))

        model.add(Dense(56))
        model.add(Activation(ACTIVATION))
        model.add(Dropout(0.1))

        model.add(Dense(128))
        model.add(Activation(ACTIVATION))
        model.add(Dropout(0.1))

        # define output layer containing 2 nodes (backscatter and depolarization)
        model.add(Dense(256, activation='linear'))
        print(f"Created model {model_path}")

    opt = Adam(lr=1e-3, decay=1.e-4)

    model.compile(optimizer=opt,
                  loss=LOSSES,
                  metrics=['mae', 'mse', 'mape', 'msle']
                  # loss='mean_squared_error',
                  # metrics=['mean_absolute_error', 'mean_squared_error']
                  )

    model.summary()

    return model

class XTensorBoard(TensorBoard):
    def on_epoch_begin(self, epoch, logs=None):
        # get values
        lr = float(K.get_value(self.model.optimizer.lr))
        decay = float(K.get_value(self.model.optimizer.decay))
        # computer lr
        lr = lr * (1. / (1 + decay * epoch))
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)

def define_cnn(n_input, n_output, MODEL_PATH='', **hyper_params):

    if os.path.exists(MODEL_PATH):
        # load model
        model = keras.models.load_model(MODEL_PATH)
        print(f'Loaded model from disk {MODEL_PATH}')
    else:
        CONV_LAYERS = hyper_params['CONV_LAYERS'] if 'CONV_LAYERS' in hyper_params else ValueError('CONV_LAYERS missing!')
        DENSE_LAYERS = hyper_params['DENSE_LAYERS'] if 'DENSE_LAYERS' in hyper_params else ValueError('DENSE_LAYERS missing!')
        DENSE_NODES = hyper_params['DENSE_NODES'] if 'DENSE_NODES' in hyper_params else ValueError('DENSE_NODES missing!')
        NFILTERS = hyper_params['NFILTERS'] if 'NFILTERS' in hyper_params else ValueError('NFILTERS missing!')
        KERNEL_SIZE = hyper_params['KERNEL_SIZE'] if 'KERNEL_SIZE' in hyper_params else ValueError('KERNEL_SIZE missing!')
        POOL_SIZE = hyper_params['POOL_SIZE'] if 'POOL_SIZE' in hyper_params else ValueError('POOL_SIZE missing!')
        ACTIVATION = hyper_params['ACTIVATIONS'] if 'ACTIVATIONS' in hyper_params else ValueError('ACTIVATIONS missing!')
        ACTIVATION_OL = hyper_params['ACTIVATION_OL'] if 'ACTIVATION_OL' in hyper_params else 'softmax'
        BATCH_NORM = hyper_params['BATCH_NORM'] if 'BATCH_NORM' in hyper_params else False
        DROPOUT = hyper_params['DROPOUT'] if 'DROPOUT' in hyper_params else -1.0

        # create the model and add the input layer
        initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)
        model = Sequential()
        model.add(Conv2D(NFILTERS[0], KERNEL_SIZE, activation=ACTIVATION, input_shape=n_input, padding="same", kernel_initializer=initializer))
        if BATCH_NORM: model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=POOL_SIZE))

        # add more conv layers
        for idl in range(CONV_LAYERS - 1):
            model.add(Conv2D(NFILTERS[idl+1], KERNEL_SIZE, activation=ACTIVATION, padding="same"))
            model.add(MaxPool2D(pool_size=POOL_SIZE))
            if BATCH_NORM: model.add(BatchNormalization())

        model.add(Flatten())

        for idense in range(DENSE_LAYERS):
            model.add(Dense(DENSE_NODES[idense], activation=ACTIVATION))
            if BATCH_NORM:    model.add(BatchNormalization())
            if DROPOUT > 0.0: model.add(Dropout(DROPOUT))

        model.add(Dense(n_output[0], activation=ACTIVATION_OL))
        print(f"Created model {MODEL_PATH}")

    LOSSES = hyper_params['LOSS_FCNS'] if 'LOSS_FCNS' in hyper_params else ValueError('LOSS_FCNS missing!')
    OPTIMIZER = hyper_params['OPTIMIZER'] if 'OPTIMIZER' in hyper_params else ValueError('OPTIMIZER missing!')
    beta_1 = hyper_params['beta_1'] if 'beta_1' in hyper_params else 0.9
    beta_2 = hyper_params['beta_2'] if 'beta_2' in hyper_params else 0.999
    learning_rate = hyper_params['LEARNING_RATE'] if 'LEARNING_RATE' in hyper_params else 1.e-4
    decay_rate = hyper_params['DECAY_RATE'] if 'DECAY_RATE' in hyper_params else learning_rate * 1.e-3
    momentum = hyper_params['MOMENTUM'] if 'MOMENTUM' in hyper_params else 0.9

    if OPTIMIZER == 'sgd':
        opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
    elif OPTIMIZER == 'Nadam':
        opt = Nadam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    elif OPTIMIZER == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate, rho=momentum)
    else:
        opt = Adam(lr=learning_rate, decay=decay_rate)

    if LOSSES == 'BinaryCrossentropy':
        loss = BinaryCrossentropy()
    elif LOSSES == 'CategoricalCrossentropy':
        loss = CategoricalCrossentropy()
    elif LOSSES == 'SparseCategoricalCrossentropy':
        loss = SparseCategoricalCrossentropy()
    else:
        raise ValueError('Unknown LOSS_FCNS!', LOSSES)

    model.compile(optimizer=opt, loss=loss, metrics=['CategoricalCrossentropy', 'CategoricalAccuracy'])
    model.summary()

    return model


def training(model, train_set, train_label, **hyper_params):
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    BATCH_SIZE = hyper_params['BATCH_SIZE'] if 'BATCH_SIZE' in hyper_params else ValueError('BATCH_SIZE missing!')
    EPOCHS     = hyper_params['EPOCHS'] if 'EPOCHS' in hyper_params else ValueError('EPOCHS missing!')
    LOG_PATH   = hyper_params['LOG_PATH'] if 'LOG_PATH' in hyper_params else ValueError('LOG_PATH missing!')
    MODEL_PATH = hyper_params['MODEL_PATH'] if 'MODEL_PATH' in hyper_params else ValueError('MODEL_PATH missing!')
    DEVICE     = hyper_params['DEVICE'] if 'DEVICE' in hyper_params else 0

    # log model training to tensorboard callback
    tensorboard_callback = TensorBoard(log_dir=LOG_PATH,
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_images=True)

    lr_callback = XTensorBoard(LOG_PATH)

    with tf.device(f'/gpu:{DEVICE}'):
        history = model.fit(train_set, train_label,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            shuffle=True,
                            callbacks=[tensorboard_callback],#, lr_callback],
                            validation_split=0.1,
                            #callbacks=[PrintDot()],
                            verbose=1
                            )

        # serialize model to HDF5
        model.save(MODEL_PATH)
        print(f"Saved model to disk :: {MODEL_PATH}")

    return history

def predict_lidar(loaded_model, test_set):
    return loaded_model.predict(test_set, verbose=1)

def predict_liquid(loaded_model, test_set):
    return loaded_model.predict(test_set, verbose=1)

#
def predict_spectra(loaded_model, test_set, dimensions):

    list_ts = dimensions['list_ts']
    list_rg = dimensions['list_rg']
    ts_radar = dimensions['ts_radar']
    rg_radar = dimensions['rg_radar']
    vel_radar = dimensions['rg_radar']
    spec_orig = dimensions['spec_container']
    system_info = dimensions['system_info']


    pred = loaded_model.predict(test_set) #, verbose=1)

    cnt = 0
    container = []
    for ic in range(len(spec_orig)):
        paraminfo = system_info
        pred_var = np.full((ts_radar.size, spec_orig[ic]['rg'].size, spec_orig[ic]['vel'].size), fill_value=-999.0)
        for iT, iR in zip(list_ts, list_rg):
            iT, iR = int(iT), int(iR)
            pred_var[iT, iR, :] = pred[cnt, :]
            # print(iT, iR, pred_list[cnt], pred_var[iT, iR])
            cnt += 1

        mask = np.full((ts_radar.size, spec_orig[ic]['rg'].size, spec_orig[ic]['vel'].size), fill_value=False)
        mask[pred_var <= -999.0] = True
        pred_var = np.ma.masked_less_equal(pred_var, -999.0)

        container.append({'dimlabel': ['time', 'range', 'vel'],
                          'filename': [],
                          'paraminfo': copy.deepcopy(paraminfo),
                          'rg_unit': paraminfo['rg_unit'],
                          'colormap': paraminfo['colormap'],
                          'var_unit': paraminfo['var_unit'],
                          'var_lims': paraminfo['var_lims'],
                          'system': 'autoencoder_output',
                          'name': paraminfo['paramkey'],
                          'rg': rg_radar.copy(),
                          'ts': ts_radar.copy(),
                          'vel': vel_radar.copy(),
                          'mask': mask,
                          'var': pred_var})

    return pred


# copy from https://github.com/jg-fisher/autoencoder/blob/master/ffae.py
class AutoEncoder:
    def __init__(self, encoding_dim=3):
        self.encoding_dim = encoding_dim
        r = lambda: np.random.randint(1, 3)
        self.x = np.array([[r(), r(), r()] for _ in range(1000)])
        print(self.x)

    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        encoded = Dense(self.encoding_dim, activation='relu')(inputs)
        model = kmodel(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        decoded = Dense(3)(inputs)
        model = kmodel(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=self.x[0].shape)
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = kmodel(inputs, dc_out)
        model.summary()
        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        self.model.compile(optimizer='sgd', loss='mse')
        log_dir = './logs/'
        tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       callbacks=[tbCallBack])

    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')