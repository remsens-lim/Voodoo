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
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as kmodel
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization, LSTM, Conv2D, MaxPool2D, Flatten, LeakyReLU, ReLU, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard



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

def add_activation(model, ACTIVATION):
    """
    Adds an activatin layer to a keras model.
    Args:
        -   ACTIVATION (string) : name of the activation function

    Return:
        -   model (tensorflow object) : the model
    """
    if ACTIVATION == 'leakyrelu':
        model.add(LeakyReLU(alpha=.001))
    elif ACTIVATION == 'relu':
        model.add(ReLU())
    else:
        model.add(Activation=ACTIVATION)
    return model

def define_dense(n_input, n_output, hyper_params):
    DENSE_LAYERS = hyper_params['DENSE_LAYERS']
    LAYER_SIZE = hyper_params['LAYER_SIZES']
    ACTIVATION = hyper_params['ACTIVATIONS']
    LOSSES = hyper_params['LOSS_FCNS']
    model_path = hyper_params['MODEL_PATH']
    OPTIMIZER = hyper_params['OPTIMIZER']

    if os.path.exists(model_path):
        # load model
        model = keras.models.load_model(model_path)
        print(f'Loaded model from disk {model_path}')
    else:
        # create the model and add the input layer
        model = Sequential()
        #model.add(LSTM(LAYER_SIZE, input_shape=(n_input,), return_sequences=True))
        model.add(Dense(LAYER_SIZE, input_shape=(n_input,)))
        model.add(BatchNormalization())
        model = add_activation(model, ACTIVATION)

        # add more dense layers
        for idl in range(DENSE_LAYERS - 1):
            #model.add(LSTM(LAYER_SIZE))
            model.add(Dense(LAYER_SIZE))
            model.add(BatchNormalization())
            model = add_activation(model, ACTIVATION)

        # define output layer containing 2 nodes (backscatter and depolarization)
        model.add(Dense(n_output, activation='linear'))
        print(f"Created model {model_path}")

    if OPTIMIZER == 'sgd':
        opt = SGD(lr=1e-2, momentum=0.5, decay=1.e-5)
    else:
        opt = Adam(lr=1e-4, decay=1.e-6)

    model.compile(optimizer=opt,
                  loss=LOSSES,
                  metrics=['mae', 'mse', 'msle']
                  )

    model.summary()

    return model



def define_cnn(n_input, n_output, hyper_params):
    CONV_LAYERS = hyper_params['CONV_LAYERS']
    DENSE_LAYERS = hyper_params['DENSE_LAYERS']
    DENSE_NODES = hyper_params['DENSE_NODES'] if 'DENSE_NODES' in hyper_params else 0
    NFILTERS = hyper_params['NFILTERS']
    KERNEL_SIZE = hyper_params['KERNEL_SIZE']
    POOL_SIZE = hyper_params['POOL_SIZE']
    ACTIVATION = hyper_params['ACTIVATIONS']
    ACTIVATION_OL = hyper_params['ACTIVATION_OL'] if 'ACTIVATION_OL' in hyper_params else 'linear'
    LOSSES = hyper_params['LOSS_FCNS']
    model_path = hyper_params['MODEL_PATH']
    OPTIMIZER = hyper_params['OPTIMIZER']
    learning_rate = 1.e-3
    decay_rate = learning_rate * 1.e-3
    momentum = 0.9

    if os.path.exists(model_path):
        # load model
        model = keras.models.load_model(model_path)
        print(f'Loaded model from disk {model_path}')
    else:
        # create the model and add the input layer
        model = Sequential()
        model.add(Conv2D(NFILTERS[0], KERNEL_SIZE, input_shape=n_input, padding="same"))
        model = add_activation(model, ACTIVATION)
        model.add(MaxPool2D(pool_size=POOL_SIZE))

        # add more conv layers
        for idl in range(CONV_LAYERS - 1):
            model.add(Conv2D(NFILTERS[idl+1], KERNEL_SIZE, padding="same"))
            model = add_activation(model, ACTIVATION)
            model.add(MaxPool2D(pool_size=POOL_SIZE))

        # define output layer containing 2 nodes (backscatter and depolarization)
        model.add(Flatten())

        for idense in range(DENSE_LAYERS):
            model.add(Dense(DENSE_NODES[idense]))
            model = add_activation(model, ACTIVATION)

        model.add(Dense(n_output[0], activation=ACTIVATION_OL))
        print(f"Created model {model_path}")


    if OPTIMIZER == 'sgd':
        opt = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
    else:
        opt = Adam(lr=learning_rate, decay=decay_rate)

    model.compile(optimizer=opt, loss=LOSSES, metrics=['mae', 'mse'])

    model.summary()

    return model


def training(model, train_set, train_label, hyper_params):
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())

    BATCH_SIZE = hyper_params['BATCH_SIZE']
    EPOCHS     = hyper_params['EPOCHS']
    LOG_PATH   = hyper_params['LOG_PATH']
    MODEL_PATH = hyper_params['MODEL_PATH']
    DEVICE     = hyper_params['DEVICE'] if 'DEVICE' in hyper_params else 0

    # log model training to tensorboard callback
    tensorboard_callback = TensorBoard(log_dir=LOG_PATH,
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_images=True)

    #with tf.device(f'/gpu:0'):
    with tf.device(f'/gpu:{DEVICE}'):
        history = model.fit(train_set, train_label,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            shuffle=True,
                            callbacks=[tensorboard_callback],
                            validation_split=0.0,
                            #callbacks=[PrintDot()],
                            verbose=1
                            )

        # serialize model to HDF5
        model.save(MODEL_PATH)
        print(f"Saved model to disk :: {MODEL_PATH}")

    return model, history

def predict_lidar(loaded_model, test_set):
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