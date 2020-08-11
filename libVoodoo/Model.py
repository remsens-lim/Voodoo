"""
This module contains functions for generating deep learning models with Tensorflow and Keras.

"""

import os
from itertools import product

import numpy as np

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

import libVoodoo.Utils as util

# neural network imports
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Conv2D, MaxPool1D, MaxPool2D, Flatten, Input
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Model as kmodel
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam
from tensorflow.python import debug as tf_debug

#tf.compat.v1.keras.backend.set_session(
#    tf_debug.TensorBoardDebugWrapperSession(tf.compat.v1.Session(), "sdig-workstation:6006"))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Create 2 virtual GPUs with 1GB memory each
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[util.pick_gpu_lowest_memory()],
        #[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=None),
        # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=None)]
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=50000.)]
        )
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'

# sys.path.append('../../larda/')
# sys.path.append('.')

"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# tf.get_logger().setLevel('ERROR')

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def _get_regularizer(reg_string):
    if reg_string == 'l1':
        return tf.keras.regularizers.l1(l=0.1)
    elif reg_string == 'l2':
        return tf.keras.regularizers.l2(l=0.01)
    elif reg_string == 'l1_l2':
        return tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)
    else:
        return None

def _get_initializer(krnl_string):
    if krnl_string == 'random_normal':
        return tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)
    else:
        return None


def _get_optimizer(opt, lr, dr, mom, beta_1=None, beta_2=None):
    if opt == 'sgd':
        return SGD(lr=lr, momentum=mom, decay=dr)
    elif opt == 'Nadam':
        return Nadam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
    elif opt == 'rmsprop':
        return RMSprop(learning_rate=lr, rho=mom)
    elif opt == 'adam':
        return Adam(lr=lr, decay=dr)
    else:
        raise ValueError('Unknown OPTIMIZER!', opt)


def _get_loss_function(loss):
    if loss == 'BinaryCrossentropy':
        return BinaryCrossentropy()
    elif loss == 'CategoricalCrossentropy':
        return CategoricalCrossentropy()
    elif loss == 'SparseCategoricalCrossentropy':
        return SparseCategoricalCrossentropy()
    else:
        raise ValueError('Unknown LOSS_FCNS!', loss)


def define_convnet_new(n_input, n_output, MODEL_PATH='', **hyper_params):
    """Defining/loading a Tensorflow model.

    Args:
        n_input (tuple): shape of the input tensor
        n_output (tuple): shape of the output tensor

    Keyword Args:
        **MODEL_PATH (string): path where Tensorflow models are stored, needs to be provided when loading an existing model
        **CONV_LAYERS (int): number of convolutional layers
        **DENSE_LAYERS (int): number of dense layers
        **DENSE_NODES (list): list containing the number of nodes per dense layer
        **NFILTERS (list): list containing the number of nodes per conv layer
        **KERNEL_SIZE (list): list containing the 2D kernel
        **POOL_SIZE (list): dimensions of the pooling layers
        **ACTIVATIONS (string): name of the activation functions for the convolutional layers
        **ACTIVATION_OL (string): name of the activation functions for the dense output layer
        **BATCH_NORM (boolean): normalize the input layer by adjusting and scaling the activations
        **LOSS_FCNS (string): name of the loss function, default: 'CategoricalCrossentropy'
        **OPTIMIZER (string): name of the optimizer method, default: 'adam'
        **beta_1 (float): additional parameter for nadam optimizer, default: 0.9
        **beta_2 (float): additional parameter for nadam optimizer, default: 0.999
        **LEARNING_RATE (float): controls the speed of the training process, default: 1.e-4
        **DECAY_RATE (float): controls the decay of the learning rate, default: 1.e-3
        **MOMENTUM (float): additional parameter for optimizers, default: 0.9

    Returns:
        model (Tensorflow object): definded/loaded Tensorflow model
    """

    if os.path.exists(MODEL_PATH):
        # load model
        model = keras.models.load_model(MODEL_PATH)
        print(f'Loaded model from disk:\n{MODEL_PATH}')
        model.summary()
        return model

    DENSE_LAYERS = hyper_params['DENSE_LAYERS'] if 'DENSE_LAYERS' in hyper_params else ValueError('DENSE_LAYERS missing!')
    NFILTERS = hyper_params['NFILTERS'] if 'NFILTERS' in hyper_params else ValueError('NFILTERS missing!')
    KERNEL_SIZE = hyper_params['KERNEL_SIZE'] if 'KERNEL_SIZE' in hyper_params else ValueError('KERNEL_SIZE missing!')
    STRIDE_SIZE = hyper_params['STRIDE_SIZE'] if 'STRIDE_SIZE' in hyper_params else ValueError('STRIDE_SIZE missing!')
    POOL_SIZE = hyper_params['POOL_SIZE'] if 'POOL_SIZE' in hyper_params else False
    KERNEL_INITIALIZER = hyper_params['KERNEL_INITIALIZER'] if 'KERNEL_INITIALIZER' in hyper_params else ''
    REGULARIZER = hyper_params['REGULARIZER'] if 'REGULARIZER' in hyper_params else ''
    ACTIVATION = hyper_params['ACTIVATIONS'] if 'ACTIVATIONS' in hyper_params else ValueError('ACTIVATIONS missing!')
    ACTIVATION_OL = hyper_params['ACTIVATION_OL'] if 'ACTIVATION_OL' in hyper_params else 'softmax'
    BATCH_NORM = hyper_params['BATCH_NORM'] if 'BATCH_NORM' in hyper_params else False
    DROPOUT = hyper_params['DROPOUT'] if 'DROPOUT' in hyper_params else -1.0

    # create the model and add the input layer
    regularizers = _get_regularizer(REGULARIZER)
    initializer = _get_initializer(KERNEL_INITIALIZER)

    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=[f'/gpu:{str(util.pick_gpu_lowest_memory())}'],
    )

    with mirrored_strategy.scope():
        inputs = Input(shape=n_input)

        # add convolutional layers
        conv_layer_settings = {
            'activation': ACTIVATION,
            'padding': "same",
            'kernel_initializer': initializer,
            'kernel_regularizer': regularizers,
        }
        for i, (ifilt, ikern, istrd, ipool) in enumerate(zip(NFILTERS, KERNEL_SIZE, STRIDE_SIZE, POOL_SIZE)):
            conv_layer_settings.update({'strides': istrd, 'name': f'Conv2D-layer-{i}'})
            x = inputs if i == 0 else x
            x = Conv2D(ifilt, ikern, **conv_layer_settings)(x)
            if BATCH_NORM: x = BatchNormalization()(x)
            if POOL_SIZE: x = MaxPool2D(pool_size=ipool)(x)

        x = Flatten()(x)
        dense_layer_settings = {
            'activation': ACTIVATION,
            'kernel_initializer': initializer,
            'kernel_regularizer': regularizers,
        }
        for i, idense in enumerate(DENSE_LAYERS):
            dense_layer_settings.update({'name': f'dense-layer-{i}'})

            x = Dense(idense, **dense_layer_settings)(x)
            if BATCH_NORM: x = BatchNormalization()(x)
            if DROPOUT > 0.0: x = Dropout(DROPOUT)(x)

        outputs = Dense(n_output[0], activation=ACTIVATION_OL, name='prediction')(x)
        model = keras.Model(inputs, outputs)
        print(f"Created model {MODEL_PATH}")

        LOSSES = hyper_params['LOSS_FCNS'] if 'LOSS_FCNS' in hyper_params else ValueError('LOSS_FCNS missing!')
        OPTIMIZER = hyper_params['OPTIMIZER'] if 'OPTIMIZER' in hyper_params else ValueError('OPTIMIZER missing!')
        beta_1 = hyper_params['beta_1'] if 'beta_1' in hyper_params else 0.9
        beta_2 = hyper_params['beta_2'] if 'beta_2' in hyper_params else 0.999
        learning_rate = hyper_params['LEARNING_RATE'] if 'LEARNING_RATE' in hyper_params else 1.e-4
        decay_rate = hyper_params['DECAY_RATE'] if 'DECAY_RATE' in hyper_params else learning_rate * 1.e-3
        momentum = hyper_params['MOMENTUM'] if 'MOMENTUM' in hyper_params else 0.9

        model.compile(
            optimizer=_get_optimizer(OPTIMIZER, learning_rate, decay_rate, momentum, beta_1=beta_1, beta_2=beta_2),
            loss=_get_loss_function(LOSSES),
            metrics=['sparse_categorical_accuracy']
        )
    model.summary()

    """
    import sys
    import pyLARDA.helpers as h
    h.change_dir(sys.path[0])
    os.environ["PATH"] += os.pathsep + '/home/sdig/anaconda3/pkgs/graphviz-2.40.1-h21bd128_2/bin/'
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='quicklook.png')
    """

    return model


def define_convnet(n_input, n_output, MODEL_PATH='', **hyper_params):
    """Defining/loading a Tensorflow model.

    Args:
        n_input (tuple): shape of the input tensor
        n_output (tuple): shape of the output tensor

    Keyword Args:
        **MODEL_PATH (string): path where Tensorflow models are stored, needs to be provided when loading an existing model
        **CONV_LAYERS (int): number of convolutional layers
        **DENSE_LAYERS (int): number of dense layers
        **DENSE_NODES (list): list containing the number of nodes per dense layer
        **NFILTERS (list): list containing the number of nodes per conv layer
        **KERNEL_SIZE (list): list containing the 2D kernel
        **POOL_SIZE (list): dimensions of the pooling layers
        **ACTIVATIONS (string): name of the activation functions for the convolutional layers
        **ACTIVATION_OL (string): name of the activation functions for the dense output layer
        **BATCH_NORM (boolean): normalize the input layer by adjusting and scaling the activations
        **LOSS_FCNS (string): name of the loss function, default: 'CategoricalCrossentropy'
        **OPTIMIZER (string): name of the optimizer method, default: 'adam'
        **beta_1 (float): additional parameter for nadam optimizer, default: 0.9
        **beta_2 (float): additional parameter for nadam optimizer, default: 0.999
        **LEARNING_RATE (float): controls the speed of the training process, default: 1.e-4
        **DECAY_RATE (float): controls the decay of the learning rate, default: 1.e-3
        **MOMENTUM (float): additional parameter for optimizers, default: 0.9

    Returns:
        model (Tensorflow object): definded/loaded Tensorflow model
    """

    if os.path.exists(MODEL_PATH):
        # load model
        model = keras.models.load_model(MODEL_PATH)
        print(f'Loaded model from disk:\n{MODEL_PATH}')
    else:
        CONV_DIMENSION = hyper_params['CONV_DIMENSION'] if 'CONV_DIMENSION' in hyper_params else ValueError('CONV_DIMENSION missing!')
        DENSE_LAYERS = hyper_params['DENSE_LAYERS'] if 'DENSE_LAYERS' in hyper_params else ValueError('DENSE_LAYERS missing!')
        NFILTERS = hyper_params['NFILTERS'] if 'NFILTERS' in hyper_params else ValueError('NFILTERS missing!')
        KERNEL_SIZE = hyper_params['KERNEL_SIZE'] if 'KERNEL_SIZE' in hyper_params else ValueError('KERNEL_SIZE missing!')
        STRIDE_SIZE = hyper_params['STRIDE_SIZE'] if 'STRIDE_SIZE' in hyper_params else ValueError('STRIDE_SIZE missing!')
        POOL_SIZE = hyper_params['POOL_SIZE'] if 'POOL_SIZE' in hyper_params else False
        KERNEL_INITIALIZER = hyper_params['KERNEL_INITIALIZER'] if 'KERNEL_INITIALIZER' in hyper_params else ''
        REGULARIZER = hyper_params['REGULARIZER'] if 'REGULARIZER' in hyper_params else ''
        ACTIVATION = hyper_params['ACTIVATIONS'] if 'ACTIVATIONS' in hyper_params else ValueError('ACTIVATIONS missing!')
        ACTIVATION_OL = hyper_params['ACTIVATION_OL'] if 'ACTIVATION_OL' in hyper_params else 'softmax'
        BATCH_NORM = hyper_params['BATCH_NORM'] if 'BATCH_NORM' in hyper_params else False
        DROPOUT = hyper_params['DROPOUT'] if 'DROPOUT' in hyper_params else -1.0

        # create the model and add the input layer

        if KERNEL_INITIALIZER == 'random_normal':
            initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None)
        else:
            initializer = None

        if REGULARIZER == 'l1':
            regularizers = tf.keras.regularizers.l1(l=0.1)
        elif REGULARIZER == 'l2':
            regularizers = tf.keras.regularizers.l2(l=0.01)
        elif REGULARIZER == 'l1_l2':
            regularizers = tf.keras.regularizers.l1_l2(l1=0.1, l2=0.01)
        else:
            regularizers = None

        mirrored_strategy = tf.distribute.MirroredStrategy(
            devices=[f'/gpu:{str(util.pick_gpu_lowest_memory())}'],
        )

        if CONV_DIMENSION == 'conv2d':
            ConvLayer, Pooling = Conv2D, MaxPool2D
        elif CONV_DIMENSION == 'conv1d':
            ConvLayer, Pooling = Conv1D, MaxPool1D
        else:
            raise ValueError(f'Unknown CONV_DIMENSION = {CONV_DIMENSION}')

        with mirrored_strategy.scope():
            model = Sequential()
            model.add(tf.keras.layers.Masking(mask_value=0., input_shape=n_input))

            # add convolutional layers
            for i, (ifilt, ikern, istrd, ipool) in enumerate(zip(NFILTERS, KERNEL_SIZE, STRIDE_SIZE, POOL_SIZE)):
                conv_layer_settings = {
                    'strides': istrd,
                    'activation': ACTIVATION,
                    'padding': "same",
                    'kernel_initializer': initializer,
                    'kernel_regularizer': regularizers,
                    'name': f'{CONV_DIMENSION}-layer-{i}'
                }
                if i == 0:
                    conv_layer_settings['input_shape'] = n_input
                    conv_layer_settings['name'] = f'{CONV_DIMENSION}-input-layer'

                model.add(ConvLayer(ifilt, ikern, **conv_layer_settings))

                if BATCH_NORM: model.add(BatchNormalization())
                if POOL_SIZE: model.add(Pooling(pool_size=ipool))

            model.add(Flatten())

            for i, idense in enumerate(DENSE_LAYERS):
                dense_layer_settings = {
                    'activation': ACTIVATION,
                    'kernel_initializer': initializer,
                    'kernel_regularizer': regularizers,
                    'name': f'dense-layer-{i}'
                }
                model.add(Dense(idense, **dense_layer_settings))
                if BATCH_NORM:    model.add(BatchNormalization())
                if DROPOUT > 0.0: model.add(Dropout(DROPOUT))

            model.add(Dense(n_output[0], activation=ACTIVATION_OL, name='prediction'))
            # model.add(Dense(n_output[0], activation=ACTIVATION_OL, name='prediction'))
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
            elif OPTIMIZER == 'adam':
                opt = Adam(lr=learning_rate, decay=decay_rate)
            else:
                raise ValueError('Unknown OPTIMIZER!', OPTIMIZER)

            if LOSSES == 'BinaryCrossentropy':
                loss = BinaryCrossentropy()
            elif LOSSES == 'CategoricalCrossentropy':
                loss = CategoricalCrossentropy()
            elif LOSSES == 'SparseCategoricalCrossentropy':
                loss = SparseCategoricalCrossentropy()
            else:
                raise ValueError('Unknown LOSS_FCNS!', LOSSES)

            model.compile(optimizer=opt, loss=loss, metrics=['sparse_categorical_accuracy'])
    model.summary()

    """
    import sys
    import pyLARDA.helpers as h
    h.change_dir(sys.path[0])
    os.environ["PATH"] += os.pathsep + '/home/sdig/anaconda3/pkgs/graphviz-2.40.1-h21bd128_2/bin/'
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='quicklook.png')
    """

    return model


def training(model, train_set, train_label, **hyper_params):
    """Training a Tensorflow model.

    Args:
        model (Tensorflow object): loaded Tensorflow model
        train_set (numpy array): the training dataset (num_samples, x_dim, y_dim, 1)
        train_label(numpy array):  the training labels (num_samples, 9)

    Keyword Args: self explanatory
        **BATCH_SIZE:
        **EPOCHS:
        **LOG_PATH: path for keeping the training log files
        **MODEL_PATH: path for keeping the tensorflow optimized weights/biases
        **validation: [features, labels] validation dataset

    Returns:
        model (Tensorflow object): Tensorflow object stored into a file
        history (dict): contains history of trained Tensorflow model

    """

    BATCH_SIZE = hyper_params['BATCH_SIZE'] if 'BATCH_SIZE' in hyper_params else ValueError('BATCH_SIZE missing!')
    EPOCHS = hyper_params['EPOCHS'] if 'EPOCHS' in hyper_params else ValueError('EPOCHS missing!')
    LOG_PATH = hyper_params['LOG_PATH'] if 'LOG_PATH' in hyper_params else ValueError('LOG_PATH missing!')
    MODEL_PATH = hyper_params['MODEL_PATH'] if 'MODEL_PATH' in hyper_params else ValueError('MODEL_PATH missing!')
    VALID_SET = hyper_params['validation'] if 'validation' in hyper_params else ()

    # log model training to tensorboard callback
    tensorboard_callback = TensorBoard(log_dir=LOG_PATH,
                                       histogram_freq=1,
                                       write_graph=True,
                                       write_images=True)

    # initialize tqdm callback with default parameters
    tqdm_callback = tfa.callbacks.TQDMProgressBar()

    training_data = tf.data.Dataset.from_tensor_slices((train_set, train_label)).shuffle(train_set.shape[0], reshuffle_each_iteration=True)
    training_data = training_data.batch(BATCH_SIZE)
    validation_data = tf.data.Dataset.from_tensor_slices(VALID_SET).batch(BATCH_SIZE, drop_remainder=True)

    history = model.fit(
        training_data,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=[
            tensorboard_callback,
            tqdm_callback,
            # lr_callback
        ],
        validation_data=validation_data,
        verbose=0
    )

    # serialize model to HDF5
    model.save(MODEL_PATH)
    print(f"Saved model to disk :: {MODEL_PATH}")

    return history


def predict_classes(model, test_set, batch_size=128):
    """Prediction of classes with a Tensorflow model.

    Args:
        model (Tensorflow object):  loaded Tensorflow model
        test_set (numpy.array): the training dataset (num_samples, x_dim, y_dim, 1)

    Returns:
        predicted_classes (Tensorflow object): trained Tensorflow model

    """
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=[f'/gpu:{str(util.pick_gpu_lowest_memory())}'],
    # cross_device_ops=tf.contrib.distribute.AllReduceCrossDeviceOps(all_reduce_alg="hierarchical_copy")
    )
    with mirrored_strategy.scope():
        #test_set = tf.data.Dataset.from_tensor_slices(test_set).batch(batch_size)
        return model.predict(test_set, verbose=1)


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
