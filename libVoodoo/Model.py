"""
This module contains functions for generating deep learning models with Tensorflow and Keras.

"""

import libVoodoo.Utils as util
import numpy as np
import os
# neural network imports
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, AvgPool2D, Flatten, Input, Activation
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.models import Model as kmodel
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Nadam

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

#tf.compat.v1.keras.backend.set_session(
#    tf_debug.TensorBoardDebugWrapperSession(tf.compat.v1.Session(), "sdig-workstation:6006"))


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
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

#config = tf.compat.v1.ConfigProto(
#        device_count = {'GPU': 0},
#        intra_op_parallelism_threads=50
#    )
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    elif krnl_string == 'random_uniform':
        return tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
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
    elif loss == 'WeightedBinaryCrossentropy':
        return WeightedBinaryCrossentropy
    else:
        raise ValueError('Unknown LOSS_FCNS!', loss)


import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

def WeightedBinaryCrossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1  # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost)  # average on all labels
    return macro_cost

@tf.function
def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1)
    return macro_f1

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
    METRICS = hyper_params['METRICS'] if 'METRICS' in hyper_params else ['sparse_categorical_accuracy']

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
            'padding': "same",
            'kernel_initializer': initializer,
            'kernel_regularizer': regularizers,
        }
        for i, (ifilt, ikern, istrd, ipool) in enumerate(zip(NFILTERS, KERNEL_SIZE, STRIDE_SIZE, POOL_SIZE)):
            conv_layer_settings.update({'strides': istrd, 'name': f'Conv2D-layer-{i}'})
            x = inputs if i == 0 else x
            x = Conv2D(ifilt, ikern, **conv_layer_settings)(x)
            if BATCH_NORM: x = BatchNormalization()(x)
            x = Activation(ACTIVATION)(x)
            if DROPOUT > 0.0: x = Dropout(DROPOUT)(x)
            if POOL_SIZE: x = AvgPool2D(pool_size=ipool)(x)

        x = Flatten()(x)
        dense_layer_settings = {
            'kernel_initializer': initializer,
            'kernel_regularizer': regularizers,
        }
        for i, idense in enumerate(DENSE_LAYERS):
            dense_layer_settings.update({'name': f'dense-layer-{i}'})

            x = Dense(idense, **dense_layer_settings)(x)
            if BATCH_NORM: x = BatchNormalization()(x)
            x = Activation(ACTIVATION)(x)
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
            metrics=METRICS
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

    train_set = tf.data.Dataset.from_tensor_slices((train_set, train_label)).shuffle(train_set.shape[0], reshuffle_each_iteration=True)
    train_set = train_set.batch(BATCH_SIZE)
    VALID_SET = tf.data.Dataset.from_tensor_slices(VALID_SET).batch(BATCH_SIZE, drop_remainder=True)

    history = model.fit(
        train_set,
        epochs=EPOCHS,
        shuffle=True,
        callbacks=[
            tensorboard_callback,
            tqdm_callback,
            # lr_callback
        ],
        validation_data=VALID_SET,
        verbose=0
    )

    # serialize model to HDF5
    model.save(MODEL_PATH)
    print(f"Saved model to disk :: {MODEL_PATH}")

    return history


def predict_classes(model, test_set):
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
