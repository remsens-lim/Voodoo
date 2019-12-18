import numpy as np
import datetime

########################################################################################################################################################
########################################################################################################################################################
#
#   ______         _____  ______  _______              _____  _______  ______ _______ _______      _______  _____  __   _ _______ _____  ______
#  |  ____ |      |     | |_____] |_____| |           |_____] |_____| |_____/ |_____| |  |  |      |       |     | | \  | |______   |   |  ____
#  |_____| |_____ |_____| |_____] |     | |_____      |       |     | |    \_ |     | |  |  |      |_____  |_____| |  \_| |       __|__ |_____|
#
#
########################################################################################################################################################
########################################################################################################################################################

t0_voodoo = datetime.datetime.today()
time_str = f'{t0_voodoo:%Y%m%d-%H%M%S}'

fig_size   = [12, 7]
plot_range = [0, 12000]

plot_training_set           = True
plot_training_set_histogram = False
plot_bsc_dpl_rangespec      = False
plot_spectra_cwt            = True

window_dimension = (3, 3)

TRAIN_SHEET = 'training_cases.toml'
TEST_SHEET  = 'training_cases.toml'

# define ANN model hyperparameter space


predict_model = False

use_mlp_regressen_model  = False
use_cnn_classfier_model  = False
use_cnn_regression_model = True

add_lidar_float          = True if use_cnn_regression_model else False
add_lidar_binary         = False if use_cnn_classfier_model else False

add_moments = False
add_spectra = False
add_cwt     = True


BATCH_SIZE   = 1
EPOCHS       = 100

DENSE_LAYERS = [1]
LAYER_SIZES  = [(32,)]

CONV_LAYERS  = [3]
KERNEL_SIZE  = (6, 6)
POOL_SIZE    = (2, 2)
NFILTERS     = [(32, 64, 128)]

# regression
OPTIMIZERS   = ['sgd']
ACTIVATIONS  = ['leakyrelu']
LOSS_FCNS    = ['mse']

# binary classifier
#OPTIMIZERS   = ['adam']
#ACTIVATIONS  = ['relu']
#LOSS_FCNS    = ['binary_crossentropy']

# define paths
VOODOO_PATH  = '/home/sdig/code/larda3/voodoo/'
BOARD_NAME   = f'voodoo-mlp-training-{time_str}__{BATCH_SIZE}-bachsize-{EPOCHS}-epochs'
LOGS_PATH    = f'{VOODOO_PATH}logs/'
MODELS_PATH  = f'{VOODOO_PATH}models/'
PLOTS_PATH   = f'{VOODOO_PATH}plots/'

#TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-8_8-kernelsize-leakyrelu--20191128-173950.h5'  # even better
#TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-2_4-kernelsize-leakyrelu--20191208-234138.h5' # ok
#TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-8_8-kernelsize-leakyrelu--20191211-232527.h5'
TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-6_6-kernelsize-leakyrelu--20191213-104630.h5'

# define normalization boundaries and conversion for radar (feature) and lidar (label) space
radar_list = []
radar_info = {'spec_lims':      [1.0e-6, 1.0e2],
              'spec_converter': 'lin2z',
              'normalization':  'normalize'
              }

lidar_list = ['attbsc1064', 'depol']
lidar_info = {'attbsc1064_lims': [1.e-7, 1.e-3],
              'voldepol_lims': [1.e-7, 0.3],
              'bsc_converter': 'log',
              'dpl_converter': 'none',
              'normalization': 'none',
              'bsc_shift': 0,
              'dpl_shift': 0
              }

# controls the ccontinuous wavelet transformation
CWT_PARAMS   = {'dim': '2d',
                'scales': np.linspace(2 ** 1.5, 2 ** 4.75, 32),
                'plot_cwt': True,
                'normalization': 'normalize'}