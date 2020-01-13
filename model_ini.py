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

plot_range = [0, 12000]

plot_training_set           = True
plot_training_set_histogram = False
plot_bsc_dpl_rangespec      = False
plot_spectra_cwt            = False

# general plot appearance
plot_settings = {'fig_size': [12, 7], 'range_interval': plot_range, 'rg_converter': True}

window_dimension = (3, 3)

TRAIN_SHEET = 'training_cases.toml'
TEST_SHEET  = 'training_cases.toml'

# define ANN model hyperparameter space

use_mlp_regressen_model  = False
use_cnn_classfier_model  = False
use_cnn_regression_model = True

regression_or_binary = 'regression'

BATCH_SIZE   = 10
EPOCHS       = 150

DENSE_LAYERS = [1]
LAYER_SIZES  = [(64,)]

CONV_LAYERS  = [3]
KERNEL_SIZE  = (8, 8)
POOL_SIZE    = (2, 4)
NFILTERS     = [(32, 64, 86)]

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
BOARD_NAME   = f'voodoo-board-{time_str}_{BATCH_SIZE}-bachsize-{EPOCHS}-epochs'
LOGS_PATH    = f'{VOODOO_PATH}logs/'
MODELS_PATH  = f'{VOODOO_PATH}models/'
PLOTS_PATH   = f'{VOODOO_PATH}plots/'

#TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-8_8-kernelsize-leakyrelu--20191128-173950.h5'  # even better
#TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-2_4-kernelsize-leakyrelu--20191208-234138.h5' # ok
#TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-8_8-kernelsize-leakyrelu--20191211-232527.h5'
#TRAINED_MODEL = '2-conv-(32, 64, 86)-filter-8_8-kernelsize-leakyrelu--20191228-210347.h5'
TRAINED_MODEL = '3-conv-(32, 64, 86)-filter-8_8-kernelsize-leakyrelu--20191230-125951.h5'

# define normalization boundaries and conversion for radar (feature) and lidar (label) space
feature_info = {'VSpec':    {'var_lims': [1.0e-6, 1.0e2], 'var_converter': 'lin2z',            'scaling': 'normalize', 'used': False},
                'Ze':       {'var_lims': [1.0e-6, 1.0e2], 'var_converter': '-',                'scaling': 'normalize', 'used': False},
                'VEL':      {'var_lims': [-6.0, 4.0],     'var_converter': '-',                'scaling': 'normalize', 'used': False},
                'sw':       {'var_lims': [0.0, 3.0],      'var_converter': '-',                'scaling': 'normalize', 'used': False},
                'skew':     {'var_lims': [-3.0, 3.0],     'var_converter': '-',                'scaling': 'normalize', 'used': False},
                'kurt':     {'var_lims': [0.0, 3.0],      'var_converter': '-',                'scaling': 'normalize', 'used': False},
                'cwt':      {'var_lims': [0.0, 30.0],     'var_converter': ['lin2z', 'chsgn'], 'scaling': 'normalize', 'used': True,
                             'plot_cwt': False,           'scales': np.linspace(2 ** 1.25, 2 ** 4.75, 32)}
                }

# define normalization boundaries and conversion for radar (feature) and lidar (label) space
target_info = {'attbsc1064': {'var_lims': [1.e-7, 1.e-3], 'var_converter': 'log',       'scaling': '-', 'used': True},
               'depol':      {'var_lims': [1.e-7, 0.3],   'var_converter': 'ldr2cdr',   'scaling': '-', 'used': True},
               }
