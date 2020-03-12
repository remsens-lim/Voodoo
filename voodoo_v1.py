#!/home/sdig/anaconda3/bin/python
"""
Short description:
    Cloud Radar Doppler Spectra --> Lidar Backscatter and Depolarization

    This is the main program of the voodoo Project. The workflow is as follows:
        1.  reading cloud radar and polarization lidar data and applies pre-processing routines to clean the data.
        2.  plotting the training set and/or the target set, additional plots are provided as well (scatter, history, histograms, ...)
        3.  create an ANN model using the Tensorflow/Keras library or load an existing ANN model.
        4.  training and/or prediction of lidar backscatter and/or lidar depolarization using the ANN model.
        5.  plot predicted target variables


Long description:
    This project aims to provides lidar predictions for variables, such as backscatter and depolarization, beyond regions where the lidar signal was
    completely attenuated by exploiting cloud radar Doppler spectra morphologies.
    An earlier approach was developed by Ed Luke et al. 2010
"""

import datetime
import logging
import os
import sys
import time
import traceback

import numpy as np
import toml
from scipy.io import loadmat
from itertools import product

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import pyLARDA.SpectraProcessing as sp

import voodoo.libVoodoo.Loader_v2 as Loader
import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model

from generate_trainingset import load_features_from_nc

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def get_logger(logger_list, status='info'):
    log = []
    for ilog in logger_list:
        log_ = logging.getLogger(ilog)
        if status == 'info':
            log_.setLevel(logging.INFO)
        elif status == 'debug':
            log_.setLevel(logging.DEBUG)
        elif status == 'critical':
            log_.setLevel(logging.CRITICAL)
        log_.addHandler(logging.StreamHandler())
        log.append(log_)

    return log


def log_dimensions(spec_dim, cn_dim, *args):
    loggers[0].info(f'Radar-Input  :: LIMRAD94\n'
                    f'      (n_ts, n_rg, n_vel) = ({spec_dim["n_ts"]}, {spec_dim["n_rg"]}, {spec_dim["n_vel"]})')
    loggers[0].info(f'Target-Input :: Cloudnetpy94\n'
                    f'      (n_ts, n_rg)        = ({cn_dim["n_ts"]}, {cn_dim["n_rg"]})')
    if len(args) > 0:
        loggers[0].info(f'Target-Input :: PollyNET\n'
                        f'      (n_ts, n_rg)        = ({args[0]["n_ts"]}, {args[0]["n_rg"]})')


def create_filename(modelname, **kwargs):
    if not TRAINED_MODEL:
        name = \
               f'{kwargs["CONV_LAYERS"]}-cl--' \
               f'{kwargs["KERNEL_SIZE"][0]}_{kwargs["KERNEL_SIZE"][1]}-ks--' \
               f'{kwargs["ACTIVATIONS"]}-af--' \
               f'{kwargs["OPTIMIZER"]}-opt--' \
               f'{kwargs["LOSS_FCNS"]}-loss--' \
               f'{kwargs["BATCH_SIZE"]}-bs--' \
               f'{kwargs["EPOCHS"]}-ep--' \
               f'{kwargs["LEARNING_RATE"]}-lr--' \
               f'{kwargs["DECAY_RATE"]}-dr--' \
               f'{kwargs["DENSE_LAYERS"]}-dl--' \
               f'{kwargs["DENSE_NODES"]}-dn--' \
               f'{kwargs["KIND"]}--' \
               f'{kwargs["INPUT_DIMENSION"]}--Fdim' \
               f'{kwargs["OUTPUT_DIMENSION"]}--Tdim' \
               f'{time_str}.h5'
        return {'MODEL_PATH': MODELS_PATH + name, 'LOG_PATH': LOGS_PATH + name}
    else:
        return {'MODEL_PATH': MODELS_PATH + modelname, 'LOG_PATH': LOGS_PATH + modelname}

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)


def set_intersection(mask0, mask1):
    mask_flt = np.where(~mask0.astype(np.bool).flatten())
    mask1_flt = np.where(~mask1.flatten())
    maskX_flt = intersection(mask_flt[0], mask1_flt[0])
    len_flt = len(maskX_flt)
    idx_list = []
    cnt = 0
    for iter, idx in enumerate(mask_flt[0]):
        if cnt >= len_flt: break
        if idx == maskX_flt[cnt]:
            idx_list.append(iter)
            cnt += 1

    return idx_list, masked

def load_matv5(path, file):
    h.change_dir(path)
    try:
        data = loadmat(file)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        raise ValueError(f'Check ~/{file}')

    return data

########################################################################################################################################################
########################################################################################################################################################
#
#
#               _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#               |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#               |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
########################################################################################################################################################
########################################################################################################################################################
if __name__ == '__main__':

    start_time = time.time()
    load_from_nc = True

    # trainingset_case = 'Multiday-01-training'
    testingset_case = '20190801-01'
    trainingset_case = '20190904-03'
    validtest_case =  '' # '20190801-03'

    CASE_LIST = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'
    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'

    DATA_PATH = f'{VOODOO_PATH}/data/'
    LOGS_PATH = f'{VOODOO_PATH}/logs/'
    MODELS_PATH = f'{VOODOO_PATH}/models/'
    PLOTS_PATH = f'{VOODOO_PATH}/plots/'

    SYSTEM = 'limrad94'

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)
    TRAINED_MODEL = kwargs['model'] + ' ' + args[0][:] if len(args) > 0 else kwargs['model'] if 'model' in kwargs else ''
    TASK = kwargs['task'] if 'task' in kwargs else 'train'
    KIND = kwargs['kind'] if 'kind' in kwargs else 'multispectra'

    n_channels_ = 4 if KIND == 'multispectra' else 1

    if 'case' in kwargs:
        case_string_list = [kwargs['case']]
    else:
        # multiple cases
        case_string_list = [
            #'20190318-02',
            #'20190102-02',
            '20190318-99',
        ]

    # gather todays date
    t0_voodoo = datetime.datetime.today()
    time_str = f'{t0_voodoo:%Y%m%d-%H%M%S}'

    # get all loggers
    loggers = get_logger(['libVoodoo'])

    # load ann model parameter and other global values
    config_global_model = toml.load(VOODOO_PATH + 'ann_model_setting.toml')
    radar_input_setting = config_global_model['feature']['info']['VSpec']
    tensorflow_settings = config_global_model['tensorflow']

    feature_set = np.empty((0, 256, 32, n_channels_), dtype=np.float32)
    target_labels = np.empty((0, 9), dtype=np.float32)

    for icase, case_str in enumerate(case_string_list):

        # gather time interval, etc.
        case = Loader.load_case_list(CASE_LIST, case_str)
        TIME_SPAN = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
        dt_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'

        # check if a mat files is available
        #mat_file_avlb = os.path.isfile(f'{DATA_PATH}/features/{KIND}/{dt_str}_{SYSTEM}_features_{KIND}.mat')

        if load_from_nc: # and not mat_file_avlb:

            feature, target, masked, _class, _status = load_features_from_nc(
                case_str,
                voodoo_path=VOODOO_PATH,    # NONSENSE PATH
                data_path=DATA_PATH,
                case_list_path=CASE_LIST,
                kind=KIND,
                system=SYSTEM,
                save=True,
                n_channels=n_channels_
            )

        else:

            _class  = load_matv5(f'{DATA_PATH}/cloudnet/', f'{dt_str}_cloudnetpy94_class.mat')
            _status = load_matv5(f'{DATA_PATH}/cloudnet/', f'{dt_str}_cloudnetpy94_status.mat')
            feature = load_matv5(f'{DATA_PATH}/features/{KIND}', f'{dt_str}_{SYSTEM}_features_{KIND}.mat')['features']
            target = load_matv5(f'{DATA_PATH}/labels/', f'{dt_str}_{SYSTEM}_labels.mat')['labels']
            masked = load_matv5(f'{DATA_PATH}/labels/', f'{dt_str}_{SYSTEM}_masked.mat')['masked']

        if icase == 0:
            masked_set = np.empty(((0,) + masked.shape[1:]), dtype=np.bool)
            masked_flt = np.empty((0,), dtype=np.bool)

        if TASK == 'train':

            mask1 = Loader.load_training_mask(_class, _status)
            valid_samples, masked = set_intersection(masked, mask1)

            feature_set = np.append(feature_set, feature[valid_samples, :, :, :], axis=0)
            target_labels = np.append(target_labels, target[valid_samples, :], axis=0)

        elif TASK == 'predict':

            feature_set = np.append(feature_set, feature, axis=0)
            target_labels = np.append(target_labels, target, axis=0)

        else:
            raise ValueError(f'Unknown TASK: {TASK}.')

        masked_set = np.append(masked_set, masked, axis=0)
        masked_flt = np.append(masked_flt, masked)

    # load validation and/or testing data
    if len(validtest_case) > 0:

        case_valid = Loader.load_case_list(CASE_LIST, validtest_case)
        TIME_SPAN = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case_valid['time_interval']]
        case_valid_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'
        loggers[0].info(f'\nloading... {TIME_SPAN[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN[1]:%H:%M:%S} of {SYSTEM} validation data')

        # load validation feature set
        h.change_dir(f'{DATA_PATH}/features/{KIND}/')
        valid_f = loadmat(f'{case_valid_str}_{SYSTEM}_features_{KIND}.mat')['features']

        # load validation labels
        h.change_dir(f'{DATA_PATH}/labels/')
        valid_l = loadmat(f'{case_valid_str}_{SYSTEM}_labels.mat')['labels']

        validation_masked = loadmat(f'{case_valid_str}_{SYSTEM}_masked.mat')['masked']
        validation_set = (valid_f, valid_l)
    else:
        validation_set = ()

    ########################################################################################################################################################
    #   ___ ____ ____ _ _  _ _ _  _ ____
    #    |  |__/ |__| | |\ | | |\ | | __
    #    |  |  \ |  | | | \| | | \| |__]
    #
    #
    if TASK == 'train':
        model_list = []

        # loop through hyperparameter space
        for cl, dl, af, il, op, lr in product(tensorflow_settings['CONV_LAYERS'],
                                              tensorflow_settings['DENSE_LAYERS'],
                                              tensorflow_settings['ACTIVATIONS'],
                                              tensorflow_settings['LOSS_FCNS'],
                                              tensorflow_settings['OPTIMIZERS'],
                                              tensorflow_settings['LEARNING_RATE']):
            hyper_params = {

                # Convolutional part of the model
                'KIND': KIND,
                'INPUT_DIMENSION': str(feature_set.shape).replace(', ', '-'),
                'OUTPUT_DIMENSION': str(target_labels.shape).replace(', ', '-'),
                'CONV_LAYERS': cl,
                'NFILTERS': tensorflow_settings['NFILTERS'],
                'KERNEL_SIZE': tensorflow_settings['KERNEL_SIZE'],
                'POOL_SIZE': tensorflow_settings['POOL_SIZE'],
                'ACTIVATIONS': af,

                # fully connected layers
                'DENSE_LAYERS': dl,
                'DENSE_NODES': tensorflow_settings['DENSE_NODES'],
                'ACTIVATIONS_OL': tensorflow_settings['ACTIVATIONS_OL'],
                'LOSS_FCNS': il,
                'OPTIMIZER': op,

                # training settings
                'BATCH_SIZE': tensorflow_settings['BATCH_SIZE'],
                'EPOCHS': tensorflow_settings['EPOCHS'],
                'LEARNING_RATE': lr,
                'DECAY_RATE': tensorflow_settings['DECAY_RATE'],
                'MOMENTUM': tensorflow_settings['MOMENTUM'],

                # validation data
                'validation': validation_set,

                # GPU
                'DEVICE': 0
            }

            # create file name and add MODEL_PATH and LOGS_PATH to hyper_parameter dict
            hyper_params.update(create_filename(TRAINED_MODEL, **hyper_params))

            # define a new model or load an existing one
            cnn_model = Model.define_cnn(feature_set.shape[1:], target_labels.shape[1:], **hyper_params)

            # parse the training set to the optimizer
            history = Model.training(cnn_model, feature_set, target_labels, **hyper_params)

            # create directory for plots
            fig, _ = Plot.History(history)
            idx = hyper_params["MODEL_PATH"].rfind('/')
            Plot.save_figure(fig,
                             path=f'{PLOTS_PATH}/training/',
                             name=f'histo_loss-acc_{dt_str}__{hyper_params["MODEL_PATH"][idx + 1:-3]}.png',
                             dpi=300
                             )

            model_list.append(hyper_params["MODEL_PATH"][idx + 1:-3])

        for model in model_list:
            loggers[0].info(f"'{model}.h5',")

    ############################################################################################################################################################
    #   ___  ____ ____ ___  _ ____ ___ _ ____ _  _
    #   |__] |__/ |___ |  \ | |     |  | |  | |\ |
    #   |    |  \ |___ |__/ | |___  |  | |__| | \|
    #
    #
    if TASK == 'predict':

        if TRAINED_MODEL:
            model_list = [TRAINED_MODEL]
        else:
            model_list = [
                '3-cl--3_3-ks--tanh-af--adam-opt--CategoricalCrossentropy-loss--4-bs--50-ep--0.0001-lr--1e-05-dr--1-dl--[64, 32]-dn--20200227-203634.h5',
            ]

        hyper_params = {
            'DEVICE': 0
        }

        for model_name in model_list:
            # define a new model or load an existing one
            cnn_model = Model.define_cnn(
                feature_set.shape[1:], target_labels.shape[1:],
                MODEL_PATH=MODELS_PATH + model_name,
                **hyper_params
            )

            # make predictions
            cnn_pred = Model.predict_classes(cnn_model, feature_set)
            prediction2D_classes = Model.one_hot_to_classes(cnn_pred, masked)

            # dspkl_mask = sp.despeckle2D(prediction2D_classes)
            # prediction2D_classes[dspkl_mask] = 0.0

            prediction_container = {}
            prediction_container['dimlabel'] = ['time', 'range']
            prediction_container['name'] = 'CLASS'
            prediction_container['joints'] = ''
            prediction_container['rg_unit'] = 'm'
            prediction_container['colormap'] = 'ann_target'
            prediction_container['system'] = 'Voodoo'
            prediction_container['ts'] = np.squeeze(_class['ts'])
            prediction_container['rg'] = np.squeeze(_class['rg'])
            prediction_container['var_lims'] = [0, 8]
            prediction_container['var_unit'] = ''
            prediction_container['mask'] = masked
            prediction_container['var'] = prediction2D_classes

            # create directory for plots
            fig, _ = tr.plot_timeheight(prediction_container, title=f'preliminary results (ANN prediction) {dt_str}',
                                        range_interval=case['range_interval'])  # , **plot_settings)

            Plot.save_figure(fig,
                             path=f'{PLOTS_PATH}/training/{dt_str}/',
                             name=f'prediction_{dt_str}__{model_name}.png',
                             dpi=200
                             )

    ####################################################################################################################################
    Plot.print_elapsed_time(start_time, '\nDone, elapsed time = ')
