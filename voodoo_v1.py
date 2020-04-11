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

import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model

import generate_trainingset as Loader

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

CASE_LIST = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'
VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
SYSTEM = 'limrad94'

DATA_PATH = f'{VOODOO_PATH}/data/'
LOGS_PATH = f'{VOODOO_PATH}/logs/'
MODELS_PATH = f'{VOODOO_PATH}/models/'
PLOTS_PATH = f'{VOODOO_PATH}/plots/'



def check_mat_file_availability(data_path, kind, dt_str, system):
    if not os.path.isfile(f'{data_path}/features/{kind}/{dt_str}_{system}_features_{kind}.mat'):
        print(f"{data_path}/features/{kind}/{dt_str}_{system}_features_{kind}.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/labels/{dt_str}_{SYSTEM}_labels.mat'):
        print(f"'{data_path}/labels/{dt_str}_{SYSTEM}_labels.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/labels/{dt_str}_{SYSTEM}_masked.mat'):
        print(f"'{data_path}/labels/{dt_str}_{SYSTEM}_masked.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/cloudnet/{dt_str}_cloudnetpy94_class.mat'):
        print(f"'{data_path}/cloudnet/{dt_str}_cloudnetpy94_class.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/cloudnet/{dt_str}_cloudnetpy94_status.mat'):
        print(f"'{data_path}/cloudnet/{dt_str}_cloudnetpy94_status.mat'  not found!")
        return False
    return True

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
            f'{kwargs["LEARNING_RATE"]:.1e}-lr--' \
            f'{kwargs["DECAY_RATE"]:.1e}-dr--' \
            f'{kwargs["DENSE_LAYERS"]}-dl--' \
            f'{str(kwargs["DENSE_NODES"])[1:-1].replace(", ", "-")}-dn--' \
            f'{kwargs["KIND"]}--' \
            f'{str(kwargs["INPUT_DIMENSION"])[1:-1].replace(", ", "-")}-dIN--' \
            f'{str(kwargs["OUTPUT_DIMENSION"])[1:-1].replace(", ", "-")}-dOUT--' \
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
        print(f'FileNotFound Error: Check ~/{file}')
        return [], True

    return data, False


def load_mat_files_list(case_string_list, kind):
    mat_files_list = []
    for icase, case_str in enumerate(case_string_list):

        # gather time interval, etc.
        case = Loader.load_case_list(CASE_LIST, case_str)
        time_span = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
        dt_str = f'{time_span[0]:%Y%m%d_%H%M}-{time_span[1]:%H%M}'

        # check if a mat files is available
        if check_mat_file_availability(DATA_PATH, kind, dt_str, SYSTEM):
            mat_files_list.append([
                [f'{DATA_PATH}/cloudnet/', f'{dt_str}_cloudnetpy94_class.mat'],
                [f'{DATA_PATH}/cloudnet/', f'{dt_str}_cloudnetpy94_status.mat'],
                [f'{DATA_PATH}/features/{KIND}', f'{dt_str}_{SYSTEM}_features_{KIND}.mat'],
                [f'{DATA_PATH}/labels/', f'{dt_str}_{SYSTEM}_labels.mat'],
                [f'{DATA_PATH}/labels/', f'{dt_str}_{SYSTEM}_masked.mat'],
            ])

    if not mat_files_list: raise ValueError('Empty mat file list!')
    return mat_files_list


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

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    TRAINED_MODEL = kwargs['model'] + ' ' + args[0][:] if len(args) > 0 else kwargs['model'] if 'model' in kwargs else ''
    TASK = kwargs['task'] if 'task' in kwargs else 'train'
    KIND = kwargs['kind'] if 'kind' in kwargs else 'HSI'

    use_only_given = True if TASK == 'train' else False

    n_channels_ = 6 if 'HSI' in KIND else 1

    if TASK == 'predict' and not os.path.isfile(f'{MODELS_PATH}/{TRAINED_MODEL}'):
        raise FileNotFoundError(f'Trained model not found! {TRAINED_MODEL}')

    if 'case' in kwargs:
        if len(kwargs['case']) == 17:
            use_only_given = True
            CASE_LIST = VOODOO_PATH + f'/tomls/auto-trainingset-{kwargs["case"]}.toml'
            case_string_list = [case for case in Loader.load_case_file(CASE_LIST).keys()]
        else:
            case_string_list = [kwargs['case']]
    else:
        # multiple cases
        case_string_list = [
            # '20190318-02',
            # '20190102-02',
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
    tf_settings = config_global_model['tensorflow']

    feature_set, target_labels = [], []

    for icase, case_str in enumerate(case_string_list):

        # gather time interval, etc.
        case = Loader.load_case_list(CASE_LIST, case_str)
        TIME_SPAN = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
        dt_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'

        # check if a mat files is available
        mat_file_avlb = check_mat_file_availability(DATA_PATH, KIND, dt_str, SYSTEM)
        #mat_file_avlb = False

        if mat_file_avlb:# and use_only_given:

            _class, flag = load_matv5(f'{DATA_PATH}/cloudnet/', f'{dt_str}_cloudnetpy94_class.mat')
            if flag: continue
            _status, flag = load_matv5(f'{DATA_PATH}/cloudnet/', f'{dt_str}_cloudnetpy94_status.mat')
            if flag: continue
            feature, flag = load_matv5(f'{DATA_PATH}/features/{KIND}', f'{dt_str}_{SYSTEM}_features_{KIND}.mat')
            if flag: continue
            target, flag = load_matv5(f'{DATA_PATH}/labels/', f'{dt_str}_{SYSTEM}_labels.mat')
            if flag: continue
            masked, flag = load_matv5(f'{DATA_PATH}/labels/', f'{dt_str}_{SYSTEM}_masked.mat')
            if flag: continue

            feature = feature['features']
            target = target['labels']
            masked = masked['masked']

            print(f'\nloaded :: {TIME_SPAN[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN[1]:%H:%M:%S} mat files\n')

        else:

            if use_only_given: continue

            feature, target, masked, _class, _status = Loader.load_features_from_nc(
                time_span=TIME_SPAN,
                voodoo_path=VOODOO_PATH,  # NONSENSE PATH
                data_path=DATA_PATH,
                kind=KIND,
                system=SYSTEM,
                save=False,
                n_channels=n_channels_
            )

            if masked.all(): continue  # if there are no datapoints

            loggers[0].info(f'min/max value in features = {np.min(feature)},  maximum = {np.max(feature)}')
            loggers[0].info(f'min/max value in targets  = {np.min(target)},  maximum = {np.max(target)}')

        if TASK == 'train':

            mask1 = Loader.load_training_mask(_class, _status)
            valid_samples, masked = set_intersection(masked, mask1)

            if len(valid_samples) < 1: continue

            if len(feature.shape) == 3: feature = feature[:, :, :, np.newaxis]
            feature_set.append(feature[valid_samples, :, :, :])
            target_labels.append(target[valid_samples, :])

        elif TASK == 'predict':
            if len(feature.shape) == 3: feature = feature[:, :, :, np.newaxis]
            feature_set.append(feature)
            target_labels.append(target)

        else:
            raise ValueError(f'Unknown TASK: {TASK}.')

    feature_set = np.concatenate(feature_set, axis=0)
    target_labels = np.concatenate(target_labels, axis=0)

    if TASK == 'train':
        # take every nth element from the trainingset for validation
        n = 10
        validation_set = (feature_set[::n], target_labels[::n])
        feature_set = np.array([item for index, item in enumerate(feature_set) if (index + 1) % n != 0])
        target_labels = np.array([item for index, item in enumerate(target_labels) if (index + 1) % n != 0])
    else:
        validation_set = ()

    names = [
        'Cloud liquid droplets only',
        'Drizzle or rain.',
        "Drizzle/rain & cloud droplets",
        'Ice particles.',
        'Ice coexisting with supercooled liquid droplets.',
        'Melting ice particles',
        "Melting ice & cloud droplets",
        'Insects or ground clutter.',
    ]
    n_samples_totall = np.count_nonzero(target_labels, axis=0)

    print('samples per class')
    print(f'{feature_set.shape[0]:12d}   total')
    for name, n_smp in zip(names, n_samples_totall):
        print(f'{n_smp:12d}   {name}')

    print(f'min max val features = {feature_set.min():.2f},  {feature_set.max():.2f}')
    print(f'min max val targets  = {target_labels.min():.2f},  {target_labels.max():.2f}')


    ########################################################################################################################################################
    #   ___ ____ ____ _ _  _ _ _  _ ____
    #    |  |__/ |__| | |\ | | |\ | | __
    #    |  |  \ |  | | | \| | | \| |__]
    #
    #
    if TASK == 'train':
        model_list = []

        # loop through hyperparameter space
        for cl, dl, af, il, op, lr in product(tf_settings['CONV_LAYERS'],
                                              tf_settings['DENSE_LAYERS'],
                                              tf_settings['ACTIVATIONS'],
                                              tf_settings['LOSS_FCNS'],
                                              tf_settings['OPTIMIZERS'],
                                              tf_settings['LEARNING_RATE']):
            hyper_params = {

                # Convolutional part of the model
                'KIND': KIND,
                'INPUT_DIMENSION': feature_set.shape,
                'OUTPUT_DIMENSION': target_labels.shape,
                'CONV_LAYERS': cl,
                'NFILTERS': tf_settings['NFILTERS'],
                'KERNEL_SIZE': tf_settings['KERNEL_SIZE'],
                'POOL_SIZE': tf_settings['POOL_SIZE'],
                'ACTIVATIONS': af,

                # fully connected layers
                'DENSE_LAYERS': dl,
                'DENSE_NODES': tf_settings['DENSE_NODES'],
                'ACTIVATIONS_OL': tf_settings['ACTIVATIONS_OL'],
                'LOSS_FCNS': il,
                'OPTIMIZER': op,

                # training settings
                'BATCH_SIZE': tf_settings['BATCH_SIZE'],
                'EPOCHS': tf_settings['EPOCHS'],
                'LEARNING_RATE': lr,
                'DECAY_RATE': tf_settings['DECAY_RATE'],
                'MOMENTUM': tf_settings['MOMENTUM'],

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
            prediction_container['colormap'] = 'ann_target_7class'
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
