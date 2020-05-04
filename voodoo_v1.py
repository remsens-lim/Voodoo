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
from itertools import product

import numpy as np
import toml
from scipy.io import loadmat
from tqdm.auto import tqdm

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

from numba import jit
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model
import voodoo.libVoodoo.Utils  as Utils

import generate_trainingset as Loader

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.1.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

CASE_LIST_PATH = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'
VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
SYSTEM = 'limrad94'
DPI_ = 200
FIG_SIZE_ = [12, 7]
PLOT_RANGE_ = [0, 12000]

CLOUDNETs = ['CLOUDNETpy94', 'CLOUDNET_LIMRAD']

ANN_MODEL_TOML = 'ann_model_setting3.toml'
DATA_PATH = f'{VOODOO_PATH}/data/'
LOGS_PATH = f'{VOODOO_PATH}/logs/'
MODELS_PATH = f'{VOODOO_PATH}/models/'
PLOTS_PATH = f'{VOODOO_PATH}/plots/'

CLOUDNET_LABELS_ = [
    'Clear sky',
    'Cloud liquid droplets only',
    'Drizzle or rain.',
    'Drizzle/rain & cloud droplet',
    'Ice particles.',
    'Ice coexisting with supercooled liquid droplets.',
    'Melting ice particles',
    'Melting ice & cloud droplets',
    'Insects or ground clutter.',
]


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


# get all loggers
loggers = get_logger(['libVoodoo'], status='info')


def check_mat_file_availability(data_path, kind, dt_str, system=SYSTEM, cloudnet='unkown', conv_dim='unknown'):
    if not os.path.isfile(f'{data_path}/features/{kind}/{conv_dim}/{dt_str}_{system}_features_{kind}.mat'):
        loggers[0].info(f"{data_path}/features/{kind}/{conv_dim}/{dt_str}_{system}_features_{kind}.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/labels/{dt_str}_{system}_labels.mat'):
        loggers[0].info(f"'{data_path}/labels/{dt_str}_{system}_labels.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/labels/{dt_str}_{system}_masked.mat'):
        loggers[0].info(f"'{data_path}/labels/{dt_str}_{system}_masked.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/cloudnet/{dt_str}_{cloudnet}_class.mat'):
        loggers[0].info(f"'{data_path}/cloudnet/{dt_str}_{cloudnet}_class.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/cloudnet/{dt_str}_{cloudnet}_status.mat'):
        loggers[0].info(f"'{data_path}/cloudnet/{dt_str}_{cloudnet}_status.mat'  not found!")
        return False
    if not os.path.isfile(f'{data_path}/cloudnet/{dt_str}_{cloudnet}_model_T.mat'):
        loggers[0].info(f"'{data_path}/cloudnet/{dt_str}_{cloudnet}_model_T.mat'  not found!")
        return False
    return True


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
        name = f"{kwargs['time_str']}_ann-model-weights_{kwargs['CONV_DIMENSION']}.h5"
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

    return idx_list


def load_matv5(path, file):
    h.change_dir(path)
    try:
        data = loadmat(file)
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        loggers[0].info(f'FileNotFound Error: Check ~/{file}')
        return [], True

    if not data:
        return [], True

    return data, False


def container_from_prediction(ts, rg, var, mask, **kwargs):
    prediction_container = {}
    prediction_container['dimlabel'] = ['time', 'range']
    prediction_container['name'] = kwargs['name'] if 'name' in kwargs else 'CLASS'
    prediction_container['joints'] = ''
    prediction_container['rg_unit'] = 'm'
    prediction_container['colormap'] = kwargs['colormap'] if 'colormap' in kwargs else 'ann_target_7class'
    prediction_container['system'] = 'Voodoo'
    prediction_container['ts'] = ts
    prediction_container['rg'] = rg
    prediction_container['var_lims'] = kwargs['var_lims'] if 'var_lims' in kwargs else [0, 8]
    prediction_container['var_unit'] = '1'
    prediction_container['mask'] = mask
    prediction_container['var'] = var
    return prediction_container


def get_isotherms(temperature_list, **kwargs):
    def toC(datalist):
        return datalist[0]['var'] - 273.15, datalist[0]['mask']

    T = {}
    T['dimlabel'] = ['time', 'range']
    T['name'] = 'Temperature'
    T['joints'] = ''
    T['paraminfo'] = ''
    T['filename'] = 'ann_input_files'
    T['rg_unit'] = 'm'
    T['colormap'] = 'cloudnet_jet'
    T['system'] = kwargs['CLOUDNET'] if 'CLOUDNET' in kwargs else 'unknown'
    T['ts'] = np.concatenate([np.squeeze(itemp['ts']) for itemp in temperature_list], axis=0)
    T['rg'] = np.squeeze(temperature_list[0]['rg'])
    T['var_lims'] = [240.0, 320.0]
    T['var_unit'] = 'K'
    T['mask'] = np.concatenate([np.squeeze(itemp['mask']) for itemp in temperature_list], axis=0)
    T['var'] = np.concatenate([np.squeeze(itemp['var']) for itemp in temperature_list], axis=0)

    return {'data': tr.combine(toC, [T], {'var_unit': "C"}), 'levels': np.arange(-40, 16, 5)}


def variable_to_container(list, **kwargs):
    container = {}
    container['dimlabel'] = ['time', 'range']
    container['name'] = kwargs['name']
    container['joints'] = ''
    container['paraminfo'] = ''
    container['filename'] = 'ann_input_files'
    container['rg_unit'] = 'm'
    container['colormap'] = 'cloudnet_jet'
    container['system'] = kwargs['CLOUDNET'] if 'CLOUDNET' in kwargs else 'unknown'
    container['ts'] = np.concatenate([np.squeeze(itemp['ts']) for itemp in list], axis=0)
    container['rg'] = np.squeeze(list[0]['rg'])
    container['var_lims'] = [240.0, 320.0]
    container['var_unit'] = 'K'
    container['mask'] = np.concatenate([np.squeeze(itemp['mask']) for itemp in list], axis=0)
    container['var'] = np.concatenate([np.squeeze(itemp['var']) for itemp in list], axis=0)
    return container


def post_processor_temperature(data, contour):
    import copy
    container = copy.deepcopy(data)

    container['var'][(contour['data']['var'] > 0.0) * (container['var'] == 4)] = 2
    container['var'][(contour['data']['var'] > 0.0) * (container['var'] == 5)] = 3
    container['var'][(contour['data']['var'] < -40.0) * ((container['var'] == 1) + (container['var'] == 5))] = 4
    container['var'][(contour['data']['var'] < 0.0) * (container['var'] == 2)] = 4

    loggers[0].info('Postprocessing temperature info done.')

    return container


def get_good_radar_and_lidar_index(version):
    if version in ['CLOUDNETpy94', 'CLOUDNETpy35']:
        return 1
    elif version in ['CLOUDNET_LIMRAD', 'CLOUDNET']:
        return 3
    else:
        raise ValueError(f'Wrong Cloudnet Version: {version}')


def post_processor_cloudnet_quality_flag(data, cloudnet_status, clodudnet_class, cloudnet_type=''):
    import copy

    container = copy.deepcopy(data)
    GoodRadarLidar = cloudnet_status == get_good_radar_and_lidar_index(cloudnet_type)

    container['var'][GoodRadarLidar] = clodudnet_class[GoodRadarLidar]

    loggers[0].info('Postprocessing status flag done.')
    return container


def post_processor_cloudnet_classes(data, cloudnet_class):
    import copy
    container = copy.deepcopy(data)
    MixedPhase = cloudnet_class == 5
    CloudDroplets = cloudnet_class == 1

    container['var'][MixedPhase] = cloudnet_class[MixedPhase]
    container['var'][CloudDroplets] = cloudnet_class[CloudDroplets]

    loggers[0].info('Postprocessing cloudnet classes done.')
    return container


def post_processor_homogenize(data):
    """
    Homogenization a la Shupe 2007:
        Remove small patches (speckle) from any given mask by checking 5x5 box
        around each pixel, more than half of the points in the box need to be 1
        to keep the 1 at current pixel

    Args:
        data (dict): larda like container containing predicted classes

    Return:
        container (dict): larda like container containing homogenized data

    """

    def gen_one_hot(classes):
        one_hot = np.zeros(len(CLOUDNET_LABELS_))
        for class_ in classes[iT:iT + WSIZE, iR:iR + WSIZE].flatten():
            one_hot[int(class_)] = 1
        return one_hot

    import copy
    container = copy.deepcopy(data)
    classes = container['var']

    mask = classes == 0
    mask_pad = np.pad(mask, (3, 3), 'constant', constant_values=(0, 0))
    mask_out = mask.copy()
    classes_out = classes.copy()

    WSIZE = 7  # 7x7 window
    min_percentage = 0.7
    n_bins = WSIZE * WSIZE
    min_bins = int(min_percentage * n_bins)
    n_ts_pad, n_rg_pad = mask_pad.shape

    loggers[0].info(f'Start Homogenizing')
    for iT, iR in tqdm(product(range(n_ts_pad - WSIZE), range(n_rg_pad - WSIZE)), total=(n_ts_pad - WSIZE) * (n_rg_pad - WSIZE), unit='pixel'):
        if mask[iT, iR]:
            continue  # skip clear sky pixel
        else:
            # If more than 35 of 49 pixels are classified
            # as clear, then the central pixel is set to clear
            if np.sum(mask_pad[iT:iT + WSIZE, iR:iR + WSIZE]) > min_bins:
                mask_out[iT, iR] = True
                continue  # skip isolated pixel (rule 7a shupe 2007)

        # Homogenize
        n_samples_total = np.count_nonzero(gen_one_hot(classes[iT:iT + WSIZE, iR:iR + WSIZE]), axis=0)

        if n_samples_total == 0: continue

        # If the central pixel is not set to clear and there are
        # more than 7 of 49 pixels with the same type as the central
        # pixel, it is left unchanged. (rule 7b shupe 2007)
        if np.any(n_samples_total > 7): continue

        # Otherwise, the central pixel is set
        # to the classification type that is most plentiful in the box.
        # (rule 7c shupe 2007) change to dominant type
        classes_out[iT, iR] = np.argmax(n_samples_total)

    classes_out[mask_out] = 0
    container['mask'], container['var'] = mask_out, classes_out

    return container


def plot_quicklooks(variables, **kwargs):
    import pyLARDA

    larda = pyLARDA.LARDA().connect(variables['campaign'], build_lists=False)
    savenames = {}
    for _i, _name in enumerate(variables['var_name']):
        fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else FIG_SIZE_
        plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else variables['range_interval']
        for sys in variables['system']:
            try:
                loggers[0].info(f"\nloading :: {variables['time_interval'][0]:%A %d. %B %Y - %H:%M:%S} to {variables['time_interval'][1]:%H:%M:%S} from nc.")
                container = larda.read(sys, _name, variables['time_interval'], plot_range)
                fig, ax = pyLARDA.Transformations.plot_timeheight(
                    container,
                    range_interval=plot_range,
                    contour=variables['contour'],
                    fig_size=fig_size,
                    z_converter=variables['var_converter'][_i],
                )
                key_name = f'{sys}-{_name}' if _name in ['CLASS', 'detection_status'] else _name
                savenames[key_name] = f'{variables["case_name"]}-{variables["campaign"]}-{key_name}--{sys}.png'
                fig.savefig(f'{variables["plot_dir"]}/{savenames[key_name]}', dpi=DPI_)
                loggers[0].info(f'plot saved --> {savenames[key_name]}')
            except:
                h.print_traceback(f"no {variables['campaign']} {_name}  {variables['time_interval']} available")

    return savenames


def print_number_of_classes(labels, text='', names=CLOUDNET_LABELS_):
    # numer of samples per class afer removing ice
    n_samples_total = np.count_nonzero(labels, axis=0)
    loggers[0].info(text)
    loggers[0].info(f'{labels.shape[0]:12d}   total')
    for name, n_smp in zip(names, n_samples_total):
        loggers[0].info(f'{n_smp:12d}   {name}')


def import_dataset(case_string_list, case_list_path, data_root='', cloudnet='', **kwargs):
    def load_cloudnet_specific_features_labels(case_string_list, case_list_path, **kwargs):

        feature_set, target_labels, masked_total = [], [], []
        cloudnet_class, cloudnet_status, model_temp = [], [], []

        for icase, case_str in tqdm(enumerate(case_string_list), total=len(case_string_list), unit='files'):

            # gather time interval, etc.
            case = Loader.load_case_list(case_list_path, case_str)
            TIME_SPAN = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
            dt_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'

            flags = np.full((6,), False)
            # check if a mat files is available
            if check_mat_file_availability(kwargs["DATA_PATH"], kwargs["KIND"], dt_str, system=kwargs['SYSTEM'], cloudnet=kwargs['CLOUDNET'],
                                           conv_dim=kwargs['CDIM']):

                _class, flags[0] = load_matv5(f'{kwargs["DATA_PATH"]}/cloudnet/', f'{dt_str}_{kwargs["CLOUDNET"]}_class.mat')
                _status, flags[1] = load_matv5(f'{kwargs["DATA_PATH"]}/cloudnet/', f'{dt_str}_{kwargs["CLOUDNET"]}_status.mat')
                _temperature, flags[2] = load_matv5(f'{kwargs["DATA_PATH"]}/cloudnet/', f'{dt_str}_{kwargs["CLOUDNET"]}_model_T.mat')
                _feature, flags[3] = load_matv5(f'{kwargs["DATA_PATH"]}/features/{kwargs["KIND"]}/{kwargs["CDIM"]}/',
                                                f'{dt_str}_{kwargs["SYSTEM"]}_features_{kwargs["KIND"]}.mat')
                _target, flags[4] = load_matv5(f'{kwargs["DATA_PATH"]}/labels/', f'{dt_str}_{kwargs["SYSTEM"]}_labels.mat')
                _masked, flags[5] = load_matv5(f'{kwargs["DATA_PATH"]}/labels/', f'{dt_str}_{kwargs["SYSTEM"]}_masked.mat')
                if np.sum(flags) > 0: continue

                _feature = _feature['features']
                _target = _target['labels']
                _masked = _masked['masked']

                loggers[0].debug(f'\nloaded :: {TIME_SPAN[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN[1]:%H:%M:%S} mat files')

            else:

                if use_only_given: continue

                _feature, _target, _masked, _class, _status, _temperature = Loader.load_features_from_nc(
                    time_span=TIME_SPAN,
                    voodoo_path=kwargs["VOODOO_PATH"],  # NONSENSE PATH
                    data_path=kwargs["DATA_PATH"],
                    kind=kwargs["KIND"],
                    system=kwargs["SYSTEM"],
                    save=True,
                    n_channels=kwargs["n_channels_"],
                    cloudnet=kwargs["CLOUDNET"],
                )

                loggers[0].debug(f'\nloaded :: {TIME_SPAN[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN[1]:%H:%M:%S} from nc')

            if _masked.all(): continue  # if there are no data points

            if len(_feature.shape) == 3 and kwargs["CDIM"] == 'conv2d': _feature = _feature[:, :, :, np.newaxis]

            if kwargs["TASK"] == 'train':
                """
                select pixel satisfying the following expression:
                training_mask = (   "Good radar & lidar echos" 
                + "Ice & supercooled liquid" 
                + "Cloud droplets only"       ) 
                - "Lidar echos only"

                NOTE: The detection status differs depending on the cloudnet version (matlab/python)!
                """
                training_mask = Loader.load_training_mask(_class, _status, cloudnet_type=kwargs["CLOUDNET"])
                idx_valid_samples = set_intersection(_masked, training_mask)

                if len(idx_valid_samples) < 1: continue

                _feature = _feature[idx_valid_samples, :, :, :] if kwargs["CDIM"] == 'conv2d' else _feature[idx_valid_samples, :, :]
                _target = _target[idx_valid_samples, :]

                """
                flip the CWT on the y-axis to generate a mirror image, 
                the goal is to overcome the miss-classification of updrafts as liquid
                """
                if add_flipped:
                    _feature_flipped = np.zeros(_feature.shape)
                for ismpl, ichan in product(range(len(idx_valid_samples)), range(_feature.shape[-1])):
                    if kwargs["CDIM"] == 'conv2d':
                        _feature_flipped[ismpl, :, :, ichan] = np.fliplr(_feature[ismpl, :, :, ichan])
                else:
                    _feature_flipped[ismpl, :, ichan] = np.flip(_feature[ismpl, :, ichan])

                _feature = np.concatenate((_feature, _feature_flipped), axis=0)
                _target = np.concatenate((_target, _target), axis=0)

            loggers[0].debug(f'\n dim = {_feature.shape}')

            feature_set.append(_feature)
            target_labels.append(_target)
            cloudnet_class.append(_class)
            cloudnet_status.append(_status)
            masked_total.append(_masked)
            model_temp.append(_temperature)

        return feature_set, target_labels, cloudnet_class, cloudnet_status, masked_total, model_temp

    if CLOUDNET in ['CLOUDNETpy94', 'CLOUDNET_LIMRAD']:
        cloudnet_data = [load_cloudnet_specific_features_labels(
            case_string_list, case_list_path, DATA_PATH=f'{data_root}/{cloudnet}', CLOUDNET=cloudnet, **kwargs)
        ]
    else:
        cloudnet_data = [load_cloudnet_specific_features_labels(
            case_string_list, case_list_path, DATA_PATH=f'{data_root}/{cn}', CLOUDNET=cn, **kwargs)
            for cn in ['CLOUDNETpy94', 'CLOUDNET_LIMRAD']
        ]

    feature_set = np.concatenate([i for icn in cloudnet_data for i in icn[0]], axis=0)
    target_labels = np.concatenate([i for icn in cloudnet_data for i in icn[1]], axis=0)
    # concatenate classes and mask for plotting
    if kwargs['TASK'] == 'predict':
        _cn = cloudnet_data[0]
        cloudnet_class = _cn[2]
        cloudnet_status = _cn[3]
        masked_total = np.concatenate(_cn[4], axis=0)
        model_temp = _cn[5]
    else:
        cloudnet_class = None
        cloudnet_status = None
        masked_total = None
        model_temp = None

    print_number_of_classes(target_labels, text=f'\nsamples per class')

    # removing X % of ice pixels
    REMOVE_ICE = 0.8
    if kwargs['TASK'] == 'train' and REMOVE_ICE > 0:
        idx_ice = np.where(target_labels[:, 4] == 1)[0]
        rand_choice = np.random.choice(idx_ice, int(idx_ice.size * REMOVE_ICE))
        feature_set = np.delete(feature_set, rand_choice, axis=0)
        target_labels = np.delete(target_labels, rand_choice, axis=0)
        print_number_of_classes(target_labels, text=f'\nsamples per class after removing {REMOVE_ICE * 100.:.2f}% of ice pixels')

    # splitting into training and validation set
    if kwargs['TASK'] == 'train':
        # take every nth element from the training set for validation
        n = 10
        validation_set = (feature_set[::n], target_labels[::n])
        feature_set = np.array([item for index, item in enumerate(feature_set) if (index + 1) % n != 0])
        target_labels = np.array([item for index, item in enumerate(target_labels) if (index + 1) % n != 0])
        print_number_of_classes(target_labels, text=f'\nsamples per class after removing validation split')
    else:
        validation_set = ()

    return feature_set, target_labels, validation_set, cloudnet_class, cloudnet_status, masked_total, model_temp


def seconds_to_fstring(time_diff):
    return datetime.datetime.fromtimestamp(time_diff).strftime("%M:%S")


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
    CLOUDNET = kwargs['cloudnet'] if 'cloudnet' in kwargs else ''
    PLOT_RANGE_[1] = float(kwargs['range']) if 'range' in kwargs else PLOT_RANGE_[1]

    use_only_given = False

    n_channels_ = 6 if 'HSI' in KIND else 1

    add_flipped = True

    if TASK == 'predict' and not os.path.isfile(f'{MODELS_PATH}/{TRAINED_MODEL}'):
        raise FileNotFoundError(f'Trained model not found! {TRAINED_MODEL}')

    if 'case' in kwargs:
        if len(kwargs['case']) == 17:  # format YYYYMMDD-YYYYMMDD
            use_only_given = True
            case_list_path = VOODOO_PATH + f'/tomls/auto-trainingset-{kwargs["case"]}.toml'
            case_string_list = [case for case in Loader.load_case_file(case_list_path).keys()]
        else:
            case_string_list = [kwargs['case']]
    else:
        raise ValueError('Keyword argument case is missing!')

    # gather todays date
    t0_voodoo = datetime.datetime.today()
    time_str = f'{t0_voodoo:%Y%m%d-%H%M%S}'
    case_plot_path = f'{PLOTS_PATH}/training/{kwargs["case"]}/'

    # load ann model parameter and other global values
    config_global_model = toml.load(VOODOO_PATH + ANN_MODEL_TOML)
    radar_input_setting = config_global_model['feature']['VSpec']
    tf_settings = config_global_model['tensorflow']

    CDIM = tf_settings["USE_MODEL"]

    loggers[0].info(f'\nLoading {CDIM} neural network input......')

    cloudnet_data_kwargs = {
        'VOODOO_PATH': VOODOO_PATH,  # NONSENSE PATH
        'KIND': KIND,
        'SYSTEM': SYSTEM,
        'SAVE': True,
        'n_channels': n_channels_,
        'CDIM': CDIM,
        'TASK': TASK
    }

    feature_set, target_labels, validation_set, cloudnet_class, cloudnet_status, masked_total, model_temp = import_dataset(
        case_string_list, case_list_path, data_root=f'{DATA_PATH}/', cloudnet=CLOUDNET, kind=KIND, **cloudnet_data_kwargs
    )

    ########################################################################################################################################################
    #   ___ ____ ____ _ _  _ _ _  _ ____
    #    |  |__/ |__| | |\ | | |\ | | __
    #    |  |  \ |  | | | \| | | \| |__]
    #
    #
    if TASK == 'train':
        hyper_params = {

            # Convolutional part of the model
            'KIND': KIND,
            'CONV_DIMENSION': tf_settings['USE_MODEL'],

            # I/O dimensions
            'INPUT_DIMENSION': feature_set.shape,
            'OUTPUT_DIMENSION': target_labels.shape,

            # time of creation
            'time_str': time_str,

            # GPU
            'DEVICE': 0
        }

        # create file name and add MODEL_PATH and LOGS_PATH to hyper_parameter dict
        hyper_params.update(create_filename(TRAINED_MODEL, **hyper_params))
        hyper_params.update(config_global_model['tensorflow'])
        hyper_params.update(config_global_model['feature'])
        hyper_params.update({'cloudnet': CLOUDNETs})

        idx = hyper_params["MODEL_PATH"].rfind('/')
        Utils.write_ann_config_file(
            name=f"{hyper_params['MODEL_PATH'][idx + 1:-3]}.json",
            path=hyper_params["MODEL_PATH"][:idx + 1],
            **hyper_params
        )

        # define a new model or load an existing one
        cnn_model = Model.define_convnet(feature_set.shape[1:], target_labels.shape[1:], **hyper_params)

        # parse the training set to the optimizer
        history = Model.training(cnn_model, feature_set, target_labels, validation=validation_set, **hyper_params)

        # create directory for plots
        fig, _ = Plot.History(history)
        Plot.save_figure(
            fig,
            path=f'{PLOTS_PATH}/training/',
            name=f'histo_loss-acc_{time_str}__{hyper_params["MODEL_PATH"][idx + 1:-3]}.png',
            dpi=300
        )

    ############################################################################################################################################################
    #   ___  ____ ____ ___  _ ____ ___ _ ____ _  _
    #   |__] |__/ |___ |  \ | |     |  | |  | |\ |
    #   |    |  \ |___ |__/ | |___  |  | |__| | \|
    #
    if TASK == 'predict':

        hyper_params = {'DEVICE': 0}

        # make predictions using the following model
        from json2html import *

        ann_params_info = Utils.read_ann_config_file(name=f'{TRAINED_MODEL[:-3]}.json', path=MODELS_PATH, **hyper_params)
        case_study_info = {'html_params': json2html.convert(json=ann_params_info)}
        traning_data_info = case_study_info['html_params'].find('')

        # define a new model or load an existing one
        cnn_model = Model.define_convnet(feature_set.shape[1:], target_labels.shape[1:], MODEL_PATH=MODELS_PATH + TRAINED_MODEL, **hyper_params)
        cnn_pred = Model.predict_classes(cnn_model, feature_set)

        prediction2D_classes, prediction2D_probs = Model.one_hot_to_classes(cnn_pred, masked_total)

        prediction_container = container_from_prediction(
            np.concatenate([np.squeeze(iclass['ts']) for iclass in cloudnet_class], axis=0),
            np.squeeze(cloudnet_class[0]['rg']),
            prediction2D_classes,
            masked_total
        )

        prediction_probabilities = container_from_prediction(
            np.concatenate([np.squeeze(iclass['ts']) for iclass in cloudnet_class], axis=0),
            np.squeeze(cloudnet_class[0]['rg']),
            prediction2D_probs,
            masked_total,
            name='probability',
            colormap='viridis',
            var_lims=[0.5, 1.0]
        )

        dt_interval = [h.ts_to_dt(prediction_container['ts'][0]), h.ts_to_dt(prediction_container['ts'][-1])]
        case_plot_path = f'{PLOTS_PATH}/training/{kwargs["case"]}/'
        h.change_dir(case_plot_path)

        contour_T = get_isotherms(model_temp, name='Temperature')
        cloudnet_status_container = variable_to_container(cloudnet_status, name='detection_status')
        cloudnet_class_container = variable_to_container(cloudnet_class, name='CLASS')

        analyser_vars = {
            'campaign': 'lacros_dacapo_gpu',
            'system': ['CLOUDNETpy94', 'CLOUDNET_LIMRAD'],
            'var_name': ['Z', 'VEL', 'width', 'LDR', 'beta', 'CLASS', 'detection_status'],
            'var_converter': ['none', 'none', 'none', 'lin2z', 'log', 'none', 'none'],
            'time_interval': dt_interval,
            'range_interval': PLOT_RANGE_,
            'contour': contour_T,
            'plot_dir': case_plot_path,
            'case_name': kwargs["case"],
        }

        # POST PROCESSOR OFF
        prediction_plot_name_PPoff = f'{kwargs["case"]}-{TRAINED_MODEL}-classification--{"-".join(x for x in CLOUDNETs)}-postprocessor-off.png'
        fig, _ = tr.plot_timeheight(prediction_container, title='', range_interval=PLOT_RANGE_, contour=contour_T, fig_size=FIG_SIZE_)
        fig.savefig(f'{case_plot_path}/{prediction_plot_name_PPoff}', dpi=DPI_)
        loggers[0].info(f'plot saved -->  {prediction_plot_name_PPoff}')

        # POST PROCESSOR OFF, class probabilities
        predprobab_plot_name_PPoff = f'{kwargs["case"]}-{TRAINED_MODEL}-class-probabilities--{"-".join(x for x in CLOUDNETs)}-postprocessor-off.png'
        fig, _ = tr.plot_timeheight(prediction_probabilities, title='', range_interval=PLOT_RANGE_, contour=contour_T, fig_size=FIG_SIZE_)
        fig.savefig(f'{case_plot_path}/{predprobab_plot_name_PPoff}', dpi=DPI_)
        loggers[0].info(f'plot saved -->  {predprobab_plot_name_PPoff}')

        # POST PROCESSOR ON
        prediction_container = post_processor_temperature(
            prediction_container,
            contour_T
        )

        prediction_container = post_processor_homogenize(
            prediction_container,
        )

        prediction_container = post_processor_cloudnet_quality_flag(
            prediction_container,
            cloudnet_status_container['var'],
            cloudnet_class_container['var'],
            cloudnet_type=CLOUDNET
        )

        prediction_container = post_processor_cloudnet_classes(
            prediction_container,
            cloudnet_class_container['var']
        )

        if CLOUDNET == 'CLOUDNET_LIMRAD':
            # the matlab/polly version missclassifies a lot drizzle as aerosol and insects
            prediction_container['var'][cloudnet_class_container['var'] == 10] = 2

        prediction_plot_name_PPon = f'{kwargs["case"]}-{TRAINED_MODEL}-classification--{"-".join(x for x in CLOUDNETs)}-postprocessor-on.png'

        # create directory for plots
        fig, _ = tr.plot_timeheight(prediction_container, title='', range_interval=PLOT_RANGE_, contour=contour_T, fig_size=FIG_SIZE_)
        fig.savefig(f'{case_plot_path}/{prediction_plot_name_PPon}', dpi=DPI_)
        loggers[0].info(f'plot saved --> {case_plot_path}/{prediction_plot_name_PPon}')

        case_study_info['link'] = Utils.get_explorer_link(
            'lacros_dacapo', dt_interval, PLOT_RANGE_,
            ["CLOUDNET|CLASS", "CLOUDNET|Z", "POLLY|attbsc1064", "POLLY|depol"]
        )
        case_study_info['location'] = 'Punta-Arenas, Chile'
        case_study_info['coordinates'] = [-53.1354, -70.8845]
        case_study_info['plot_dir'] = analyser_vars['plot_dir']
        case_study_info['case_name'] = analyser_vars['case_name']
        case_study_info['time_interval'] = analyser_vars['time_interval']
        case_study_info['range_interval'] = analyser_vars['range_interval']

        png_names_cloudnet = plot_quicklooks(analyser_vars)

        analyser_vars_polly = {
            'campaign': 'lacros_dacapo_gpu',
            'system': ['POLLYNET'],
            'var_name': ['attbsc1064', 'attbsc532', 'attbsc355', 'voldepol532'],
            'var_converter': ['log', 'log', 'log', 'none'],
            'time_interval': dt_interval,
            'range_interval': PLOT_RANGE_,
            'contour': contour_T,
            'plot_dir': case_plot_path,
            'case_name': kwargs["case"]
        }
        png_names_polly = plot_quicklooks(analyser_vars_polly)

        png_names = {**png_names_cloudnet, **png_names_polly}
        png_names['prediction_PPoff'] = prediction_plot_name_PPoff
        png_names['prediction_PPon'] = prediction_plot_name_PPon

        Utils.make_html_overview(VOODOO_PATH, case_study_info, png_names)

    ####################################################################################################################################
    loggers[0].info(f'\n        *****Done*****, elapsed time = {datetime.timedelta(seconds=int(time.time() - start_time))} [min:sec]')
