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

import traceback

import datetime
import logging
import numpy as np
import os
import sys
import time
import toml
import xarray as xr
from itertools import product
from tqdm.auto import tqdm

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')
import pyLARDA
import matplotlib
import matplotlib.pyplot as plt

import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model
import voodoo.libVoodoo.Utils  as Utils

import voodoo.generate_trainingset as Loader

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
PLOT_RANGE_ = [0, 12000]
USE_ONLY_GIVEN = True
add_flipped = False
N_VAL = 10

FIG_SIZE_ = [12, 7]
DPI_ = 200
_FONT_SIZE = 12
_FONT_WEIGHT = 'semibold'

# list of cloudnet data sets used for training
CLOUDNETs = ['CLOUDNETpy94']

ANN_MODEL_TOML = 'ann_model_setting.toml'
DATA_PATH = f'{VOODOO_PATH}/data/'
LOGS_PATH = f'{VOODOO_PATH}/logs/'
MODELS_PATH = f'{VOODOO_PATH}/models/'
PLOTS_PATH = f'{VOODOO_PATH}/plots/'

_CLOUDNET_LABELS = [
    'Clear sky',
    'Cloud liquid droplets only',
    'Drizzle or rain.',
    'Drizzle/rain & cloud droplet',
    'Ice particles.',
    'Ice coexisting with supercooled liquid droplets.',
    'Melting ice particles',
    'Melting ice & cloud droplets',
    'Aerosol & || insects',
    '-',
    '-'
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


def check_zarr_file_availability(data_path, dt_str, system=SYSTEM):
    # data/CLOUDNETpy94/xarray/20190410_2015-2030_limrad94.zarr
    if not os.path.isdir(f'{data_path}/xarray/{dt_str}_{system}.zarr'):
        loggers[0].info(f"{data_path}/xarray/{dt_str}_{system}.zarr'  not found!")
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


def container_from_prediction(ts, rg, var, mask, **kwargs):
    prediction_container = {}
    prediction_container['dimlabel'] = ['time', 'range']
    prediction_container['name'] = kwargs['name'] if 'name' in kwargs else 'CLASS'
    prediction_container['joints'] = ''
    prediction_container['rg_unit'] = 'm'
    prediction_container['colormap'] = kwargs['colormap'] if 'colormap' in kwargs else 'cloudnet_target_new'
    prediction_container['system'] = 'Voodoo'
    prediction_container['ts'] = ts
    prediction_container['rg'] = rg
    prediction_container['var_lims'] = kwargs['var_lims'] if 'var_lims' in kwargs else [0, 8]
    prediction_container['var_unit'] = '1'
    prediction_container['mask'] = mask
    prediction_container['var'] = var
    return prediction_container


def get_isotherms(temperature, ts, rg, mask, **kwargs):
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
    T['ts'] = ts
    T['rg'] = rg
    T['var_lims'] = [240.0, 320.0]
    T['var_unit'] = 'K'
    T['mask'] = mask
    T['var'] = temperature

    return {'data': tr.combine(toC, [T], {'var_unit': "C"}), 'levels': np.arange(-40, 16, 5)}


def variable_to_container(var, ts, rg, mask, **kwargs):
    container = {}
    container['dimlabel'] = ['time', 'range']
    container['name'] = kwargs['name']
    container['joints'] = ''
    container['paraminfo'] = ''
    container['filename'] = 'ann_input_files'
    container['rg_unit'] = 'm'
    container['colormap'] = 'cloudnet_jet'
    container['system'] = kwargs['CLOUDNET'] if 'CLOUDNET' in kwargs else 'unknown'
    container['ts'] = ts
    container['rg'] = rg
    container['var_lims'] = [240.0, 320.0]
    container['var_unit'] = 'K'
    container['mask'] = mask
    container['var'] = var
    return container


def post_processor_temperature(data, contour):
    import copy
    container = copy.deepcopy(data)
    melting_temp = 2.5  # °C
    idx_Tplus_ice = (contour['data']['var'] > melting_temp) * (container['var'] == 4)
    container['var'][idx_Tplus_ice] = 2

    idx_Tplus_mixed = (contour['data']['var'] > melting_temp) * (container['var'] == 5)
    container['var'][idx_Tplus_mixed] = 3  # set to drizzle/rain & cloud droplets

    idx_droplets_mixed = ((container['var'] == 1) + (container['var'] == 5))
    idx_hetero_freezing = (contour['data']['var'] < -40.0)
    container['var'][idx_hetero_freezing * idx_droplets_mixed] = 4  # set to ice

    idx_Tneg0_drizzle = (contour['data']['var'] < melting_temp) * (container['var'] == 2)
    container['var'][idx_Tneg0_drizzle] = 4  # set to ice

    loggers[0].info('Postprocessing temperature info done.')

    return container


def post_processor_cloudnet_quality_flag(data, cloudnet_status, clodudnet_class, cloudnet_type=''):
    import copy

    container = copy.deepcopy(data)
    GoodRadarLidar = cloudnet_status == Loader.get_good_radar_and_lidar_index(cloudnet_type)
    GoodLidarOnly = cloudnet_status == Loader.get_good_lidar_only_index(cloudnet_type)

    container['var'][GoodRadarLidar] = clodudnet_class[GoodRadarLidar]
    container['var'][GoodLidarOnly] = clodudnet_class[GoodLidarOnly]

    if cloudnet_type in ['CLOUDNET', 'CLOUDNET_LIMRAD']:
        KnownAttenuation = cloudnet_status == 6
        container['var'][KnownAttenuation] = clodudnet_class[KnownAttenuation]

    loggers[0].info('Postprocessing status flag done.')
    return container


def post_processor_cloudnet_classes(data, cloudnet_class):
    import copy
    container = copy.deepcopy(data)
    MixedPhase = cloudnet_class == 5
    CloudDroplets = cloudnet_class == 1
    Drizzle = cloudnet_class == 2
    MeltingLayer = (cloudnet_class == 6) + (cloudnet_class == 7)

    container['var'][MixedPhase] = cloudnet_class[MixedPhase]
    container['var'][CloudDroplets] = cloudnet_class[CloudDroplets]
    container['var'][Drizzle] = cloudnet_class[Drizzle]
    container['var'][MeltingLayer] = cloudnet_class[MeltingLayer]

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

    WSIZE = 7  # 7x7 window

    def gen_one_hot(classes):
        one_hot = np.zeros(len(_CLOUDNET_LABELS))
        for class_ in classes[iT:iT + WSIZE, iR:iR + WSIZE].flatten():
            one_hot[int(class_)] = 1
        return one_hot

    import copy
    container = copy.deepcopy(data)
    classes = container['var']

    n_dim = WSIZE // 2
    mask = classes == 0
    mask_pad = np.pad(mask, (n_dim, n_dim), 'constant', constant_values=(0, 0))
    mask_out = mask.copy()
    classes_out = classes.copy()

    min_percentage = 0.8
    min_bins = WSIZE * WSIZE * int(min_percentage)
    n_ts_pad, n_rg_pad = mask_pad.shape

    loggers[0].info(f'Start Homogenizing')
    for iT, iR in tqdm(product(range(n_ts_pad - WSIZE), range(n_rg_pad - WSIZE)), total=(n_ts_pad - WSIZE) * (n_rg_pad - WSIZE), unit='pixel'):
        if mask[iT, iR]:
            continue  # skip clear sky pixel
        #        else:
        #            # If more than 35 of 49 pixels are classified
        #            # as clear, then the central pixel is set to clear
        #            if np.sum(mask_pad[iT:iT + WSIZE, iR:iR + WSIZE]) > min_bins:
        #                mask_out[iT, iR] = True
        #                continue  # skip isolated pixel (rule 7a shupe 2007)

        # Homogenize
        n_samples_total = np.count_nonzero(gen_one_hot(classes[iT:iT + WSIZE, iR:iR + WSIZE]), axis=0)

        if n_samples_total == 0: continue

        # If the central pixel is not set to clear and there are
        # more than 7 of 49 pixels with the same type as the central
        # pixel, it is left unchanged. (rule 7b shupe 2007)
        if np.any(n_samples_total > min_bins): continue

        # Otherwise, the central pixel is set
        # to the classification type that is most plentiful in the box.
        # (rule 7c shupe 2007) change to dominant type
        classes_out[iT, iR] = np.argmax(n_samples_total)

    classes_out[mask_out] = 0
    container['mask'], container['var'] = mask_out, classes_out

    return container


def plot_quicklooks(variables, **kwargs):
    larda = pyLARDA.LARDA().connect(variables['campaign'], build_lists=False)
    savenames = {}
    for _i, _name in enumerate(variables['var_name']):
        fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else FIG_SIZE_
        plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else variables['range_interval']
        for sys in variables['system']:
            try:
                loggers[0].info(f"\nloading :: {variables['time_interval'][0]:%A %d. %B %Y - %H:%M:%S} to {variables['time_interval'][1]:%H:%M:%S} from nc.")
                container = larda.read(sys, _name, variables['time_interval'], plot_range)
                # container['var'] = np.ma.masked_where(container['var'], container['mask'])

                fig, ax = pyLARDA.Transformations.plot_timeheight(
                    container,
                    range_interval=plot_range,
                    contour=variables['contour'],
                    fig_size=fig_size,
                    z_converter=variables['var_converter'][_i],
                    rg_converter=True,
                    font_size=_FONT_SIZE,
                    font_weight=_FONT_WEIGHT,
                )
                key_name = f'{sys}-{_name}' if _name in ['CLASS', 'detection_status'] else _name
                savenames[key_name] = f'{variables["case_name"]}-{variables["campaign"]}-{key_name}--{sys}.png'
                fig.savefig(f'{variables["plot_dir"]}/{savenames[key_name]}', dpi=DPI_)
                matplotlib.pyplot.close(fig=fig)
                loggers[0].info(f'plot saved --> {savenames[key_name]}')
            except:
                h.print_traceback(f"no {variables['campaign']} {_name}  {variables['time_interval']} available")

    return savenames


def print_number_of_classes(labels, text='', names=_CLOUDNET_LABELS):
    # numer of samples per class afer removing ice
    loggers[0].info(text)
    loggers[0].info(f'{labels.shape[0]:12d}   total')
    for i, name in enumerate(names):
        loggers[0].info(f'{np.sum(labels == i):12d}   {name}')


def import_dataset(case_string_list, case_list_path, data_root='', cloudnet='', remove_ice=0.0, **kwargs):
    def load_cloudnet_specific_features_labels(case_string_list, case_list_path, **kwargs):

        feature_set, target_labels, masked_total = [], [], []
        cloudnet_class, cloudnet_status, model_temp, ts_cloudnet, rg_cloundet = [], [], [], [], []

        for icase, case_str in tqdm(enumerate(case_string_list), total=len(case_string_list), unit='files'):

            # gather time interval, etc.
            case = Loader.load_case_list(case_list_path, case_str)
            TIME_SPAN = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
            dt_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'

            # check if a mat files is available
            try:
                with xr.open_zarr(f'{kwargs["DATA_PATH"]}/xarray/{dt_str}_{kwargs["SYSTEM"]}.zarr') as zarr_data:
                    _class = zarr_data['CLASS'].values if 'CLASS' in zarr_data else []
                    _status = zarr_data['detection_status'].values if 'detection_status' in zarr_data else []
                    _temperature = zarr_data['T'].values if 'T' in zarr_data else []
                    _feature = zarr_data['features'].values
                    _target = zarr_data['targets'].values
                    _masked = zarr_data['masked'].values
                    _ts = zarr_data['ts'].values
                    _rg = zarr_data['rg'].values

                    loggers[0].debug(f'\nloaded :: {TIME_SPAN[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN[1]:%H:%M:%S} zarr files')

            except FileNotFoundError:
                loggers[0].info(f"{kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['SYSTEM']}.zarr  not found!")
                loggers[0].info(f'USE_ONLY_GIVEN: {USE_ONLY_GIVEN}')

                if USE_ONLY_GIVEN: continue
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

            except ValueError as e:
                if 'group not found at path' in str(e):
                    loggers[0].info(f"{kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['SYSTEM']}.zarr  not found!")
                else:
                    loggers[0].info(f"{kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['SYSTEM']}.zarr  some value is missing!")
                    loggers[0].info(f"{e}")
                continue

            except Exception as e:
                loggers[0].critical(f"Unexpected error: {sys.exc_info()[0]}\n Check folder: {kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['SYSTEM']}.zarr")
                exc_type, exc_value, exc_tb = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_tb)
                loggers[0].critical(f'Exception: Check ~/{kwargs["DATA_PATH"]}/xarray/{dt_str}_{kwargs["SYSTEM"]}.zarr)')
                loggers[0].critical(f'{e}')
                continue

            if _masked.all(): continue  # if there are no data points

            if len(_feature.shape) == 3 and kwargs["CDIM"] == 'conv2d': _feature = _feature[:, :, :, np.newaxis]

            # apply training mask
            if kwargs["TASK"] == 'train':
                """
                select pixel satisfying the following expression:
                training_mask = (   "Good radar & lidar echos" 
                + "Ice & supercooled liquid" 
                + "Cloud droplets only"       ) 
                - "Lidar echos only"

                NOTE: The detection status differs depending on the cloudnet version (matlab/python)!
                """

                if (_target == -999.0).all(): continue  # if there are no labels available
                training_mask = Loader.load_training_mask(_class, _status, cloudnet_type=kwargs["CLOUDNET"])
                idx_valid_samples = set_intersection(_masked, training_mask)

                if len(idx_valid_samples) < 1: continue

                _feature = _feature[idx_valid_samples, :, :]
                _target = _target[idx_valid_samples, np.newaxis]

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
            ts_cloudnet.append(_ts)

        return feature_set, target_labels, cloudnet_class, cloudnet_status, masked_total, model_temp, ts_cloudnet, _rg

    if CLOUDNET in ['CLOUDNETpy94', 'CLOUDNET_LIMRAD']:
        cloudnet_data = [load_cloudnet_specific_features_labels(
            case_string_list, case_list_path, DATA_PATH=f'{data_root}/{cloudnet}', CLOUDNET=cloudnet, **kwargs)
        ]
    else:
        cloudnet_data = [load_cloudnet_specific_features_labels(
            case_string_list, case_list_path, DATA_PATH=f'{data_root}/{cn}', CLOUDNET=cn, **kwargs)
            for cn in CLOUDNETs
        ]

    feature_set = np.concatenate([i for icn in cloudnet_data for i in icn[0]], axis=0)
    target_labels = np.concatenate([i for icn in cloudnet_data for i in icn[1]], axis=0)

    # concatenate classes and mask for plotting
    if kwargs['TASK'] == 'predict':
        _cn = cloudnet_data[0]
        cloudnet_class = np.concatenate(_cn[2], axis=0)
        cloudnet_status = np.concatenate(_cn[3], axis=0)
        masked_total = np.concatenate(_cn[4], axis=0)
        model_temp = np.concatenate(_cn[5], axis=0)
        cloudnet_ts = np.concatenate(_cn[6], axis=0)
        cloudnet_rg = _cn[7]
    else:
        cloudnet_class = None
        cloudnet_status = None
        masked_total = None
        model_temp = None
        cloudnet_ts = None
        cloudnet_rg = None

    print_number_of_classes(target_labels, text=f'\nsamples per class')

    # removing X % of ice pixels
    if kwargs['TASK'] == 'train' and remove_ice > 0:
        idx_ice = np.where(target_labels == 4)[0]
        rand_choice = np.random.choice(idx_ice, int(idx_ice.size * remove_ice))
        feature_set = np.delete(feature_set, rand_choice, axis=0)
        target_labels = np.delete(target_labels, rand_choice, axis=0)
        print_number_of_classes(target_labels, text=f'\nsamples per class after removing {remove_ice * 100.:.2f}% of ice pixels')

    # splitting into training and validation set
    validation_set = ()
    if kwargs['TASK'] == 'train':
        # take every nth element from the training set for validation
        validation_set = (feature_set[::N_VAL], target_labels[::N_VAL])
        feature_set = np.array([item for index, item in enumerate(feature_set) if (index + 1) % N_VAL != 0])
        target_labels = np.array([item for index, item in enumerate(target_labels) if (index + 1) % N_VAL != 0])
        print_number_of_classes(target_labels, text=f'\nsamples per class after removing validation split')

    return feature_set, np.squeeze(target_labels), validation_set, cloudnet_class, cloudnet_status, masked_total, model_temp, cloudnet_ts, cloudnet_rg


def seconds_to_fstring(time_diff):
    return datetime.datetime.fromtimestamp(time_diff).strftime("%M:%S")


def sum_liquid_layer_thickness(liquid_pixel_mask, rg_res=30.0):
    """Calculating the liquid layer thickness of the total vertical column"""
    return np.sum(liquid_pixel_mask, axis=1) * rg_res


def get_liquid_pixel_mask(classes):
    return (classes == 1) + (classes == 2) + (classes == 3) + (classes == 5) + (classes == 7)


def ma_corr_coef(X1, X2):
    return np.ma.corrcoef(np.ma.masked_less_equal(X1, 0.0), np.ma.masked_less_equal(X2, 0.0))[0, 1]


def add_lwp_to_classification(prediction, classification, fig, ax, cloudnet=''):
    # add the lwp ontop
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=False)
    lwp_container = larda.read(cloudnet, 'LWP', dt_interval, PLOT_RANGE_)
    lwp_container = tr.interpolate1d(lwp_container, new_time=prediction['ts'], new_rg=prediction['rg'])
    dt_lwp = [h.ts_to_dt(ts) for ts in lwp_container['ts']]

    ax.set_xlim([h.ts_to_dt(lwp_container['ts'][0]), h.ts_to_dt(lwp_container['ts'][-1])])
    lwp_ax = Plot._plot_bar_data(fig, ax, lwp_container['var'], dt_lwp)

    sum_ll_thickness_nn = sum_liquid_layer_thickness(get_liquid_pixel_mask(prediction['var']), rg_res=prediction['rg'][1] - prediction['rg'][0])
    sum_ll_thickness_cn = sum_liquid_layer_thickness(get_liquid_pixel_mask(classification['var']), rg_res=prediction['rg'][1] - prediction['rg'][0])
    Plot.plot_ll_thichkness(lwp_ax, [h.ts_to_dt(ts) for ts in prediction['ts']], sum_ll_thickness_nn, sum_ll_thickness_cn)

    # these are matplotlib.patch.Patch properties
    props = {
        'transform': ax.transAxes,
        'fontsize': _FONT_SIZE,
        'verticalalignment': 'top',
        'bbox': dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    }

    corr_lwp_nn = r'$R_{lwp-nn}^2=$' + f'{ma_corr_coef(lwp_container["var"], sum_ll_thickness_nn):.3f}'
    corr_lwp_cn = r'$R_{lwp-cn}^2=$' + f'{ma_corr_coef(lwp_container["var"], sum_ll_thickness_cn):.3f}'
    # place a text box in upper left in axes coords
    loggers[0].info('------ CORRELATIONS ------')
    ax.text(1.1, 1.6, corr_lwp_nn, **props)
    ax.text(1.1, 1.5, corr_lwp_cn, **props)
    loggers[0].info(f'correlation mwr-lwp vs. neural network liquid containing range gates :: {corr_lwp_nn}')
    loggers[0].info(f'correlation mwr-lwp vs. cloudnet liquid containing range gates :: {corr_lwp_cn}')

    lwp_smoothed5min = h.smooth(lwp_container['var'], 10)  # 10 bins = 5 min

    corr_lwp_nn_smoohed = r'$\tilde{R}_{lwp-nn}^2=$' + f'{ma_corr_coef(lwp_smoothed5min, h.smooth(sum_ll_thickness_nn, 10)):.3f}'
    corr_lwp_cn_smoohed = r'$\tilde{R}_{lwp-cn}^2=$' + f'{ma_corr_coef(lwp_smoothed5min, h.smooth(sum_ll_thickness_cn, 10)):.3f}'
    # place a text box in upper left in axes coords
    ax.text(1.1, 1.4, corr_lwp_nn_smoohed, **props)
    ax.text(1.1, 1.3, corr_lwp_cn_smoohed, **props)
    loggers[0].info(f'correlation 5min smoothed mwr-lwp vs. neural network liquid containing range gates :: {corr_lwp_nn_smoohed}')
    loggers[0].info(f'correlation 5min smoothed mwr-lwp vs. cloudnet liquid containing range gates :: {corr_lwp_cn_smoohed}')

    return fig, ax


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

    n_channels_ = 6 if 'HSI' in KIND else 1

    if TASK == 'predict' and not os.path.isfile(f'{MODELS_PATH}/{TRAINED_MODEL}'):
        raise FileNotFoundError(f'Trained model not found! {TRAINED_MODEL}')

    if 'case' in kwargs:
        if len(kwargs['case']) == 17:  # format YYYYMMDD-YYYYMMDD
            USE_ONLY_GIVEN = True
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

    feature_set, target_labels, validation_set, cloudnet_class, cloudnet_status, masked_total, model_temp, cloudnet_ts, cloudnet_rg = import_dataset(
        case_string_list, case_list_path, data_root=f'{DATA_PATH}/', cloudnet=CLOUDNET, kind=KIND, **cloudnet_data_kwargs
    )

    ########################################################################################################################################################
    #   ___ ____ ____ _ _  _ _ _  _ ____
    #    |  |__/ |__| | |\ | | |\ | | __
    #    |  |  \ |  | | | \| | | \| |__]
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
        cnn_model = Model.define_convnet(feature_set.shape[1:], 9, **hyper_params)

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
        cnn_model = Model.define_convnet(feature_set.shape[1:], (9,), MODEL_PATH=MODELS_PATH + TRAINED_MODEL, **hyper_params)
        cnn_pred = Model.predict_classes(cnn_model, feature_set)

        prediction2D_classes, prediction2D_probs = Model.one_hot_to_classes(cnn_pred, masked_total)

        prediction_container = container_from_prediction(
            np.copy(cloudnet_ts),
            np.copy(cloudnet_rg),
            np.copy(prediction2D_classes),
            np.copy(masked_total)
        )

        prediction_probabilities = container_from_prediction(
            np.copy(cloudnet_ts),
            np.copy(cloudnet_rg),
            prediction2D_probs,
            masked_total,
            name='probability',
            colormap='viridis',
            var_lims=[0.5, 1.0]
        )

        dt_interval = [h.ts_to_dt(prediction_container['ts'][0]), h.ts_to_dt(prediction_container['ts'][-1])]
        case_plot_path = f'{PLOTS_PATH}/training/{kwargs["case"]}/'
        h.change_dir(case_plot_path)

        if cloudnet_class.size > 0:
            contour_T = get_isotherms(model_temp, cloudnet_ts, cloudnet_rg, masked_total, name='Temperature')
            cloudnet_status_container = variable_to_container(cloudnet_status, cloudnet_ts, cloudnet_rg, masked_total, name='detection_status')
            cloudnet_class_container = variable_to_container(cloudnet_class, cloudnet_ts, cloudnet_rg, masked_total, name='CLASS')

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
        else:
            contour_T = None
        # ---------------------------
        # POST PROCESSOR OFF, class probabilities
        predprobab_plot_name_PPoff = f'{kwargs["case"]}-{TRAINED_MODEL}-class-probabilities--{"-".join(x for x in CLOUDNETs)}-postprocessor-off.png'
        fig_P, _ = tr.plot_timeheight(
            prediction_probabilities,
            title='',
            range_interval=PLOT_RANGE_,
            contour=contour_T,
            fig_size=FIG_SIZE_,
            rg_converter=True,
            font_size=_FONT_SIZE,
            font_weight=_FONT_WEIGHT,
        )
        fig_P.savefig(f'{case_plot_path}/{predprobab_plot_name_PPoff}', dpi=DPI_)
        matplotlib.pyplot.close(fig=fig_P)
        loggers[0].info(f'plot saved -->  {predprobab_plot_name_PPoff}')

        # ---------------------------
        # POST PROCESSOR OFF
        fig_size_plus_extra = np.copy(FIG_SIZE_)
        if cloudnet_class.size > 0:
            fig_size_plus_extra[1] = fig_size_plus_extra[1] + 3

        prediction_plot_name_PPoff = f'{kwargs["case"]}-{TRAINED_MODEL}-classification--{"-".join(x for x in CLOUDNETs)}-postprocessor-off.png'
        fig_raw_pred, ax_raw_pred = tr.plot_timeheight(
            prediction_container,
            title='',
            range_interval=PLOT_RANGE_,
            contour=contour_T,
            fig_size=fig_size_plus_extra,
            rg_converter=True,
            font_size=_FONT_SIZE,
            font_weight=_FONT_WEIGHT,
        )

        if cloudnet_class.size > 0:
            fig_raw_pred.tight_layout(rect=[0., 0., 1.0, .65])
            fig_raw_pred, ax_raw_pred = add_lwp_to_classification(prediction_container, cloudnet_class_container, fig_raw_pred, ax_raw_pred, cloudnet=CLOUDNET)
        fig_raw_pred.savefig(f'{case_plot_path}/{prediction_plot_name_PPoff}', dpi=DPI_)
        matplotlib.pyplot.close(fig=fig_raw_pred)
        loggers[0].info(f'plot saved -->  {prediction_plot_name_PPoff}')

        if cloudnet_class.size < 1:
            print('exit')
            sys.exit(0)

        # POST PROCESSOR ON
        prediction_container = post_processor_temperature(
            prediction_container,
            contour_T
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

        prediction_container = post_processor_homogenize(
            prediction_container,
        )

        prediction_plot_name_PPon = f'{kwargs["case"]}-{TRAINED_MODEL}-classification--{"-".join(x for x in CLOUDNETs)}-postprocessor-on.png'

        # create directory for plots
        fig, ax = tr.plot_timeheight(
            prediction_container,
            title='',
            range_interval=PLOT_RANGE_,
            contour=contour_T,
            fig_size=fig_size_plus_extra,
            rg_converter=True
        )

        fig.tight_layout(rect=[0., 0., 1.0, .65])
        fig, ax = add_lwp_to_classification(prediction_container, cloudnet_class_container, fig, ax, cloudnet=CLOUDNET)

        fig.savefig(f'{case_plot_path}/{prediction_plot_name_PPon}', dpi=DPI_)
        matplotlib.pyplot.close(fig=fig)
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
