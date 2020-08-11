import traceback

import logging
import numpy as np
import pyLARDA.helpers as h
import re
import subprocess
import sys
from itertools import product
from jinja2 import Template
from pyLARDA.Transformations import combine
from tqdm.auto import tqdm
import datetime
import toml
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.StreamHandler())

def read_cmd_line_args(argv=None):
    """Command-line -> method call arg processing.

    - positional args:
            a b -> method('a', 'b')
    - intifying args:
            a 123 -> method('a', 123)
    - json loading args:
            a '["pi", 3.14, null]' -> method('a', ['pi', 3.14, None])
    - keyword args:
            a foo=bar -> method('a', foo='bar')
    - using more of the above
            1234 'extras=["r2"]'  -> method(1234, extras=["r2"])

    @param argv {list} Command line arg list. Defaults to `sys.argv`.
    @returns (<method-name>, <args>, <kwargs>)

    Reference: http://code.activestate.com/recipes/577122-transform-command-line-arguments-to-args-and-kwarg/
    """
    import json
    import sys
    if argv is None:
        argv = sys.argv

    method_name, arg_strs = argv[0], argv[1:]
    args = []
    kwargs = {}
    for s in arg_strs:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            key, value = None, s
        try:
            value = json.loads(value)
        except ValueError:
            pass
        if key:
            kwargs[key] = value
        else:
            args.append(value)
    return method_name, args, kwargs

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def write_ann_config_file(**kwargs):
    """
    Creates at folder and saves a matplotlib figure.

    Args:
        fig (matplotlib figure): figure to save as png

    Keyword Args:
        dpi (int): dots per inch
        name (string): name of the png
        path (string): path where the png is stored

    Returns:    0

    """
    path = kwargs['path'] if 'path' in kwargs else ''
    name = kwargs['name'] if 'name' in kwargs else 'no-name.cfg'
    if len(path) > 0: h.change_dir(path)

    import json
    with open(f"{name}", 'wt', encoding='utf8') as out:
        json.dump(kwargs, out, sort_keys=True, indent=4, ensure_ascii=False)
    print(f'Saved ann configure file :: {name}')
    return 0

def read_ann_config_file(**kwargs):
    """
    Creates at folder and saves a matplotlib figure.

    Args:
        fig (matplotlib figure): figure to save as png

    Keyword Args:
        dpi (int): dots per inch
        name (string): name of the png
        path (string): path where the png is stored

    Returns:    0

    """
    path = kwargs['path'] if 'path' in kwargs else ''
    name = kwargs['name'] if 'name' in kwargs else 'no-name.cfg'
    if len(path) > 0: h.change_dir(path)

    import json
    with open(f"{name}", 'r', encoding='utf8') as json_file:
        data = json.load(json_file)
    print(f'Loaded ann configure file :: {name}')
    return data

def make_html_overview(template_loc, case_study_info, png_names):
    print('case_config', case_study_info)
    print('savenames', png_names.keys())

    with open(f'{template_loc}/static_template.html.jinja2') as file_:
        template = Template(file_.read())

        with open(case_study_info['plot_dir'] + 'overview.html', 'w') as f:
            f.write(
                template.render(
                    png_names=png_names,
                    case_study_info=case_study_info,
                )
            )


        """
        <!--
        {% for key, value in data.items() %}
            <li>{{ key }}: {{ value['file_history'] }}</li>
        {% endfor %}
        -->
        """

def get_explorer_link(campaign, time_interval, range_interval, params):
    s = "http://larda.tropos.de/larda3/explorer/{}?interval={}-{}%2C{}-{}&params={}".format(
        campaign, h.dt_to_ts(time_interval[0]), h.dt_to_ts(time_interval[1]),
        *range_interval, ",".join(params))
    return s

def traceback_error(time_span):
    exc_type, exc_value, exc_tb = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_tb)
    logger.error(ValueError(f'Something went wrong with this interval: {time_span}'))


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

    return {'data': combine(toC, [T], {'var_unit': "C"}), 'levels': np.arange(-40, 16, 5)}


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

    logger.info('Postprocessing temperature info done.')

    return container


def get_good_radar_and_lidar_index(version):
    if version in ['CLOUDNETpy94', 'CLOUDNETpy35']:
        return 1
    elif version in ['CLOUDNET_LIMRAD', 'CLOUDNET']:
        return 3
    else:
        raise ValueError(f'Wrong Cloudnet Version: {version}')


def get_good_lidar_only_index(version):
    if version in ['CLOUDNETpy94', 'CLOUDNETpy35']:
        return 3
    elif version in ['CLOUDNET_LIMRAD', 'CLOUDNET']:
        return 1
    else:
        raise ValueError(f'Wrong Cloudnet Version: {version}')


def post_processor_cloudnet_quality_flag(data, cloudnet_status, clodudnet_class, cloudnet_type=''):
    import copy

    container = copy.deepcopy(data)
    GoodRadarLidar = cloudnet_status == get_good_radar_and_lidar_index(cloudnet_type)
    GoodLidarOnly = cloudnet_status == get_good_lidar_only_index(cloudnet_type)

    container['var'][GoodRadarLidar] = clodudnet_class[GoodRadarLidar]
    container['var'][GoodLidarOnly] = clodudnet_class[GoodLidarOnly]

    if cloudnet_type in ['CLOUDNET', 'CLOUDNET_LIMRAD']:
        KnownAttenuation = cloudnet_status == 6
        container['var'][KnownAttenuation] = clodudnet_class[KnownAttenuation]

    logger.info('Postprocessing status flag done.')
    return container


def post_processor_cloudnet_classes(data, cloudnet_class):
    import copy
    container = copy.deepcopy(data)
    CloudDroplets = cloudnet_class == 1
    Drizzle = cloudnet_class == 2
    DrizzleCloudDroplets = cloudnet_class == 3
    MixedPhase = cloudnet_class == 5
    MeltingLayer = (cloudnet_class == 6) + (cloudnet_class == 7)

    container['var'][MixedPhase] = cloudnet_class[MixedPhase]
    container['var'][CloudDroplets] = cloudnet_class[CloudDroplets]
    container['var'][Drizzle] = cloudnet_class[Drizzle]
    container['var'][MeltingLayer] = cloudnet_class[MeltingLayer]
    container['var'][DrizzleCloudDroplets] = cloudnet_class[DrizzleCloudDroplets]

    logger.info('Postprocessing cloudnet classes done.')
    return container


def post_processor_homogenize(data, nlabels):
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
        one_hot = np.zeros(nlabels)
        for class_ in classes[iT:iT + WSIZE, iR:iR + WSIZE].flatten():
            one_hot[int(class_)] = 1
        return one_hot

    import copy
    container = copy.deepcopy(data)
    classes = container['var']

    n_dim = WSIZE // 2
    mask = classes == 0
    mask_pad = np.pad(mask, (n_dim, n_dim), 'constant', constant_values=(0, 0))
    classes_out = classes.copy()

    min_percentage = 0.8
    min_bins = WSIZE * WSIZE * int(min_percentage)
    n_ts_pad, n_rg_pad = mask_pad.shape

    logger.info(f'Start Homogenizing')
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

    classes_out[mask] = 0
    container['var'] = classes_out

    return container


def sum_liquid_layer_thickness(liquid_pixel_mask, rg_res=30.0):
    """Calculating the liquid layer thickness of the total vertical column"""
    return np.sum(liquid_pixel_mask, axis=1) * rg_res


def get_liquid_pixel_mask(classes):
    return (classes == 1) + (classes == 2) + (classes == 3) + (classes == 5) + (classes == 7)


def ma_corr_coef(X1, X2):
    return np.ma.corrcoef(np.ma.masked_less_equal(X1, 0.0), np.ma.masked_less_equal(X2, 0.0))[0, 1]


def load_training_mask(classes, status, cloudnet_type):
    idx_good_radar_and_lidar = get_good_radar_and_lidar_index(cloudnet_type)
    idx_good_lidar_only = get_good_lidar_only_index(cloudnet_type)

    """
    CLOUDNET
    "0: Clear sky\n
    1: Cloud droplets only\n
    2: Drizzle or rain\n
    3: Drizzle/rain & cloud droplets\n
    4: Ice\n
    5: Ice & supercooled droplets\n
    6: Melting ice\n
    7: Melting ice & cloud droplets\n
    8: Aerosol\n
    9: Insects\n
    10: Aerosol & insects";

    (CLOUDNET MATLAB VERSION)
    "0: Clear sky\n
    1: Lidar echo only\n
    2: Radar echo but uncorrected atten.\n
    3: Good radar & lidar echos\n
    4: No radar but unknown attenuation\n
    5: Good radar echo only\n
    6: No radar but known attenuation\n
    7: Radar corrected for liquid atten.\n
    8: Radar ground clutter\n
    9: Lidar molecular scattering";

    """
    # create mask
    valid_samples = np.full(status.shape, False)
    valid_samples[status == idx_good_radar_and_lidar] = True  # add good radar radar & lidar
    # valid_samples[status == 2]  = True   # add good radar only
    valid_samples[classes == 5] = True  # add mixed-phase class pixel
    # valid_samples[classes == 6] = True   # add melting layer class pixel
    # valid_samples[classes == 7] = True   # add melting layer + SCL class pixel
    valid_samples[classes == 1] = True  # add cloud droplets only class

    # at last, remove lidar only pixel caused by adding cloud droplets only class
    valid_samples[status == idx_good_lidar_only] = False

    # if polly processing, remove aerosol and insects (most actually drizzle)
    if cloudnet_type in ['CLOUDNET_LIMRAD', 'CLOUDNET']:
        valid_samples[classes == 10] = False
    return ~valid_samples


def load_case_file(path):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case']


def load_case_list(path, case_name):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case'][case_name]


def load_dataset_from_zarr(case_string_list, case_list_path, **kwargs):
    N_NOT_AVAILABLE = 0
    feature_set, target_labels, masked_total = [], [], []
    cloudnet_class, cloudnet_status, model_temp, ts_cloudnet, rg_cloundet = [], [], [], [], []

    for icase, case_str in tqdm(enumerate(case_string_list), total=len(case_string_list), unit='files'):

        # gather time interval, etc.
        case = load_case_list(case_list_path, case_str)
        TIME_SPAN = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
        dt_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'

        # check if a mat files is available
        try:
            with xr.open_zarr(f'{kwargs["DATA_PATH"]}/xarray/{dt_str}_{kwargs["RADAR"]}.zarr') as zarr_data:
                _class = zarr_data['CLASS'].values if 'CLASS' in zarr_data else []
                _status = zarr_data['detection_status'].values if 'detection_status' in zarr_data else []
                _temperature = zarr_data['T'].values if 'T' in zarr_data else []
                _feature = zarr_data['features'].values
                _target = zarr_data['targets'].values
                _masked = zarr_data['masked'].values
                _ts = zarr_data['ts'].values
                _rg = zarr_data['rg'].values

                logger.debug(f'\nloaded :: {TIME_SPAN[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN[1]:%H:%M:%S} zarr files')

        except FileNotFoundError:
            N_NOT_AVAILABLE += 1
            logger.info(f"{kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['RADAR']}.zarr  not found! n_Failed = {N_NOT_AVAILABLE}")

        except ValueError as e:
            if 'group not found at path' in str(e):
                logger.info(f"{kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['RADAR']}.zarr  not found! n_Failed = {N_NOT_AVAILABLE}")
            else:
                logger.info(f"{kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['RADAR']}.zarr  some value is missing! n_Failed = {N_NOT_AVAILABLE}")
                logger.info(f"{e}")

            N_NOT_AVAILABLE += 1
            continue

        except Exception as e:
            logger.critical(f"Unexpected error: {sys.exc_info()[0]}\n"
                                f"Check folder: {kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['RADAR']}.zarr, n_Failed = {N_NOT_AVAILABLE}")
            exc_type, exc_value, exc_tb = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_tb)
            logger.critical(f'Exception: Check ~/{kwargs["DATA_PATH"]}/xarray/{dt_str}_{kwargs["RADAR"]}.zarr)')
            logger.critical(f'{e}')
            N_NOT_AVAILABLE += 1
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
            training_mask = load_training_mask(_class, _status, cloudnet_type=kwargs["CLOUDNET"])
            idx_valid_samples = set_intersection(_masked, training_mask)

            if len(idx_valid_samples) < 1: continue

            _feature = _feature[idx_valid_samples, :, :]
            _target = _target[idx_valid_samples, np.newaxis]

            """
            flip the CWT on the y-axis to generate a mirror image, 
            the goal is to overcome the miss-classification of updrafts as liquid
            """
            if kwargs['add_flipped']:
                _feature_flipped = np.zeros(_feature.shape)
                for ismpl, ichan in product(range(len(idx_valid_samples)), range(_feature.shape[-1])):
                    if kwargs["CDIM"] == 'conv2d':
                        _feature_flipped[ismpl, :, :, ichan] = np.fliplr(_feature[ismpl, :, :, ichan])
                else:
                    _feature_flipped[ismpl, :, ichan] = np.flip(_feature[ismpl, :, ichan])

                _feature = np.concatenate((_feature, _feature_flipped), axis=0)
                _target = np.concatenate((_target, _target), axis=0)

        logger.debug(f'\n dim = {_feature.shape}')
        logger.debug(f'\n Number of missing files = {N_NOT_AVAILABLE}')

        feature_set.append(_feature)
        target_labels.append(_target)
        cloudnet_class.append(_class)
        cloudnet_status.append(_status)
        masked_total.append(_masked)
        model_temp.append(_temperature)
        ts_cloudnet.append(_ts)

    return feature_set, target_labels, cloudnet_class, cloudnet_status, masked_total, model_temp, ts_cloudnet, _rg


def one_hot_to_classes(cnn_pred, mask):
    """Converts a one-hot-encodes ANN prediction into Cloudnet-like classes.

    Args:
        cnn_pred (numpy.array): predicted ANN results (num_samples, 9)
        mask (numpy.array, boolean): needs to be provided to skip missing/cloud-free pixels

    Returns:
        predicted_classes (numpy.array): predicted values converted to Cloudnet classes
    """
    predicted_classes = np.zeros(mask.shape, dtype=np.float32)
    predicted_probability = np.zeros(mask.shape, dtype=np.float32)
    cnt = 0
    for iT, iR in product(range(mask.shape[0]), range(mask.shape[1])):
        if mask[iT, iR]: continue
        predicted_classes[iT, iR] = np.argmax(cnn_pred[cnt])
        predicted_probability[iT, iR] = np.max(cnn_pred[cnt])
        #        if predicted_classes[iT, iR] in [1, 5]:
        #            if np.max(cnn_pred[cnt]) < 0.5:
        #                predicted_classes[iT, iR] = 4
        cnt += 1

    return predicted_classes, predicted_probability


