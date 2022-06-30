import datetime
from numba import jit
from tqdm.auto import tqdm
import logging
import re
import subprocess
import sys, os
import traceback
from itertools import groupby
from itertools import product
import torch
from typing import List, Tuple, Dict

import numpy as np

np.random.seed(2000)

import toml
import xarray as xr
from jinja2 import Template
from tqdm.auto import tqdm
from matplotlib.colors import ListedColormap

logger = logging.getLogger('libVoodoo')
logger.setLevel(logging.CRITICAL)

cloudnetpy_classes_n = [
    'Clear sky',
    'Cloud liquid\ndroplets only',
    'Drizzle or rain.',
    'Drizzle/rain &\ncloud droplet',
    'Ice particles.',
    'Ice coexisting with\nsupercooled\nliquid droplets.',
    'Melting ice\nparticles',
    'Melting ice &\ncloud droplets',
    'Aerosol',
    'Insects',
    'Aerosol and\nInsects',
]
cloudnetpy_classes = [
    'Clear sky',
    'Cloud liquid droplets only',
    'Drizzle or rain.',
    'Drizzle/rain & cloud droplet',
    'Ice particles.',
    'Ice & supercooled liquid droplets.',
    'Melting ice particles',
    'Melting ice & cloud droplets',
    'Aerosol',
    'Insects',
    'Aerosol and Insects',
    'No data',
]

cloudnetpy_category_bits = [
    'Small liquid droplets are present.',
    'Falling hydrometeors are present',
    'Wet-bulb temperature is less than 0 degrees C, implying the phase of Bit-1 particles.',
    'Melting ice particles are present.',
    'Aerosol particles are present and visible to the lidar.',
    'Insects are present and visible to the radar.'
]

# Voodoo cloud droplet likelyhood colorbar (viridis + grey below minimum value)
from matplotlib import cm
viridis = cm.get_cmap('viridis', 8)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[:1, :] = np.array([220/256, 220/256, 220/256, 1])
probability_cmap = ListedColormap(newcolors)

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


def get_nan_mask(arr):
    mask = np.full(arr.shape, False)
    mask[arr >= 0.0] = True
    return mask


def get_combined_mask(classes, indices, *status):
    mask = np.full(classes.shape, False)
    for idx in indices:
        mask[classes == idx] = True
    if len(status) > 0: mask[status[0] == 1] = False  # remove lidar echos only
    return mask


def correlation(X, Y, n_smooth=10):
    import pyLARDA.helpers as h

    corr = ma_corr_coef(X, Y)

    X_smoothed = h.smooth(X, n_smooth)  # 10 bins = 5 min
    Y_smoothed = h.smooth(Y, n_smooth)  # 10 bins = 5 min
    corr_smoothed = ma_corr_coef(X_smoothed, Y_smoothed)
    return corr, corr_smoothed


def corr_(x, y):
    return np.ma.corrcoef(np.ma.masked_less_equal(x, 0.0), np.ma.masked_less_equal(y, 0.0))[0, 1]


def get_cloud_base_from_liquid_mask(liq_mask, rg):
    """
    Function returns the time series of cloud base height in meter.
    Args:
        liq_mask:
        rg: range values

    Returns: cloud base height

    """
    _, cbct_mask = find_bases_tops(liq_mask * 1, rg)
    n_ts = liq_mask.shape[0]

    CB = np.full(n_ts, np.nan)

    for ind_time in range(n_ts):
        idx = np.argwhere(cbct_mask[ind_time, :] == -1)
        CB[ind_time] = rg[int(idx[0])] if len(idx) > 0 else 0.0
    return CB


def get_bases_or_tops(dt_list, bases_tops, key='cb'):
    dt_s, rg_s, dt1key, rg1key = [], [], [], []
    for i in range(len(bases_tops[0])):
        for j in range(bases_tops[0][i][f'idx_{key}'].size):
            if bases_tops[0][i]['width'][j] > 150.:
                dt_s.append(dt_list[i])
                rg_s.append(bases_tops[0][i][f'val_{key}'][j] / 1000.)
                if j == 0:
                    dt1key.append(dt_list[i])
                    rg1key.append(bases_tops[0][i][f'val_{key}'][0] / 1000.)
    return {'all': [dt_s, rg_s], 'first': [dt1key, rg1key]}


def find_bases_tops(mask, rg_list):
    """
    This function finds cloud bases and tops for a provided binary cloud mask.
    Args:
        mask (np.array, dtype=bool) : bool array containing False = signal, True=no-signal
        rg_list (list) : list of range values

    Returns:
        cloud_prop (list) : list containing a dict for every time step consisting of cloud bases/top indices, range and width
        cloud_mask (np.array) : integer array, containing +1 for cloud tops, -1 for cloud bases and 0 for fill_value
    """
    cloud_prop = []
    cloud_mask = np.full(mask.shape, 0, dtype=np.int)
    for ind_time in range(mask.shape[0]):  # tqdm(range(mask.shape[0]), ncols=100, unit=' time steps'):
        cloud = [(k, sum(1 for j in g)) for k, g in groupby(mask[ind_time, :])]
        idx_cloud_edges = np.cumsum([prop[1] for prop in cloud])
        bases, tops = idx_cloud_edges[0:][::2][:-1], idx_cloud_edges[1:][::2]
        if tops.size > 0 and tops[-1] == mask.shape[1]:
            tops[-1] = mask.shape[1] - 1
        cloud_mask[ind_time, bases] = -1
        cloud_mask[ind_time, tops] = +1
        cloud_prop.append({'idx_cb': bases, 'val_cb': rg_list[bases],  # cloud bases
                           'idx_ct': tops, 'val_ct': rg_list[tops],  # cloud tops
                           'width': [ct - cb for ct, cb in zip(rg_list[tops], rg_list[bases])]
                           })
    return cloud_prop, cloud_mask


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
    if len(path) > 0: change_dir(path)

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
    if len(path) > 0: change_dir(path)

    import json
    with open(f"{name}", 'r', encoding='utf8') as json_file:
        data = json.load(json_file)
    print(f'Loaded ann configure file :: {name}')
    return data


def change_dir(folder_path, **kwargs):
    """
    This routine changes to a folder or creates it (including subfolders) if it does not exist already.

    Args:
        folder_path (string): path of folder to switch into
    """

    folder_path = folder_path.replace('//', '/', 1)

    if not os.path.exists(os.path.dirname(folder_path)):
        os.makedirs(os.path.dirname(folder_path))
    os.chdir(folder_path)
    logger.debug('\ncd to: {}'.format(folder_path))


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
    import pyLARDA.helpers as h
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
    mask1_flt = np.where(~mask1.astype(np.bool).flatten())
    maskX_flt = intersection(mask_flt[0], mask1_flt[0])
    len_flt = len(maskX_flt)
    idx_list = []
    cnt = 0
    for iter, idx in enumerate(mask_flt[0]):
        if cnt >= len_flt: break
        if idx == maskX_flt[cnt]:
            idx_list.append(iter)
            cnt += 1

    return np.array(idx_list, dtype=int)


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
    from pyLARDA.Transformations import combine
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
    container['dimlabel'] = ['time', 'range'] if len(var.shape) == 2 else ['time']
    container['name'] = kwargs['name']
    container['joints'] = ''
    container['paraminfo'] = ''
    container['filename'] = 'ann_input_files'
    container['rg_unit'] = 'm'
    container['colormap'] = 'cloudnet_jet'
    container['system'] = kwargs['CLOUDNET'] if 'CLOUDNET' in kwargs else 'unknown'
    container['ts'] = ts
    container['rg'] = rg
    container['var_lims'] = kwargs['var_lims'] if 'var_lims' in kwargs else [0., 1000.0]
    container['var_unit'] = kwargs['var_unit'] if 'var_unit' in kwargs else 'K'
    container['mask'] = mask
    container['var'] = var
    return container


def container_to_xarray(container, path):
    xr_ds = xr.DataArray(
        container['var'],
        coords=[container['ts'], container['rg']],
        dims=["time", "height"],
        attrs=container['paraminfo']
    )
    return xr_ds


def post_processor_temperature(data, temperature, **kwargs):
    data_out = np.copy(data)

    if 'melting' in kwargs:
        melting_temp = 0.0  # °C
        # idx_Tplus_ice = (temperature > melting_temp) * (data_out == 4)
        # data_out[idx_Tplus_ice] = 2

        idx_Tplus_mixed = (temperature > melting_temp) * (data_out == 5)
        data_out[idx_Tplus_mixed] = 3  # set to drizzle/rain & cloud droplets

        idx_Tneg0_drizzle = (temperature < melting_temp) * (data_out == 2)
        data_out[idx_Tneg0_drizzle] = 4  # set to ice

        idx_Tneg_melting = (temperature < - 1.0) * (data_out == 6)
        data_out[idx_Tneg_melting] = 4

    if 'hetero_freezing' in kwargs:
        idx_droplets_mixed = ((data_out == 1) + (data_out == 5))
        idx_hetero_freezing = (temperature < -40.0)
        data_out[idx_hetero_freezing * idx_droplets_mixed] = 4  # set to ice

    logger.info('Postprocessing temperature info done.')

    return data_out


def get_good_radar_and_lidar_index(version):
    if 'py' in version.lower():
        return 1
    else:
        return 3


def get_good_lidar_only_index(version):
    if 'py' in version.lower():
        return 3
    else:
        return 1


def post_processor_cloudnet_quality_flag(data, cloudnet_status, cloudnet_class, cloudnet_type=''):
    data_out = np.copy(data)
    GoodRadarLidar = cloudnet_status == get_good_radar_and_lidar_index(cloudnet_type)
    GoodLidarOnly = cloudnet_status == get_good_lidar_only_index(cloudnet_type)

    data_out[GoodRadarLidar] = cloudnet_class[GoodRadarLidar]
    data_out[GoodLidarOnly] = cloudnet_class[GoodLidarOnly]

    if cloudnet_type in ['CLOUDNET', 'CLOUDNET_LIMRAD']:
        KnownAttenuation = cloudnet_status == 6
        data_out[KnownAttenuation] = cloudnet_class[KnownAttenuation]

    logger.info('Postprocessing status flag done.')
    return data_out


def post_processor_cloudnet_classes(data, cloudnet_class):
    data_out = np.copy(data)

    # reclassify all except of ice
    for iclass in [1, 2, 3, 5, 6, 7, 8, 9, 10]:
        indices = cloudnet_class == iclass
        data_out[indices] = cloudnet_class[indices]

    logger.info('Postprocessing cloudnet classes done.')
    return data_out


def post_processor_homogenize(classes, nlabels=9):
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
        for class_ in classes[ind_time:ind_time + WSIZE, ind_range:ind_range + WSIZE].flatten():
            if int(class_) < 9:
                one_hot[int(class_)] = 1
            else:
                one_hot[8] = 1
        return one_hot

    n_dim = WSIZE // 2
    mask = classes == 0
    mask_pad = np.pad(mask, (n_dim, n_dim), 'constant', constant_values=(0, 0))
    classes_out = classes.copy()

    min_percentage = 0.8
    min_bins = WSIZE * WSIZE * int(min_percentage)
    n_ts_pad, n_rg_pad = mask_pad.shape

    logger.info(f'Start Homogenizing')
    for ind_time, ind_range in tqdm(product(range(n_ts_pad - WSIZE), range(n_rg_pad - WSIZE)), total=(n_ts_pad - WSIZE) * (n_rg_pad - WSIZE), unit='pixel'):
        if mask[ind_time, ind_range]:
            continue  # skip clear sky pixel
        #        else:
        #            # If more than 35 of 49 pixels are classified
        #            # as clear, then the central pixel is set to clear
        #            if np.sum(mask_pad[ind_time:ind_time + WSIZE, ind_range:ind_range + WSIZE]) > min_bins:
        #                mask_out[ind_time, ind_range] = True
        #                continue  # skip isolated pixel (rule 7a shupe 2007)

        # Homogenize
        n_samples_total = np.count_nonzero(gen_one_hot(classes[ind_time:ind_time + WSIZE, ind_range:ind_range + WSIZE]), axis=0)

        if n_samples_total == 0: continue

        # If the central pixel is not set to clear and there are
        # more than 7 of 49 pixels with the same type as the central
        # pixel, it is left unchanged. (rule 7b shupe 2007)
        if np.any(n_samples_total > min_bins): continue

        # Otherwise, the central pixel is set
        # to the classification type that is most plentiful in the box.
        # (rule 7c shupe 2007) change to dominant type
        classes_out[ind_time, ind_range] = np.argmax(n_samples_total)

    return classes_out


def postprocessor(xr_ds, smooth=False):
    # POST PROCESSOR ON
    postprocessed = xr_ds.voodoo_classification.copy()

    postprocessed.values = post_processor_temperature(
        postprocessed.values,
        xr_ds.temperature.values - 273.15,
        melting=True
    )
    # postprocessed.values = Utils.post_processor_temperature(
    #    postprocessed.values, xr_ds.temperature, melting=True
    # )
    postprocessed.values = post_processor_cloudnet_quality_flag(
        postprocessed.values, xr_ds.detection_status.values, xr_ds.target_classification.values,
        cloudnet_type='CLOUDNETpy94',
        # cloudnet_type = self.CLOUDNET
    )
    postprocessed.values = post_processor_cloudnet_classes(
        postprocessed.values, xr_ds.target_classification.values
    )
    postprocessed.values = post_processor_homogenize(
        postprocessed.values
    )
    postprocessed.values = post_processor_temperature(
        postprocessed.values, xr_ds.temperature.values - 273.15, hetero_freezing=True
    )
    # update the mask
    mask_proc = xr_ds.mask.copy()
    mask_proc.values[postprocessed.values > 0] = False

    postprocessed.attrs['system'] = 'Voodoo'

    return postprocessed


def sum_liquid_layer_thickness(liquid_pixel_mask, rg_res=30.0):
    """Calculating the liquid layer thickness of the total vertical column"""
    return np.sum(liquid_pixel_mask, axis=1) * rg_res


def get_liquid_pixel_mask(classes):
    return (classes == 1) + (classes == 2) + (classes == 3) + (classes == 5) + (classes == 7)


def ma_corr_coef(X1, X2):
    return np.ma.corrcoef(np.ma.masked_less_equal(X1, 0.0), np.ma.masked_less_equal(X2, 0.0))[0, 1]


def load_training_mask(classes, status, cloudnet_type='CLOUDNETpy'):
    """
    classes
    0: Clear sky\n
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

    status
    0: Clear sky.\n
    1: Good radar and lidar echos.\n
    2: Good radar echo only.\n
    3: Radar echo, corrected for liquid attenuation.
    4: Lidar echo only.\nValue
    5: Radar echo, uncorrected for liquid attenuation.\nValue
    6: Radar ground clutter.\nValue
    7: Lidar clear-air molecular scattering.";
    """
    # create mask
    valid_samples = np.full(status.shape, False)
    valid_samples[status == 1] = True  # add good radar radar & lidar
    valid_samples[classes == 1] = True  # add cloud droplets only class
    # valid_samples[classes == 2] = True  # add drizzle/rain
    valid_samples[classes == 3] = True  # add cloud droplets + drizzle/rain
    valid_samples[classes == 5] = True  # add mixed-phase class pixel
    # valid_samples[classes == 6] = True  # add melting layer
    valid_samples[classes == 7] = True  # add melting layer + SCL class pixel

    # at last, remove lidar only pixel caused by adding cloud droplets only class
    valid_samples[status == 4] = False

    return ~valid_samples


def load_case_file(path):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case']


def load_case_list(path, case_name):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case'][case_name]


def log_number_of_classes(classes, text=''):
    # numer of samples per class afer removing ice
    class_name_list = ['droplets available', 'no droplets availabe']
    class_dist = np.zeros(len(class_name_list), dtype=int)
    logger.info(text)
    logger.info(f'{classes.size:12d}   total')
    for ind, ind_name in enumerate(class_name_list):
        logger.info(f'{class_dist[ind]:12_d}   {ind_name}')
        class_dist[ind] = np.sum(classes == ind + 1)
    return class_dist

def log_number_of_classes2(classes, text=''):
    # numer of samples per class afer removing ice
    class_name_list = ['droplets available', 'no droplets availabe']
    class_dist = np.zeros(len(class_name_list), dtype=int)
    logger.info(text)
    logger.info(f'{classes.shape[0]:12d}   total')
    for ind, ind_name in enumerate(class_name_list):
        logger.info(f'{class_dist[ind]:12_d}   {ind_name}')
        class_dist[ind] = torch.sum(classes == ind)
    return class_dist

def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis

    Args:
        array (np.array): sorted array with values, list will be converted to 1D array
        value: value to find
    Returns:
        index
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i = i + 1
    return i


def performance_metrics(TP, TN, FP, FN):
    sum_stats = {
        'precision': TP / max(TP + FP, 1.0e-7),
        'npv': TN / max(TN + FN, 1.0e-7),
        'recall': TP / max(TP + FN, 1.0e-7),
        'specificity': TN / max(TN + FP, 1.0e-7),
        'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
        'F1-score': TP / max(TP + 0.5 * (FP + FN), 1.0e-7),
    }

    return sum_stats

def performance_metrics2(TP, TN, FP, FN):
    sum_stats = {
        'precision': TP / max(TP + FP, 1.0e-7),
        'far:': FP / max(TP + FP, 1.0e-7),
        'fpr':  FP / max(FP + TN, 1.0e-7),
        'fbi': (TP + FP) / max(TN + FN, 1.0e-7),
        'npv': TN / max(TN + FN, 1.0e-7),
        'recall': TP / max(TP + FN, 1.0e-7),
        'specificity': TN / max(TN + FP, 1.0e-7),
        'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
        'F1-score': TP / max(TP + 0.5 * (FP + FN), 1.0e-7),
        'ets': equitable_thread_score(TP, TN, FP, FN),
    }

    return sum_stats


def remove_cloud_edges(mask, n=2):
    def outline(im):
        ''' Input binary 2D (NxM) image. Ouput array (2xK) of K (y,x) coordinates
            where 0 <= K <= 2*M.
        '''
        topbottom = np.empty((1, 2 * im.shape[1]), dtype=np.uint16)
        topbottom[0, 0:im.shape[1]] = np.argmax(im, axis=0)
        topbottom[0, im.shape[1]:] = (im.shape[0] - 1) - np.argmax(np.flipud(im), axis=0)
        mask = np.tile(np.any(im, axis=0), (2,))
        xvalues = np.tile(np.arange(im.shape[1]), (1, 2))
        return np.vstack([topbottom, xvalues])[:, mask].T

    out_mask = mask.copy()
    for i in range(n):
        cloud_edge = np.full(out_mask.shape, False)
        for boundaries in outline(out_mask):
            cloud_edge[boundaries[0], boundaries[1]] = True
        for boundaries in outline(out_mask.T):
            cloud_edge[boundaries[1], boundaries[0]] = True

        out_mask[cloud_edge] = 0

    return out_mask


def histogram_corr_lwp(array: np.array, n_bins: int=15):
    """
    Args:
        array: two dim. array 0=n_time_steps, 1=n_lwp_bins
    """

    hist_list = []
    len_y = array.shape[1]
    for j in range(len_y):
        tmp = array[:, j]
        hist = np.histogram(tmp[tmp > 0], range=(0.01, 1.01), bins=n_bins)
        hist_list.append(hist[0])

    X = hist[1]
    Y = np.linspace(0.5, len_y - 0.5, len_y)
    Z = np.ma.masked_less_equal(hist_list, 0)
    return X, Y, Z

def interpolate2d(data, mask_thres=0.1, **kwargs):
    """interpolate timeheight data container

    Args:
        mask_thres (float, optional): threshold for the interpolated mask
        **new_time (np.array): new time axis
        **new_range (np.array): new range axis
        **method (str): if not given, use scipy.interpolate.RectBivariateSpline
        valid method arguments:
            'linear' - scipy.interpolate.interp2d
            'nearest' - scipy.interpolate.NearestNDInterpolator
            'rectbivar' (default) - scipy.interpolate.RectBivariateSpline
    """
    import scipy.interpolate as SI

    var = data['var'].copy()
    # var = h.fill_with(data['var'], data['mask'], data['var'][~data['mask']].min())
    # logger.debug('var min {}log_number_of_classes'.format(data['var'][~data['mask']].min()))
    method = kwargs['method'] if 'method' in kwargs else 'rectbivar'
    args_to_pass = {}
    if method == 'rectbivar':
        kx, ky = 1, 1
        interp_var = SI.RectBivariateSpline(data['ts'], data['rg'], var, kx=kx, ky=ky)
        interp_mask = SI.RectBivariateSpline(data['ts'], data['rg'], data['mask'].astype(np.float), kx=kx, ky=ky)
        args_to_pass["grid"] = True
    elif method == 'linear1d':
        points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
        interp_var = SI.LinearNDInterpolator(points, var.flatten(), fill_value=-999.0)
        interp_mask = SI.LinearNDInterpolator(points, (data['mask'].flatten()).astype(np.float))
    elif method == 'linear':
        interp_var = SI.interp2d(data['ts'], data['rg'], np.transpose(var), fill_value=np.nan)
        interp_mask = SI.interp2d(data['ts'], data['rg'], np.transpose(data['mask']).astype(np.float))
    elif method == 'nearest':
        points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
        interp_var = SI.NearestNDInterpolator(points, var.flatten())
        interp_mask = SI.NearestNDInterpolator(points, (data['mask'].flatten()).astype(np.float))
    else:
        raise ValueError('Unknown Interpolation Method', method)

    new_time = data['ts'] if not 'new_time' in kwargs else kwargs['new_time']
    new_range = data['rg'] if not 'new_range' in kwargs else kwargs['new_range']

    if method in ["nearest", "linear1d"]:
        new_points = np.array(list(zip(np.repeat(new_time, len(new_range)), np.tile(new_range, len(new_time)))))
        new_var = interp_var(new_points).reshape((len(new_time), len(new_range)))
        new_mask = interp_mask(new_points).reshape((len(new_time), len(new_range)))
    else:
        new_var = interp_var(new_time, new_range, **args_to_pass)
        new_mask = interp_mask(new_time, new_range, **args_to_pass)

    # print('new_mask', new_mask)
    new_mask[new_mask > mask_thres] = 1
    new_mask[new_mask < mask_thres] = 0
    # print('new_mask', new_mask)

    # print(new_var.shape, new_var)
    # deepcopy to keep data immutable
    interp_data = {**data}

    interp_data['ts'] = new_time
    interp_data['rg'] = new_range
    interp_data['var'] = new_var if method in ['nearest', "linear1d", 'rectbivar'] else np.transpose(new_var)
    interp_data['mask'] = new_mask if method in ['nearest', "linear1d", 'rectbivar'] else np.transpose(new_mask)
    logger.info("interpolated shape: time {} range {} var {} mask {}".format(
        new_time.shape, new_range.shape, new_var.shape, new_mask.shape))

    return interp_data


def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def dh_to_ts(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh*3600))


def get_unixtime(dt64):
    return dt64.astype('datetime64[s]').astype('int')


def dh_to_dt(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    t0 = datetime.datetime.strptime(day_str, '%Y%m%d') - datetime.datetime(1970, 1, 1)
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh*3600))

def ts_to_dt(ts):
    """unix timestamp to dt"""
    return datetime.datetime.utcfromtimestamp(ts)

def decimalhour2unix(dt, time):
    return np.array(
        [x*3600. + dt_to_ts(datetime.datetime(int(dt[:4]), int(dt[4:6]), int(dt[6:]), 0, 0, 0)) for x in time]
    ).astype(int).astype( 'datetime64[s]')


def lin2z(array):
    """linear values to dB (for np.array or single number)"""
    return 10 * np.ma.log10(array)


def z2lin(array):
    """dB to linear values (for np.array or single number)"""
    return 10 ** (array / 10.)


@jit(nopython=True, fastmath=True)
def isKthBitSet(n, k):
    if n & (1 << (k - 1)):
        return True
    else:
        return False

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def evaluation_metrics(pred_labels, true_labels, status=None):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # if no status is given, metric is calculated during optimization
    if status is None:
        for pred, truth in zip(pred_labels, true_labels):
            if truth in [1, 3, 5, 7]:  # TRUE
                if pred == 1:  # is droplet
                    TP += 1
                else:
                    FN += 1
            else:  # FALSE
                if pred == 1:
                    FP += 1
                else:
                    TN += 1


    # if status is given, prediction and true labels in time-range
    else:
        evaluation_mask = ~load_training_mask(true_labels, status)
        # also remove good radar echos only
        evaluation_mask[status == 2] = False

        n_time, n_range = true_labels.shape
        for ind_time in range(n_time):
            for ind_range in range(n_range):
                if evaluation_mask[ind_time, ind_range]:
                    cloundet_label = true_labels[ind_time, ind_range]
                    predicted_label = pred_labels[ind_time, ind_range]
                    if cloundet_label in [1, 3, 5, 7]:  # TRUE
                        if predicted_label == 1:  # is droplet
                            TP += 1
                        else:
                            FN += 1
                    else:  # FALSE
                        if predicted_label == 1:
                            FP += 1
                        else:
                            TN += 1

    from collections import OrderedDict
    out = OrderedDict({
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'precision': TP / max(TP + FP, 1.0e-7),
        'npv': TN / max(TN + FN, 1.0e-7),
        'recall': TP / max(TP + FN, 1.0e-7),
        'specificity': TN / max(TN + FP, 1.0e-7),
        'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
        'F1-score': 2 * TP / max(2 * TP + FP + FN, 1.0e-7),
        'Jaccard-index': TP / max(TP + FN + FP, 1.0e-7),
    })
    out.update({
        'array': np.array([val for val in out.values()], dtype=float)
    })
    return out


def LWP(ds_2D, liq_masks, bases_tops=None):
    from libVoodoo.meteoSI import mod_ad
    def calc_adLWP(ds_2D, liquid_mask, bt=None):
        import scipy.interpolate
        ts, rg = ds_2D['time'].values, ds_2D['height'].values
        f = scipy.interpolate.interp2d(
            ds_2D['model_time'].values,
            ds_2D['model_height'].values,
            ds_2D['temperature'].values.T,
            kind='linear',
            copy=True,
            bounds_error=False,
            fill_value=None
        )
        temperature = f(ts, rg)[:, :].T

        f = scipy.interpolate.interp2d(
            ds_2D['model_time'].values,
            ds_2D['model_height'].values,
            ds_2D['pressure'].values.T,
            kind='linear',
            copy=True,
            bounds_error=False,
            fill_value=None
        )
        pressure = f(ts, rg)[:, :].T

        if bt is not None:
            bt_lists, bt_mask = bt
        else:
            bt_lists, bt_mask = find_bases_tops(liquid_mask, rg)

        adLWP = np.zeros(ts.size)
        for ind_time in range(ts.size):  # tqdm(range(ts.size), ncols=100, unit=' time steps'):
            n_cloud_layers = len(bt_lists[ind_time]['idx_cb'])
            if n_cloud_layers < 1: continue
            Tclouds, Pclouds, RGclouds = [], [], []
            for iL in range(n_cloud_layers):
                tmp_idx = range(bt_lists[ind_time]['idx_cb'][iL], bt_lists[ind_time]['idx_ct'][iL])
                if tmp_idx.stop - tmp_idx.start > 1:  # exclude single range gate clouds
                    Tclouds.append(temperature[ind_time, tmp_idx])
                    Pclouds.append(pressure[ind_time, tmp_idx])
                    RGclouds.append(rg[tmp_idx])
            try:
                Tclouds = np.concatenate(Tclouds)
                Pclouds = np.concatenate(Pclouds)
                RGclouds = np.concatenate(RGclouds)
                adLWP[ind_time] = np.sum(mod_ad(Tclouds, Pclouds, [], RGclouds))
            except:
                continue
        return adLWP

    lwp = calc_adLWP(ds_2D, liq_masks, bt=bases_tops)

    return np.array(lwp)


def correlation_coefficient(x, y):
    return np.corrcoef(x, y)[0, 1]

def ma_correlation_coefficient(x, y):
    x, y = np.ma.masked_invalid(x), np.ma.masked_invalid(y)
    x, y = np.ma.masked_greater_equal(x, 1.0e10), np.ma.masked_greater_equal(y, 1.0e10)
    x, y = np.ma.masked_less_equal(x, 0.0), np.ma.masked_less_equal(y, 0.0)
    return np.ma.corrcoef(x, y)[0, 1]


def timerange(begin, end):
    result = []
    pivot = begin
    while pivot <= end:
        result.append([pivot, pivot + datetime.timedelta(hours=23, minutes=59)])
        pivot += datetime.timedelta(days=1)
    return result


def get_subset(dt_list, mask):
    return [dt for dt, msk in zip(dt_list, mask) if msk]


############

grav = 9.80991  # mean earth gravitational acceleration in m s-2
R = 8.31446261815324  # gas constant in kg m2 s−2 K−1 mol−1
eps = 0.62  # ratio of gas constats for dry air and water vapor dimensionless
cp = 1005.  # specific heat of air at constant pressure in J kg-1 K-1
cv = 1860.  # specific heat of air at constant volume in J kg-1 K-1
gamma_d = 9.76786e-3  # dry-adiabatic lapse rate in K m-1
Rair = 287.058e-3  # specific gas constant of air J kg-1 K-1
m_mol = 0.0289644  # molar mass of air in kg mol-1


@jit(nopython=True, fastmath=True)
def saturated_water_vapor_pressure(T: np.array) -> np.array:
    """ Calculates the saturated water vapor pressure.

    Args:
        T: temperature in [C]
    Returns:
        saturated_water_vapor_pressure in [Pa] = [kg m-1 s-2]
    Source:
        https://en.wikipedia.org/wiki/Vapour_pressure_of_water
    """
    return 0.61078 * np.exp(17.27 * T / (T + 237.3))


@jit(nopython=True, fastmath=True)
def mixing_ratio(e_sat: np.array, p: np.array) -> np.array:
    """ Calculates the ratio of the mass of a variable atmospheric constituent to the mass of dry air.

    Args:
        e_sat: saturated water vapor pressure in [kPa]
        p: pressure in [Pa]
    Returns:
        mixing_ratio dimensionless
    Source:
        https://glossary.ametsoc.org/wiki/Mixing_ratio
    """
    return 0.622 * e_sat / (p - e_sat)


@jit(nopython=True, fastmath=True)
def latent_heat_of_vaporization(T: np.array) -> np.array:
    """ Latent heat (also known as latent energy or heat of transformation) is energy released or absorbed,
        by a body or a thermodynamic system, during a constant-temperature process — usually a first-order
        phase transition.

    Args:
        T: temperature in [C]
    Returns:
        latent_heat_of_vaporization in [J kg-1]
    Source:
        https://en.wikipedia.org/wiki/Latent_heat
    """
    return (2500.8 - 2.36 * T + 1.6e-3 * T * T - 6e-5 * T * T * T) * 1.0e-3


@jit(nopython=True, fastmath=True)
def air_density(T: np.array, p: np.array) -> np.array:
    """ Calculates the density using the ideal gas law.

    Args:
        T: temperature in [C]
        p: pressure in [Pa]
    Returns:
        air_density in [kg m-3]
    Source:
        https://en.wikipedia.org/wiki/Barometric_formula
    """
    return p * m_mol / ((T + 273.15) * Rair) * 1.0e-3


def pseudo_adiabatic_lapse_rate(T: np.array, p: np.array, Lv: np.array) -> np.array:
    """ The rate of decrease of temperature with height of a parcel undergoing a pseudoadiabatic process.

    Args:
        T: temperature in [C]
        p: pressure in [Pa]
    Returns:
        pseudo_adiabatic_lapse_rate in [kg m-1]
    Source:
        https:https://glossary.ametsoc.org/wiki/Pseudoadiabatic_lapse_rate
    """
    e_sat = saturated_water_vapor_pressure(T)
    rv = mixing_ratio(e_sat, p)

    numerator = (1 + rv) * ((1 + Lv * rv) / (R + T))
    denominator = (cp + rv * cv + (Lv * Lv * rv * (eps + rv)) / (R * T * T))
    return grav * numerator / denominator


def adiabatic_liquid_water_content(T: np.array, p: np.array, mask: np.array, delta_h: float = 0.035):
    """ Computes the liquid water content under adiabatic assumtion.

    Args:
        T: temperature in [K]
        p: pressure in [Pa]
        mask: liquid cloud droplet mask
        delta_h: mean range resolution in [km]
    Returns:
        pseudo_adiabatic_lapse_rate in [kg m-1]
    Source:
        https://link.springer.com/article/10.1007/BF01030057
    """
    T_cel = T - 273.15
    LWCad = np.zeros(mask.shape)
    for ind_time in range(mask.shape[0]):
        for ind_range in range(mask.shape[1]):
            if mask[ind_time, ind_range]:
                Lv = latent_heat_of_vaporization(T_cel[ind_time, ind_range])
                gamma_s = pseudo_adiabatic_lapse_rate(T_cel[ind_time, ind_range], p[ind_time, ind_range], Lv)
                rho = air_density(T_cel[ind_time, ind_range], p[ind_time, ind_range])
                # formula (1)
                LWCad[ind_time, ind_range] = rho * cp / Lv * (gamma_d - gamma_s) * delta_h

    # correction term formula (2)
    LWCad = LWCad * (1.239 - 0.145 * np.log(delta_h))

    return LWCad


def equitable_thread_score(a, b, c, d):
    n = a+b+c+d
    E = (a+b)*(a+c)/max(1.0e-7, n)
    return (a - E) /max(1.0e-7, (a - E + b + c))

def find_stratiform_liquid_topped(cth, lcbh, depth=500):
    """
        Finds time steps with, where stratiform cloud with liquid top is observed.
    Args:
        cth: cloud top heighte
        lcbh: liquid cloud base height
        depth:

    Returns: list of boolean wherer True = straiform cloud with liquid top, False Otherwise

    """
    return [True if (cth[i] - depth < lcbh[i] < cth[i]) else False for i in range(len(cth))]

def compute_metrics(cc, cs, liq_mask, lwp_masks):
    # compute the errormatrix for binned lwp values
    beta = 0.5
    _eps = 1.0e-7
    arr = np.zeros((len(lwp_masks) + 1, 11))
    _valid_cloudnet = (cs == 1)
    _valid_cloudnet[((cc == 1) + (cc == 3) + (cc == 5) + (cc == 7))] = True
    _valid_cloudnet[cs == 4] = False
    #_valid_cloudnet[:, :38] = False #do not validate first chirp

    for i in range(1, len(lwp_masks) + 1):

        _valid_mask = np.zeros(liq_mask.shape)
        _lwp_bin_mask = np.array([lwp_masks[i - 1]] * cc.shape[1]).T
        _valid_mask[liq_mask] = 1

        _valid_mask = _valid_mask[_lwp_bin_mask * _valid_cloudnet]
        _ref_mask = cc[_lwp_bin_mask * _valid_cloudnet]

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, truth in zip(_valid_mask, _ref_mask):
            if truth in [1, 3, 5, 7]:  # TRUE
                if pred == 1:  # is droplet
                    TP += 1
                else:
                    FN += 1
            else:  # FALSE
                if pred == 1:
                    FP += 1
                else:
                    TN += 1

        arr[i, :4] = TP, TN, FP, FN
        arr[i, 4] = TP / max(TP + FP, _eps)  # precision
        arr[i, 5] = TN / max(TN + FN, _eps)  # npv
        arr[i, 6] = TP / max(TP + FN, _eps)  # recall
        arr[i, 7] = TN / max(TN + FP, _eps)  # specificity
        arr[i, 8] = (TP + TN) / max(TP + TN + FP + FN, _eps)  # accuracy
        arr[i, 9] = 2 * TP / max(2 * TP + FP + FN, _eps)  # F1-score
        arr[i, 10] = (1 + beta*beta)*arr[i, 4]*arr[i, 6]/(arr[i, 6] + beta*beta*arr[i, 4])

    arr[0, :4] = np.sum(arr[1:, :4], axis=0)
    arr[0, 4:] = np.mean(arr[1:, 4:], axis=0)
    return arr



def convection_index_fast(mdv: np.array, dts: int = 2, drg: int = 1):
    """ Computes the convective index, see formula (6) in Kneifel et al. 2020.

     Args:
         mdv: mean Doppler velocity
         dts: number of time steps, default=minimum=2
         drg: number range bins, default=minimum=1
     Source:
         https://journals.ametsoc.org/view/journals/atsc/77/10/jasD200007.xml

    """
    assert drg*dts > 1, f'increase number of time steps (dts={dts}) or range bins (drg={drg}) used to compute the mean'
    from scipy.signal import convolve2d
    vel = np.ma.fix_invalid(-mdv, copy=True)
    vel = np.ma.masked_greater_equal(vel, 99)
    vel = np.ma.masked_invalid(vel)

    rect = np.ones((dts, drg))/(dts*drg)
    mean_vel = convolve2d(vel, rect, mode='same', boundary='fill', fillvalue=-999)

    return np.abs(vel-mean_vel)/mean_vel


def convection_index(mdv: xr.DataArray, dts: int = 2, drg: int = 1):
    """ Computes the convective index, see formula (6) in Kneifel et al. 2020.

     Args:
         mdv: mean Doppler velocity
         dts: number of time steps, default=minimum=2
         drg: number range bins, default=minimum=1
     Source:
         https://journals.ametsoc.org/view/journals/atsc/77/10/jasD200007.xml

    """
    assert drg*dts > 1, f'increase number of time steps (dts={dts}) or range bins (drg={drg}) used to compute the mean'

    ts, rg = mdv['time'].values, mdv['height'].values
    vel = np.ma.fix_invalid(-mdv.values, copy=True)
    vel = np.ma.masked_greater_equal(vel, 99)
    vel = np.ma.masked_invalid(vel)
    kappa = np.zeros(vel.shape)

    for it in tqdm(range(0, len(ts), dts), unit=' time steps'):
        for ir in range(0, len(rg), drg):
            orig = vel[it:it + dts, ir:ir + dts]
            mean_val = np.ma.mean(orig)
            kappa[it:it + dts, ir:ir + dts] = np.ma.abs(orig - mean_val) / mean_val

    return kappa



import matplotlib.pyplot as plt

def ql_features(X_train, y_train):

    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(14, 10))
    cnt = 1
    for row_ax in ax:
        for iax in row_ax:
            idx_iclass = np.argwhere(y_train == cnt)[:, 0]
            mean_feature = np.mean(X_train[idx_iclass, :, :], axis=0)
            pmesh = iax.pcolormesh(mean_feature.T, cmap='jet')
            cbar = plt.colorbar(pmesh, ax=iax)
            iax.set(
                xlabel='', xticklabels='',
                title=cloudnetpy_classes[cnt]+f'  n={idx_iclass.size}'
            )
            cnt += 1
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    return fig, ax


def ql_features_1D(list_fn, list_counts):
    fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(14, 10))
    x_axis = np.linspace(start=-127.5, stop=127.5, num=list_fn.shape[1])
    x_ticks = np.linspace(start=-128, stop=128, num=5)
    for fn in range(list_counts.shape[0]):
        cnt = 1
        for i, row_ax in enumerate(ax):
            for j, iax in enumerate(row_ax):
                mean_feature = list_fn[fn, :, cnt]
                l, = iax.plot(x_axis, mean_feature, alpha=0.7, label=f'fn{fn}')
                if fn == 0:
                    iax.grid()
                    iax.set(
                        xlabel='', xticklabels='', xlim=[-150, 150], xticks=x_ticks,
                        title=cloudnetpy_classes[cnt] + f'  n={np.sum(list_counts[:, cnt]):_.0f}'
                    )
                cnt += 1
    ax[4, 0].legend(
        bbox_to_anchor=(0., -0.5, 1., .102),
        loc=2, ncol=4)
    ax[4, 0].set(xlabel='Dopple bins', xticklabels=x_ticks)
    ax[4, 1].set(xlabel='Dopple bins', xticklabels=x_ticks)
    return fig, ax


def interpolate_to_256(rpg_data, rpg_header, polarization='TotSpec'):
    from scipy.interpolate import interp1d

    rng_offsets = rpg_header['RngOffs']
    nts, nrg, nvel = rpg_data['TotSpec'].shape

    spec_new = np.zeros((nts, nrg, 256))
    for ichirp in range(len(rng_offsets) - 1):

        ia = rng_offsets[ichirp]
        ib = rng_offsets[ichirp + 1]
        nvel = rpg_header['SpecN'][ichirp]
        spec = rpg_data[polarization][:, ia:ib, :]

        if nvel == 256:
            spec_new[:, ia:ib, :] = spec
        else:
            old = rpg_header['velocity_vectors'][ichirp]
            f = interp1d(old, spec, axis=2, bounds_error=False, fill_value=-999., kind='linear')
            spec_new[:, ia:ib, :] = f(np.linspace(old[np.argmin(old)], old[np.argmax(old)], 256))

    return spec_new


def log_number_of_cla():
    return None