import datetime
import logging
import re
import subprocess
import sys
import traceback
from itertools import groupby
from itertools import product

import numpy as np
import toml
import xarray as xr
from jinja2 import Template
from tqdm.auto import tqdm

logger = logging.getLogger('libVoodoo')
logger.setLevel(logging.CRITICAL)

cloudnetpy_classes = [
    'Clear sky',
    'Cloud liquid droplets only',
    'Drizzle or rain.',
    'Drizzle/rain & cloud droplet',
    'Ice particles.',
    'Ice coexisting with supercooled liquid droplets.',
    'Melting ice particles',
    'Melting ice & cloud droplets',
    'Aerosol',
    'Insects',
    'Aerosol and Insects',
]

cloudnetpy_category_bits = [
    'Small liquid droplets are present.',
    'Falling hydrometeors are present',
    'Wet-bulb temperature is less than 0 degrees C, implying the phase of Bit-1 particles.',
    'Melting ice particles are present.',
    'Aerosol particles are present and visible to the lidar.',
    'Insects are present and visible to the radar.'
]


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

    for iT in range(n_ts):
        idx = np.argwhere(cbct_mask[iT, :] == -1)
        CB[iT] = rg[int(idx[0])] if len(idx) > 0 else 0.0
    return CB


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
    for iT in range(mask.shape[0]):
        cloud = [(k, sum(1 for j in g)) for k, g in groupby(mask[iT, :])]
        idx_cloud_edges = np.cumsum([prop[1] for prop in cloud])
        bases, tops = idx_cloud_edges[0:][::2][:-1], idx_cloud_edges[1:][::2]
        if tops.size > 0 and tops[-1] == mask.shape[1]:
            tops[-1] = mask.shape[1] - 1
        cloud_mask[iT, bases] = -1
        cloud_mask[iT, tops] = +1
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
    import pyLARDA.helpers as h
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
    import pyLARDA.helpers as h
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
        for class_ in classes[iT:iT + WSIZE, iR:iR + WSIZE].flatten():
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
    valid_samples[classes == 5] = True  # add mixed-phase class pixel
    valid_samples[classes == 6] = True  # add melting layer class pixel
    valid_samples[classes == 7] = True  # add melting layer + SCL class pixel
    valid_samples[classes == 1] = True  # add cloud droplets only class

    # at last, remove lidar only pixel caused by adding cloud droplets only class
    valid_samples[status == idx_good_lidar_only] = False

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
    class_n_distribution = np.zeros(len(cloudnetpy_classes))
    logger.critical(text)
    logger.critical(f'{classes.size:12d}   total')
    for i in range(len(cloudnetpy_classes)):
        n = np.sum(classes == i)
        logger.critical(f'{n:12d}   {cloudnetpy_classes[i]}')
        class_n_distribution[i] = n
    return class_n_distribution


def target_class2bit_mask(target_labels):
    #
    #     clutter = categorize_bits.quality_bits['clutter']
    #     classification = np.zeros(bits['cold'].shape, dtype=int)
    #     classification[bits['droplet'] & ~bits['falling']] = 1
    #     classification[~bits['droplet'] & bits['falling']] = 2
    #     classification[bits['droplet'] & bits['falling']] = 3
    #     classification[~bits['droplet'] & bits['falling'] & bits['cold']] = 4
    #     classification[bits['droplet'] & bits['falling'] & bits['cold']] = 5
    #     classification[bits['melting']] = 6
    #     classification[bits['melting'] & bits['droplet']] = 7
    #     classification[bits['aerosol']] = 8
    #     classification[bits['insect'] & ~clutter] = 9
    #     classification[bits['aerosol'] & bits['insect'] & ~clutter] = 10
    #     classification[clutter & ~bits['aerosol']] = 0

    #     bit_mask: droplet(0) / falling(1) / cold(2) / melting(3) / insect(4)

    bit_mask = np.zeros((target_labels.size, 5))
    # cloud droplets only
    droplets = (target_labels == 1) + (target_labels == 3) + (target_labels == 5) + (target_labels == 7)
    falling = (target_labels == 2) + (target_labels == 3) + (target_labels == 4) + (target_labels == 5)
    cold = (target_labels == 4) + (target_labels == 5)
    melting = (target_labels == 6) + (target_labels == 7)
    insects = (target_labels == 9) + (target_labels == 10)
    bit_mask[droplets, 0] = 1
    bit_mask[falling, 1] = 1
    bit_mask[cold, 2] = 1
    bit_mask[melting, 3] = 1
    bit_mask[insects, 4] = 1

    return bit_mask


def load_dataset_from_zarr(case_string_list, case_list_path, **kwargs):
    N_NOT_AVAILABLE = 0
    features, labels, multilabels, mask = [], [], [], []
    class_, status, catbits, qualbits, insect_prob = [], [], [], [], []
    mT, mP, mq, ts, rg, lwp, u_w, v_w = [], [], [], [], [], [], [], []
    Z, VEL, VEL_sigma, width, beta, attbsc532, depol = [], [], [], [], [], [], []

    xarr_ds = []

    for icase, case_str in tqdm(enumerate(case_string_list), total=len(case_string_list), unit='files'):

        # gather time interval, etc..:505

        case = load_case_list(case_list_path, case_str)
        TIME_SPAN = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
        dt_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'

        # check if a mat files is available
        try:
            with xr.open_zarr(f'{kwargs["DATA_PATH"]}/xarray/{dt_str}_{kwargs["RADAR"]}.zarr') as zarr_data:
                # for training & validation
                _multitarget = zarr_data['multitargets'].values
                _feature = zarr_data['features'].values
                _target = zarr_data['targets'].values
                _masked = zarr_data['masked'].values

                # for validation
                _class = zarr_data['CLASS'].values if 'CLASS' in zarr_data else []
                _status = zarr_data['detection_status'].values if 'detection_status' in zarr_data else []
                _catbits = zarr_data['category_bits'].values if 'category_bits' in zarr_data else []
                _qualbits = zarr_data['quality_bits'].values if 'quality_bits' in zarr_data else []
                _insect_prob = zarr_data['insect_prob'].values if 'insect_prob' in zarr_data else []
                _temperature = zarr_data['T'].values if 'T' in zarr_data else []
                _pressure = zarr_data['P'].values if 'P' in zarr_data else []
                _q = zarr_data['q'].values if 'q' in zarr_data else []
                _ts = zarr_data['ts'].values if 'ts' in zarr_data else []
                _rg = zarr_data['rg'].values if 'rg' in zarr_data else []
                _lwp = zarr_data['LWP'].values if 'LWP' in zarr_data else []
                _uw = zarr_data['UWIND'].values if 'UWIND' in zarr_data else []
                _vw = zarr_data['VWIND'].values if 'VWIND' in zarr_data else []
                _Z = zarr_data['Z'].values if 'Z' in zarr_data else []
                _VEL = zarr_data['VEL'].values if 'VEL' in zarr_data else []
                _VEL_sigma = zarr_data['VEL_sigma'].values if 'VEL_sigma' in zarr_data else []
                _width = zarr_data['width'].values if 'width' in zarr_data else []
                _beta = zarr_data['beta'].values if 'beta' in zarr_data else []
                _attbsc532 = zarr_data['attbsc532'].values if 'attbsc532' in zarr_data else []
                _depol = zarr_data['depol'].values if 'depol' in zarr_data else []

                logger.debug(f'\nloaded :: {TIME_SPAN[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN[1]:%H:%M:%S} zarr files')

        except KeyError:
            N_NOT_AVAILABLE += 1
            logger.info(f"{kwargs['DATA_PATH']}/xarray/{dt_str}_{kwargs['RADAR']}.zarr variable 'multitargets' not found! n_Failed = {N_NOT_AVAILABLE}")
            continue

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
            _multitarget = _multitarget[idx_valid_samples, :]

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
                _multitarget = np.concatenate((_multitarget, _multitarget), axis=0)

        logger.debug(f'\n dim = {_feature.shape}')
        logger.debug(f'\n Number of missing files = {N_NOT_AVAILABLE}')

        features.append(_feature)
        labels.append(_target)
        multilabels.append(_multitarget)
        xarr_ds.append(zarr_data)

        if kwargs["TASK"] == 'train':
            continue

        class_.append(_class)
        status.append(_status)
        catbits.append(_catbits)
        qualbits.append(_qualbits)
        insect_prob.append(_insect_prob)
        mask.append(_masked)
        mT.append(_temperature)
        mP.append(_pressure)
        mq.append(_q)
        ts.append(_ts)
        lwp.append(_lwp)
        u_w.append(_uw)
        v_w.append(_vw)
        Z.append(_Z)
        VEL.append(_VEL)
        VEL_sigma.append(_VEL_sigma)
        width.append(_width)
        beta.append(_beta)
        attbsc532.append(_attbsc532)
        depol.append(_depol)

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    multilabels = np.concatenate(multilabels, axis=0)
    returns = [features, np.squeeze(labels), multilabels]

    if kwargs["TASK"] != 'train':
        returns.append(np.concatenate(class_, axis=0))
        returns.append(np.concatenate(status, axis=0))
        returns.append(np.concatenate(catbits, axis=0))
        returns.append(np.concatenate(qualbits, axis=0))
        returns.append(np.concatenate(insect_prob, axis=0))
        returns.append(np.concatenate(mask, axis=0))
        returns.append(np.concatenate(mT, axis=0))
        returns.append(np.concatenate(mP, axis=0))
        returns.append(np.concatenate(mq, axis=0))
        returns.append(np.concatenate(ts, axis=0))
        returns.append(np.array(_rg))
        returns.append(np.concatenate(lwp, axis=0))
        returns.append(np.concatenate(u_w, axis=0))
        returns.append(np.concatenate(v_w, axis=0))
        returns.append(np.concatenate(Z, axis=0))
        returns.append(np.concatenate(VEL, axis=0))
        returns.append(np.concatenate(VEL_sigma, axis=0))
        returns.append(np.concatenate(width, axis=0))
        returns.append(np.concatenate(beta, axis=0))
        returns.append(np.concatenate(attbsc532, axis=0))
        returns.append(np.concatenate(depol, axis=0))
    else:
        for i in range(22):  returns.append(None)

    return returns


def one_hot_to_classes(cnn_pred, mask):
    """Converts a one-hot-encodes ANN prediction into Cloudnet-like classes.

    Args:
        cnn_pred (numpy.array): predicted ANN results (num_samples, 9)
        mask (numpy.array, boolean): needs to be provided to skip missing/cloud-free pixels

    Returns:
        predicted_classes (numpy.array): predicted values converted to Cloudnet classes
    """
    predicted_classes = np.zeros(mask.shape, dtype=np.float32)
    predicted_probability = np.zeros(mask.shape + (9,), dtype=np.float32)
    cnt = 0
    for iT, iR in product(range(mask.shape[0]), range(mask.shape[1])):
        if mask[iT, iR]: continue
        predicted_classes[iT, iR] = np.argmax(cnn_pred[cnt])
        predicted_probability[iT, iR, :] = cnn_pred[cnt]
        cnt += 1

    return predicted_classes, predicted_probability


def classes_to_one_hot(classes, mask):
    """Converts a one-hot-encodes ANN prediction into Cloudnet-like classes.

    Args:
        cnn_pred (numpy.array): predicted ANN results (num_samples, 9)
        mask (numpy.array, boolean): needs to be provided to skip missing/cloud-free pixels

    Returns:
        predicted_classes (numpy.array): predicted values converted to Cloudnet classes
    """
    one_hot = []

    for iT, iR in product(range(classes.shape[0]), range(classes.shape[1])):
        if mask[iT, iR]: continue
        one_hot.append(classes[iT, iR])

    return np.array(one_hot)


def one_hot_to_spectra(features, mask):
    """Converts a one-hot-encodes ANN prediction into Cloudnet-like classes.

    Args:
        cnn_pred (numpy.array): predicted ANN results (num_samples, 9)
        mask (numpy.array, boolean): needs to be provided to skip missing/cloud-free pixels

    Returns:
        predicted_classes (numpy.array): predicted values converted to Cloudnet classes
    """
    from itertools import product
    print((mask.shape,) + (features.shape[2],))
    spectra = np.zeros(mask.shape + (features.shape[1], features.shape[2],), dtype=np.float32)
    cnt = 0
    for iT, iR in product(range(mask.shape[0]), range(mask.shape[1])):
        if mask[iT, iR]: continue
        spectra[iT, iR, :, :] = features[cnt]
        cnt += 1

    return spectra


def random_choice(xr_ds, rg_int, N=4, iclass=4, var='voodoo_classification'):
    import pyLARDA.helpers as h
    nts, nrg = xr_ds.ZSpec.ts.size, xr_ds.ZSpec.rg.size

    icnt = 0
    indices = np.zeros((N, 2), dtype=np.int)
    nnearest = h.argnearest(xr_ds.ZSpec.rg.values, rg_int)

    while icnt < N:
        while True:
            idxts = int(np.random.randint(0, high=nts, size=1))
            idxrg = int(np.random.randint(0, high=nnearest, size=1))
            if ~xr_ds.mask[idxts, idxrg] and xr_ds[var].values[idxts, idxrg] == iclass:
                indices[icnt, :] = [idxts, idxrg]
                icnt += 1
                break
    return indices

