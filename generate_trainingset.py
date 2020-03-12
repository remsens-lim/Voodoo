#!/home/sdig/anaconda3/bin/python3
"""
Short description:
    Creating a .mat file containing input features and labels for the VOOODOO neural network.
"""

from datetime import timedelta, datetime
import sys

sys.path.append('../larda/')
sys.path.append('.')

import logging
import numpy as np
from time import time
import toml

from scipy.io import savemat
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator, interp2d, NearestNDInterpolator
from tqdm.auto import tqdm

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpectraProcessing as sp
from larda.pyLARDA.Transformations import plot_timeheight, interpolate2d

import voodoo.libVoodoo.Loader_v2 as Loader
import voodoo.libVoodoo.Plot   as Plot

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project - Feature Extractor"
__credits__ = ["Willi Schimmel"]
__license__ = "MIT"
__version__ = "1.1.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def load_data(larda, system, time_span, var_list):
    data = {}
    for i, var in enumerate(var_list):
        var_info = larda.read(system, var, time_span, [0, 'max'])
        var_info['n_ts'] = var_info['ts'].size
        var_info['n_rg'] = var_info['rg'].size
        data.update({var: var_info})
    return data


def average_time_dim(ts, rg, var, mask, **kwargs):
    assert 'new_time' in kwargs, ValueError('new_time key needs to be provided')

    new_ts = kwargs['new_time']
    n_ts_new = len(new_ts)
    n_ts = len(ts)
    n_rg = len(rg)

    # if 2D array is provided
    if len(var.shape) == 2:
        var = var.reshape((n_ts, n_rg, 1))

    n_vel = var.shape[2]

    ip_var = np.zeros((n_ts_new, n_rg, n_vel), dtype=np.float32)
    ip_mask = np.full((n_ts_new, n_rg, n_vel), True)

    print('Averaging over N radar spectra time-steps (30 sec avg)...')
    # for iBin in range(n_vel):
    for iBin in tqdm(range(n_vel)):

        for iT_cn in range(n_ts_new - 1):
            iT_rd0 = h.argnearest(ts, new_ts[iT_cn])
            ts_diff = int((new_ts[iT_cn + 1] - new_ts[iT_cn]) / (ts[iT_rd0 + 1] - ts[iT_rd0]))

            iT_rdN = iT_rd0 + ts_diff if iT_rd0 + ts_diff < n_ts else n_ts  # catch edge case
            ip_var[iT_cn, :, iBin] = np.mean(var[iT_rd0:iT_rdN, :, iBin], axis=0)
            rg_with_signal = np.sum(mask[iT_rd0:iT_rdN, :, iBin], axis=0) < ts_diff
            ip_mask[iT_cn, rg_with_signal, iBin] = False

        ip_var[-1, :, iBin] = ip_var[-2, :, iBin]
        ip_mask[-1, :, iBin] = ip_mask[-2, :, iBin]

    if ip_var.shape[2] == 1:
        ip_var = ip_var.reshape((n_ts_new, n_rg))
        ip_mask = ip_mask.reshape((n_ts_new, n_rg))

    return ip_var, ip_mask


def multichannelspectra(ts, var, mask, **kwargs):
    assert 'new_time' in kwargs, ValueError('new_time key needs to be provided')

    new_ts = kwargs['new_time']
    n_channels = kwargs['n_channels'] if 'n_channels' in kwargs else 4
    n_ts_new = len(new_ts) if len(new_ts) > 0  else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = var.shape
    mid = n_channels//2

    ip_var = np.zeros((n_ts_new, n_rg, n_vel, n_channels), dtype=np.float32)
    ip_mask = np.empty((n_ts_new, n_rg, n_vel, n_channels), dtype=np.bool)

    print(f'\nConcatinate {n_channels} spectra to 1 sample:\n'
          f'    --> resulting tensor dimension (n_samples, n_velocity_bins, n_cwt_scales, n_channels) = (????, 256, 32, {n_channels}) ......')
    # for iBin in range(n_vel):
    for iBin in tqdm(range(n_vel)):
        for iT_cn in range(n_ts_new):
            iT_rd0 = h.argnearest(ts, new_ts[iT_cn])
            for itmp in range(-mid, mid):
                iTdiff = itmp if iT_rd0 + itmp < n_ts else 0
                ip_var[iT_cn,  :, iBin, iTdiff + mid] = var[iT_rd0 + iTdiff, :, iBin]
                ip_mask[iT_cn, :, iBin, iTdiff + mid] = mask[iT_rd0 + iTdiff, :, iBin]

    return ip_var, ip_mask


def interpolate3d(data, mask_thres=0.1, **kwargs):
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
    all_var_bins = h.fill_with(data['var'], data['mask'], data['var'][~data['mask']].min())
    all_mask_bins = data['mask']
    var_interp = np.full((kwargs['new_time'].size, kwargs['new_range'].size, data['vel'].size), 0.0)
    mask_interp = np.full((kwargs['new_time'].size, kwargs['new_range'].size, data['vel'].size), 1)
    method = kwargs['method'] if 'method' in kwargs else 'rectbivar'

    new_time = data['ts'] if not 'new_time' in kwargs else kwargs['new_time']
    new_range = data['rg'] if not 'new_range' in kwargs else kwargs['new_range']

    if method in ['nearest', 'linear1d']:
        points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
        new_points = np.array(list(zip(np.repeat(new_time, len(new_range)), np.tile(new_range, len(new_time)))))

    args_to_pass = {}

    print('Start interpolation......')
    for iBin in tqdm(range(data['vel'].size)):
        var, mask = all_var_bins[:, :, iBin], all_mask_bins[:, :, iBin]
        if method == 'rectbivar':
            kx, ky = 3, 3
            interp_var = RectBivariateSpline(data['ts'], data['rg'], var, kx=kx, ky=ky)
            interp_mask = RectBivariateSpline(data['ts'], data['rg'], mask.astype(np.float), kx=kx, ky=ky)
            args_to_pass = {"grid": True}
        elif method == 'linear1d':
            interp_var = LinearNDInterpolator(points, var.flatten(), fill_value=-999.0)
            interp_mask = LinearNDInterpolator(points, (mask.flatten()).astype(np.float))
        elif method in ['linear', 'cubic', 'quintic']:
            interp_var = interp2d(data['rg'], data['ts'], var, fill_value=-999.0, kind=method)
            interp_mask = interp2d(data['rg'], data['ts'], mask.astype(np.float))
            args_to_pass = {}
        elif method == 'nearest':
            interp_var = NearestNDInterpolator(points, var.flatten())
            interp_mask = NearestNDInterpolator(points, (mask.flatten()).astype(np.float))

        else:
            raise ValueError('Unknown Interpolation Method', method)

        if method in ['nearest', 'linear1d']:
            new_var = interp_var(new_points).reshape((len(new_time), len(new_range)))
            new_mask = interp_mask(new_points).reshape((len(new_time), len(new_range)))
        else:
            new_var = interp_var(new_time, new_range, **args_to_pass)
            new_mask = interp_mask(new_time, new_range, **args_to_pass)

        new_mask[new_mask > mask_thres] = 1
        new_mask[new_mask < mask_thres] = 0

        if var_interp.shape[:2] == new_var.shape:
            var_interp[:, :, iBin] = new_var
            mask_interp[:, :, iBin] = new_mask
        else:
            var_interp[:, :, iBin] = new_var.T
            mask_interp[:, :, iBin] = new_mask.T

    # deepcopy to keep data immutable
    interp_data = {**data}

    interp_data['ts'] = new_time
    interp_data['rg'] = new_range
    interp_data['var'] = var_interp
    interp_data['mask'] = mask_interp
    print("interpolated shape: time {} range {} var {} mask {}".format(
        new_time.shape, new_range.shape, var_interp.shape, mask_interp.shape))

    return interp_data


def load_features_from_nc(
        case_string,
        voodoo_path='',
        data_path='',
        case_list_path='',
        kind='3spectra',
        system='limra94',
        interp='rectbivar',
        save=True,
        task='predict',
        **kwargs
):
    def quick_check(dummy_container, name_str):
        if spec_settings['quick_check']:
            if len(ZSpec['VHSpec']['mask'].shape) == 4:
                ZE =  h.put_in_container(np.sum(np.mean(ZSpec['VHSpec']['var'], axis=3), axis=2), dummy_container)
                ZE['mask'] = np.all(np.all(ZSpec['VHSpec']['mask'], axis=3), axis=2)
            elif len(ZSpec['VHSpec']['mask'].shape) == 3:
                ZE =  h.put_in_container(np.sum(ZSpec['VHSpec']['var'], axis=2), dummy_container)
                ZE['mask'] = np.all(ZSpec['VHSpec']['mask'], axis=2)
            else:
                ZE =  h.put_in_container(ZSpec['VHSpec']['var'], dummy_container)
                ZE['mask'] = ZSpec['VHSpec']['mask']

            ZE['dimlabel'] = ['time', 'range']
            ZE['name'] = name_str
            ZE['joints'] = ''
            ZE['rg_unit'] = ''
            ZE['colormap'] = 'jet'
            ZE['system'] = 'LIMRAD94'
            ZE['var_lims'] = [-60, 20]
            ZE['var_unit'] = 'dBZ'

            fig, _ = plot_timeheight(ZE, var_converter='lin2z', title=f'pseudo Ze ({name_str}) for testing interpolations')  # , **plot_settings)
            Plot.save_figure(fig, name=f'limrad_{name_str}_{dt_string}.png', dpi=200)

    spec_settings = toml.load(voodoo_path + 'ann_model_setting.toml')['feature']['info']

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu')

    # load case information
    case = Loader.load_case_list(case_list_path, case_string)

    TIME_SPAN_ = [datetime.strptime(case['time_interval'][0], '%Y%m%d-%H%M'),
                  datetime.strptime(case['time_interval'][1], '%Y%m%d-%H%M')]
    TIME_SPAN_2 = [datetime.strptime(case['time_interval'][0], '%Y%m%d-%H%M') - timedelta(seconds=100.0),
                   datetime.strptime(case['time_interval'][1], '%Y%m%d-%H%M') + timedelta(seconds=100.0)]
    begin_dt, end_dt = TIME_SPAN_
    dt_string = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}'

    ########################################################################################################################################################
    #
    #   _    ____ ____ ___    / ____ ____ _  _ ____    ____ _    ____ _  _ ___  _  _ ____ ___    ___  ____ ___ ____
    #   |    |  | |__| |  \  /  [__  |__| |  | |___    |    |    |  | |  | |  \ |\ | |___  |     |  \ |__|  |  |__|
    #   |___ |__| |  | |__/ /   ___] |  |  \/  |___    |___ |___ |__| |__| |__/ | \| |___  |     |__/ |  |  |  |  |
    #                      /
    #
    cnpy94_model = load_data(larda, 'CLOUDNETpy94', TIME_SPAN_, ['T', 'P'])

    if save:
        h.change_dir(f'{data_path}/cloudnet/')
        savemat(f'{dt_string}_cloudnetpy94_model_T.mat', cnpy94_model.pop('T'), do_compression=True)
        savemat(f'{dt_string}_cloudnetpy94_model_P.mat', cnpy94_model.pop('P'), do_compression=True)
        print(f'save :: {dt_string}_cloudnetpy94_model_T/P')

    """
    " STATUS

    \nValue 0: Clear sky.
    \nValue 1: Good radar and lidar echos.
    \nValue 2: Good radar echo only.
    \nValue 3: Radar echo, corrected for liquid attenuation.
    \nValue 4: Lidar echo only.
    \nValue 5: Radar echo, uncorrected for liquid attenuation.
    \nValue 6: Radar ground clutter.
    \nValue 7: Lidar clear-air molecular scattering.";

    " CLASSES

    \nValue 0: Clear sky.
    \nValue 1: Cloud liquid droplets only.
    \nValue 2: Drizzle or rain.
    \nValue 3: Drizzle or rain coexisting with cloud liquid droplets.
    \nValue 4: Ice particles.
    \nValue 5: Ice coexisting with supercooled liquid droplets.
    \nValue 6: Melting ice particles.
    \nValue 7: Melting ice particles coexisting with cloud liquid droplets.
    \nValue 8: Aerosol particles, no cloud or precipitation.
    \nValue 9: Insects, no cloud or precipitation.
    \nValue 10: Aerosol coexisting with insects, no cloud or precipitation.";

    """
    cnpy94_class = load_data(larda, 'CLOUDNETpy94', TIME_SPAN_, ['CLASS', 'detection_status'])
    ts_cnpy94, rg_cnpy94 = cnpy94_class['CLASS']['ts'], cnpy94_class['CLASS']['rg']

    if save:
        savemat(f'{dt_string}_cloudnetpy94_class.mat', cnpy94_class['CLASS'], do_compression=True)
        savemat(f'{dt_string}_cloudnetpy94_status.mat', cnpy94_class['detection_status'], do_compression=True)
        print(f'\nloaded :: {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} of cLoudnetpy94 Class & Status\n')

    ########################################################################################################################################################
    #   _    ____ ____ ___     ____ ____ ___  ____ ____    ___  ____ ___ ____
    #   |    |  | |__| |  \    |__/ |__| |  \ |__| |__/    |  \ |__|  |  |__|
    #   |___ |__| |  | |__/    |  \ |  | |__/ |  | |  \    |__/ |  |  |  |  |
    #
    #

    # add more radar data lodaer later on
    if system == 'limrad94':
        ZSpec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_2, **spec_settings['VSpec'])
    else:
        raise ValueError('Unknown system.', system)

    # interpolate time dimension of spectra
    ZSpec['VHSpec']['var'] = Loader.replace_fill_value(ZSpec['VHSpec']['var'], ZSpec['SLv']['var'])
    print(f'\nloaded :: {TIME_SPAN_2[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_2[1]:%H:%M:%S} of {system} VHSpectra')

    quick_check(ZSpec['SLv'], f'pseudoZe-{kind}-High-res')

    if kind == 'multispectra':
        n_channels_ = kwargs['n_channels'] if 'n_channels' in kwargs else 3
        ZSpec['VHSpec'] = interpolate3d(ZSpec['VHSpec'], new_time=ZSpec['VHSpec']['ts'], new_range=rg_cnpy94, method=interp)

        quick_check(ZSpec['SLv'], 'pseudoZe_3spec-range_interp')

        # average N time-steps of the radar spectra over the cloudnet time resolution (~30 sec)
        interp_var, interp_mask = multichannelspectra(
            ZSpec['VHSpec']['ts'],
            ZSpec['VHSpec']['var'],
            ZSpec['VHSpec']['mask'],
            new_time=ts_cnpy94,
            n_channels=n_channels_
        )

        ZSpec['VHSpec']['ts'] = ts_cnpy94
        ZSpec['VHSpec']['rg'] = rg_cnpy94
        ZSpec['VHSpec']['var'] = interp_var
        ZSpec['VHSpec']['mask'] = interp_mask
        ZSpec['VHSpec']['dimlabel'] = ['time', 'range', 'vel', 'channel']

        quick_check(cnpy94_class['CLASS'], 'pseudoZe_3spec-time-range-interp')

    elif kind == 'avg30sec':
        # average N time-steps of the radar spectra over the cloudnet time resolution (~30 sec)
        interp_var, interp_mask = average_time_dim(
            ZSpec['VHSpec']['ts'],
            ZSpec['VHSpec']['rg'],
            ZSpec['VHSpec']['var'],
            ZSpec['VHSpec']['mask'],
            new_time=ts_cnpy94
        )

        ZSpec['VHSpec']['ts'] = ts_cnpy94
        ZSpec['VHSpec']['var'] = interp_var
        ZSpec['VHSpec']['mask'] = interp_mask

        quick_check(cnpy94_class['CLASS'], 'pseudoZe_avg30spec-interp')

        ZSpec['VHSpec'] = interpolate3d(ZSpec['VHSpec'], new_time=ts_cnpy94, new_range=rg_cnpy94, method=interp)
        quick_check(cnpy94_class['CLASS'], 'pseudoZe_avg30spec-range-interp')

    else:
        raise ValueError('Unknown KIND of preprocessing.', kind)

    ############################################################################################################################################################
    #   _    ____ ____ ___     ___ ____ ____ _ _  _ _ _  _ ____ ____ ____ ___
    #   |    |  | |__| |  \     |  |__/ |__| | |\ | | |\ | | __ [__  |___  |
    #   |___ |__| |  | |__/     |  |  \ |  | | | \| | | \| |__] ___] |___  |
    #

    config_global_model = toml.load(voodoo_path + 'ann_model_setting.toml')

    features, targets, masked = Loader.load_data(
        ZSpec['VHSpec'],
        cnpy94_class['CLASS'],
        **config_global_model['feature']['info']
    )

    if save:
        h.change_dir(f'{data_path}/features/{kind}/')
        # save features (subfolders for different tensor dimension)
        FILE_NAME_1 = f'{dt_string}_{system}'
        try:
            savemat(f'{FILE_NAME_1}_features_{kind}.mat', {'features': features}, do_compression=True)
        except Exception as e:
            print('Data too large?', e)

        # same labels for different tensor dimensions
        h.change_dir(f'{data_path}/labels/')
        savemat(f'{FILE_NAME_1}_labels.mat', {'labels': targets}, do_compression=True)
        savemat(f'{FILE_NAME_1}_masked.mat', {'masked': masked}, do_compression=True)
        print(f'save :: {FILE_NAME_1}_limrad94_{kind}_features/labels.mat')

    return features, targets, masked, cnpy94_class['CLASS'], cnpy94_class['detection_status']




########################################################################################################################
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#

if __name__ == '__main__':
    start_time = time()

    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
    DATA_PATH = '/home/sdig/code/larda3/voodoo/data/'
    CASE_LIST = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'

    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    SYSTEM = kwargs['system'] if 'system' in kwargs else 'limrad94'
    KIND = kwargs['kind'] if 'kind' in kwargs else 'multispectra'       # avg30sec or multispectra
    case_string = kwargs['case'] if 'case' in kwargs else '20190801-03'
    case = Loader.load_case_list(CASE_LIST, case_string)
    n_channels_ = 4 if KIND == 'multispectra' else 1

    TIME_SPAN_ = [datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
    dt_string = f'{case["time_interval"][0][:-5]}_{case["time_interval"][0][-4:]}-{case["time_interval"][1][-4:]}'

    features, targets, masked, cn_class, cn_status = load_features_from_nc(
        case_string,
        voodoo_path=VOODOO_PATH,
        data_path=DATA_PATH,
        case_list_path=CASE_LIST,
        kind=KIND,
        system=SYSTEM,
        save=True,
        n_channels=n_channels_
    )

    print('total elapsed time = {:.3f} sec.'.format(time() - start_time))
