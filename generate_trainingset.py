#!/home/sdig/anaconda3/bin/python3
"""
Short description:
    Creating a *.zarr file containing input features and labels for the voodoo neural network.
"""

import sys
from datetime import timedelta, datetime
from time import time
import traceback

import numpy as np
import xarray as xr
from scipy import signal
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator, interp2d, NearestNDInterpolator

import logging
import toml
from tqdm.auto import tqdm

sys.path.append('../larda/')
sys.path.append('.')

import pyLARDA
import pyLARDA.SpectraProcessing as sp
import pyLARDA.Transformations as tr
import pyLARDA.helpers as h
from larda.pyLARDA.Transformations import plot_timeheight
from larda.pyLARDA.SpectraProcessing import spectra2moments, load_spectra_rpgfmcw94

import voodoo.libVoodoo.Plot as Plot
import voodoo.libVoodoo.Utils as Utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.2.2"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


# resolution of python version is master
class VoodooXR(xr.Dataset):

    def __init__(self, _time, _range, *_vel):
        # build xarray dataset
        super().__init__()

        # set metadata
        self.attrs['ts_unit'], self.attrs['ts_unit_long'] = 'sec', 'Unix time, seconds since Jan 1. 1979'
        self.attrs['rg_unit'], self.attrs['rg_unit_long'] = 'm', 'Meter'
        self.attrs['dt_unit'], self.attrs['dt_unit_long'] = 'date', 'Datetime format'

        # use cloudnet time and range resolution as default
        self.coords['ts'] = _time
        self.coords['rg'] = _range
        if len(_vel) > 0: self.coords['vel'] = _vel[0]
        if len(_vel) > 1: self.coords.update({f'vel_{ic+1}': _vel[ic] for ic in range(1, len(_vel))})
        self.coords['dt'] = [h.ts_to_dt(ts) for ts in _time]

    def _add_coordinate(self, name, unit, val):
        """
        Adding a coordinate to an xarray structure.
        Args:
            name (dict): key = variable name of the new coordinate, item = long name of the variable
            unit (string): variable unit
            val (numpy.array): values

        """
        for key, item in name.items():
            self.attrs[key] = item
            self.attrs[f'{key}_unit'] = unit
            self.coords[key] = val

    def _add_nD_variable(self, name, dims, val, **kwargs):
        self[name] = (dims, val)
        for key, item in kwargs.items():
            self[name].attrs[key] = item

_DEFAULT_CHANNELS = 6
_DEFAULT_DOPPBINS = 256

QUICKLOOK_PATH = '/home/sdig/code/larda3/voodoo/plots/'
lidar = 'POLLY'
cloudnet_vars = ['CLASS', 'detection_status']
model_vars = ['T', 'P']
lidar_vars = ['attbsc1064', 'attbsc532', 'depol']
larda_params = ['filename', 'system', 'colormap', 'rg_unit', 'var_unit']

def scaling(strat='none'):
    def ident(x):
        return x

    def norm(x, min_, max_):
        x[x < min_] = min_
        x[x > max_] = max_
        return (x - min_) / max(1.e-15, max_ - min_)

    if strat == 'normalize':
        return norm
    else:
        return ident

def ldr2cdr(ldr):
    ldr = np.array(ldr)
    return np.log10(-2.0 * ldr / (ldr - 1.0)) / (np.log10(2) + np.log10(5))

def cdr2ldr(cdr):
    cdr = np.array(cdr)
    return np.power(10.0, cdr) / (2 + np.power(10.0, cdr))

def replace_fill_value(data, newfill):
    """
    Replaces the fill value of an spectrum container by their time and range specific mean noise level.
    Args:
        data (numpy.array) : 3D spectrum array (time, range, velocity)
        newfill (numpy.array) : 2D new fill values for 3rd dimension (velocity)

    Return:
        var (numpy.array) : spectrum with mean noise
    """

    n_ts, n_rg, _ = data.shape
    var = data.copy()
    masked = np.all(data <= 0.0, axis=2)

    for iT in range(n_ts):
        for iR in range(n_rg):
            if masked[iT, iR]:
                var[iT, iR, :] = newfill[iT, iR]
            else:
                var[iT, iR, var[iT, iR, :] <= 0.0] = newfill[iT, iR]
    return var

def load_features_and_labels(spectra, classes, **feature_info):
    """
     For orientation

    " status
    \nValue 0: Clear sky.
    \nValue 1: Good radar and lidar echos.
    \nValue 2: Good radar echo only.
    \nValue 3: Radar echo, corrected for liquid attenuation.
    \nValue 4: Lidar echo only.
    \nValue 5: Radar echo, uncorrected for liquid attenuation.
    \nValue 6: Radar ground clutter.
    \nValue 7: Lidar clear-air molecular scattering.";

    " classes
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
    t0 = time()

    if len(spectra['var'].shape) == 3:
        (n_time, n_range, n_Dbins), n_chan = spectra['var'].shape, 1
        spectra_3d = spectra['var'].reshape((n_time, n_range, n_Dbins, n_chan)).astype('float32')
        spectra_mask = spectra['mask'].reshape((n_time, n_range, n_Dbins, n_chan))
    elif len(spectra['var'].shape) == 4:
        n_time, n_range, n_Dbins, n_chan = spectra['var'].shape
        spectra_3d = spectra['var'].astype('float32')
        spectra_mask = spectra['mask']
    else:
        raise ValueError('Spectra has wrong dimension!', spectra['var'].shape)

    MASK = np.all(np.all(spectra_mask, axis=3), axis=2)

    quick_check = feature_info['quick_check'] if 'quick_check' in feature_info else False
    if quick_check and classes != []:
        ZE = np.sum(np.mean(spectra_3d, axis=3), axis=2)
        ZE = h.put_in_container(ZE, classes)  # , **kwargs)
        ZE['dimlabel'] = ['time', 'range']
        ZE['name'] = 'pseudoZe'
        ZE['joints'] = ''
        ZE['rg_unit'] = ''
        ZE['colormap'] = 'jet'
        # ZE['paraminfo'] = dict(ZE['paraminfo'][0])
        ZE['system'] = 'LIMRAD94'
        # ZE['var_lims'] = [ZE['var'].min(), ZE['var'].max()]
        ZE['var_lims'] = [-60, 20]
        ZE['var_unit'] = 'dBZ'
        ZE['mask'] = MASK

        fig, _ = tr.plot_timeheight(ZE, var_converter='lin2z', title='bla inside wavelett')  # , **plot_settings)
        Plot.save_figure(fig, name=f'limrad_pseudoZe.png', dpi=200)

    spectra_lims = np.array(feature_info['VSpec']['var_lims'])
    # convert to logarithmic units
    if 'var_converter' in feature_info['VSpec'] and 'lin2z' in feature_info['VSpec']['var_converter']:
        spectra_3d = h.get_converter_array('lin2z')[0](spectra_3d)
        spectra_lims = h.get_converter_array('lin2z')[0](feature_info['VSpec']['var_lims'])

    # load scaling functions
    spectra_scaler = scaling(strat='normalize')
    spectra_scaled = spectra_scaler(spectra_3d, spectra_lims[0], spectra_lims[1])
    feature_list, target_labels = [], []

    logger.info(f'\nConv2D Feature Extraction......')
    iterator = range(n_time) if logger.level > 20 else tqdm(range(n_time))
    for iT in iterator:
        for iR in range(n_range):
            if MASK[iT, iR]: continue  # skip MASK values
            feature_list.append(spectra_scaled[iT, iR, :, :])
            target_labels.append(classes['var'][iT, iR] if classes['var'][iT, iR] < 8 else 8)    # sparse one hot encoding

    Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    FEATURES = np.array(feature_list, dtype=np.float32)
    LABELS = np.array(target_labels, dtype=np.float32)

    logger.debug(f'min/max value in features = {np.min(FEATURES)},  maximum = {np.max(FEATURES)}')
    logger.debug(f'min/max value in targets  = {np.min(LABELS)},  maximum = {np.max(LABELS)}')

    return FEATURES, LABELS, MASK

def load_radar_data(larda_connected, begin_dt, end_dt, **kwargs):
    """ This routine loads the radar spectra from an RPG cloud radar and caluclates the radar moments.

    Args:
        - larda_connected (larda_connected object) : the class for reading NetCDF files
        - begin_dt (datetime) : datetime object containing the start time
        - end_dt (datetime) : datetime object containing the end time

    **Kwargs:
        - rm_precip_ghost (bool) : removing speckle ghost echos which occur in all chirps caused by precipitation, default: False
        - rm_curtain_ghost (bool) : removing gcurtain host echos which occur in chirp==1, caused by strong signals between 2 and 4 km altitude, default: False
        - do_despeckle (bool) : removes isolated pixel in moments, using a 5x5 pixel window, where at least 80% of neighbouring pixel must be nonzeros,
        default: False
        - do_despeckle3d (bool) : removes isolated pixel i spectrum, using a 5x5x5 pixel window, whereat least 95% of neighbouring pixel must be nonzeros,
        default: 0.95
        - estimate_noise (bool) : remove the noise from the Doppler spectra, default: False
        - NF (float) : calculating the noise factor, mean noise level, left and right edge of a signal, if this value is larger than 0.0. The noise_threshold =
        mean_noise + NF * std(mean_noise), default: 6.0
        - main_peak (bool) : if True, calculate moments only for the main peak in the spectrum, default: True
        - fill_value (float) : non-signal values will be set to this value, default: -999.0

    Returns:
        - radar_data (dict) : containing radar Doppler spectra and radar moments, where non-signal pixels == fill_value, the pectra and Ze are
        stored in linear units [mm6 m-3]
    """

    rm_prcp_ghst = kwargs['ghost_echo_1'] if 'ghost_echo_1' in kwargs else True
    rm_crtn_ghst = kwargs['ghost_echo_2'] if 'ghost_echo_2' in kwargs else True
    dspckl = kwargs['despeckle2D'] if 'despeckle2D' in kwargs else True
    dspckl3d = kwargs['do_despeckle3d'] if 'do_despeckle3d' in kwargs else 95.
    est_noise = kwargs['estimate_noise'] if 'estimate_noise' in kwargs else False
    NF = kwargs['noise_factor'] if 'noise_factor' in kwargs else 6.0
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else True
    fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else -999.0

    t0 = time()

    time_span = [begin_dt, end_dt]

    radar_spectra = load_spectra_rpgfmcw94(
        larda_connected,
        time_span,
        rm_precip_ghost=rm_prcp_ghst,
        do_despeckle3d=dspckl3d,
        estimate_noise=est_noise,
        noise_factor=NF)

    radar_moments = spectra2moments(
        radar_spectra,
        larda_connected.connectors['LIMRAD94'].system_info['params'],
        despeckle=dspckl,
        main_peak=main_peak,
        filter_ghost_C1=rm_crtn_ghst
    )

    Plot.print_elapsed_time(t0, f'Reading spectra + moment calculation, elapsed time = ')
    return {'spectra': radar_spectra, 'moments': radar_moments}

def load_data(larda_connected, system, time_span, var_list):
    data = {}
    for i, var in enumerate(var_list):
        var_info = larda_connected.read(system, var, time_span, [0, 'max'])
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

    logger.info('Averaging over N radar spectra time-steps (30 sec avg)...')
    # for iBin in range(n_vel):
    iterator = range(n_vel) if logger.level > 20 else tqdm(range(n_vel))
    for iBin in iterator:

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

def hyperspectralimage(ts, var, msk, **kwargs):

    new_ts = kwargs['new_time']
    n_channels = kwargs['n_channels'] if 'n_channels' in kwargs else _DEFAULT_CHANNELS
    n_ts_new = len(new_ts) if len(new_ts) > 0  else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = var.shape
    mid = n_channels//2

    ip_var = np.zeros((n_ts_new, n_rg, n_vel, n_channels), dtype=np.float32)
    ip_msk = np.empty((n_ts_new, n_rg, n_vel, n_channels), dtype=np.bool)

    logger.info(f'\nConcatinate {n_channels} spectra to 1 sample:\n'
          f'    --> resulting tensor dimension (n_samples, n_velocity_bins, n_cwt_scales, n_channels) = (????, 256, 32, {n_channels}) ......')
    # for iBin in range(n_vel):
    iterator = range(n_vel) if logger.level > 20 else tqdm(range(n_vel))
    for iBin in iterator:
        for iT_cn in range(n_ts_new):
            iT_rd0 = h.argnearest(ts, new_ts[iT_cn])
            for itmp in range(-mid, mid):
                iTdiff = itmp if iT_rd0 + itmp < n_ts else 0
                ip_var[iT_cn, :, iBin, iTdiff + mid] = var[iT_rd0 + iTdiff, :, iBin]
                ip_msk[iT_cn, :, iBin, iTdiff + mid] = msk[iT_rd0 + iTdiff, :, iBin]

    return ip_var, ip_msk

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

    logger.info('Start interpolation......')

    iterator = range(data['vel'].size) if logger.level > 20 else tqdm(range(data['vel'].size))
    for iBin in iterator:
        var, mask = all_var_bins[:, :, iBin], all_mask_bins[:, :, iBin]
        if method == 'rectbivar':
            kx, ky = 1, 1
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
    logger.info("interpolated shape: time {} range {} var {} mask {}".format(
        new_time.shape, new_range.shape, var_interp.shape, mask_interp.shape))

    return interp_data

def load_features_from_nc(
        time_span,
        voodoo_path='',
        data_path='',
        kind='HSI',
        system='limrad94',
        cloudnet='CLOUDNETpy94',
        interp='rectbivar',
        ann_settings_file='ann_model_settings.toml',
        save=True,
        site='lacros_dacapo_gpu',
        **kwargs
):
    def quick_check(dummy_container, name_str, path):
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
            Plot.save_figure(fig, name=f'{path}/limrad_{name_str}_{dt_string}.png', dpi=200)


    start_time = time()
    spec_settings = toml.load(voodoo_path + ann_settings_file)['feature']

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda_connected = pyLARDA.LARDA().connect(site, build_lists=True)

    TIME_SPAN_ = time_span

    TIME_SPAN_RADAR = [TIME_SPAN_[0] - timedelta(seconds=35.0), TIME_SPAN_[1] + timedelta(seconds=35.0)]
    TIME_SPAN_LIDAR = [TIME_SPAN_[0] - timedelta(seconds=60.0), TIME_SPAN_[1] + timedelta(seconds=60.0)]
    TIME_SPAN_MODEL = [datetime(TIME_SPAN_[0].year, TIME_SPAN_[0].month, TIME_SPAN_[0].day) + timedelta(minutes=1),
                       datetime(TIME_SPAN_[0].year, TIME_SPAN_[0].month, TIME_SPAN_[0].day) + timedelta(minutes=1439)]

    begin_dt, end_dt = TIME_SPAN_
    dt_string = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}'
    data_path = f'{data_path}/'

    ########################################################################################################################################################
    #   _    ____ ____ ___     ____ ____ ___  ____ ____    ___  ____ ___ ____
    #   |    |  | |__| |  \    |__/ |__| |  \ |__| |__/    |  \ |__|  |  |__|
    #   |___ |__| |  | |__/    |  \ |  | |__/ |  | |  \    |__/ |  |  |  |  |
    #
    if system == 'limrad94':
        ZSpec = sp.load_spectra_rpgfmcw94(larda_connected, TIME_SPAN_RADAR, **spec_settings['VSpec'])
    else:
        raise ValueError('Unknown system.', system)

    # interpolate time dimension of spectra
    # not sure if this has an effect, when using a masked layer in the neural network
    #artificial_minimum = np.full(ZSpec['SLv']['var'].shape, fill_value=0.0)
    #ZSpec['VHSpec']['var'] = replace_fill_value(ZSpec['VHSpec']['var'], artificial_minimum)
    ZSpec['VHSpec']['var'] = replace_fill_value(ZSpec['VHSpec']['var'], ZSpec['SLv']['var'])
    logger.info(f'\nloaded :: {TIME_SPAN_RADAR[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_RADAR[1]:%H:%M:%S} of {system} VHSpectra')

    quick_check(ZSpec['SLv'], f'pseudoZe-{kind}-High-res', QUICKLOOK_PATH)

    ########################################################################################################################################################
    #
    #   _    ____ ____ ___     ____ _    ____ _  _ ___  _  _ ____ ___    ___  ____ ___ ____
    #   |    |  | |__| |  \    |    |    |  | |  | |  \ |\ | |___  |     |  \ |__|  |  |__|
    #   |___ |__| |  | |__/    |___ |___ |__| |__| |__/ | \| |___  |     |__/ |  |  |  |  |
    #
    try:
        cloudnet_variables = load_data(larda_connected, cloudnet, TIME_SPAN_, cloudnet_vars)
        cloudnet_model = load_data(larda_connected, cloudnet, TIME_SPAN_MODEL, model_vars)
        ts_master, rg_master = cloudnet_variables['CLASS']['ts'], cloudnet_variables['CLASS']['rg']
        cn_available = True

    except Exception as e:
        logger.warning('WARNING :: Skipped Cloudnet Data --> set 30 sec time res. as master')
        ts_master, rg_master = np.arange(ZSpec['VHSpec']['ts'][0], ZSpec['VHSpec']['ts'][-1], 30.0), ZSpec['VHSpec']['rg']
        cn_available = False
        cloudnet_variables, cloudnet_model = {'CLASS': [], 'detection_status': []}, {'T': []}

    # Create a new xarray dataset
    ds = VoodooXR(ts_master, rg_master)

    # Add cloudnet data if available
    if cn_available:
        for ivar in cloudnet_vars:
            ds._add_nD_variable(ivar, ('ts', 'rg'), cloudnet_variables[ivar]['var'], **{key: cloudnet_variables[ivar][key] for key in larda_params})
        for ivar in model_vars:
            cloudnet_model[ivar] = tr.interpolate2d(cloudnet_model[ivar], new_time=ts_master, new_range=rg_master)
            ds._add_nD_variable(ivar, ('ts', 'rg'), cloudnet_model[ivar]['var'], **{key: cloudnet_model[ivar][key] for key in larda_params})

    ########################################################################################################################################################
    #
    #   ____ ____ ____ ___     _    _ ___  ____ ____    ___  ____ ___ ____
    #   |__/ |___ |__| |  \    |    | |  \ |__| |__/    |  \ |__|  |  |__|
    #   |  \ |___ |  | |__/    |___ | |__/ |  | |  \    |__/ |  |  |  |  |
    #
    try:
        polly_variables = load_data(larda_connected, lidar, TIME_SPAN_LIDAR, lidar_vars)

        for ivar in lidar_vars:
            polly_variables[ivar] = tr.interpolate2d(polly_variables[ivar], new_time=ts_master, new_range=rg_master)
            ds._add_nD_variable(ivar, ('ts', 'rg'), polly_variables[ivar]['var'], **{key: polly_variables[ivar][key] for key in larda_params})

    except Exception as e:
        logger.warning('WARNING :: Skipped Lidar Data --> set 30 sec time.')

    ########################################################################################################################################################
    #
    #   ___  ____ ____ ___  ____ ____ ____    ____ ___  ____ ____ ___ ____ ____
    #   |__] |__/ |___ |__] |__| |__/ |___    [__  |__] |___ |     |  |__/ |__|
    #   |    |  \ |___ |    |  | |  \ |___    ___] |    |___ |___  |  |  \ |  |
    #
    if len(ZSpec['VHSpec']['rg']) != len(rg_master):
        ZSpec['VHSpec'] = interpolate3d(
            ZSpec['VHSpec'],
            new_time=ZSpec['VHSpec']['ts'],
            new_range=rg_master,
            method=interp
        )

    quick_check(ZSpec['SLv'], 'pseudoZe_3spec-range_interp', QUICKLOOK_PATH)

    # average N time-steps of the radar spectra over the cloudnet time resolution (~30 sec)
    interp_var, interp_mask = hyperspectralimage(
        ZSpec['VHSpec']['ts'],
        ZSpec['VHSpec']['var'],
        ZSpec['VHSpec']['mask'],
        new_time=ts_master,
        n_channels=kwargs['n_channels']
    )

    ZSpec['VHSpec']['ts'] = ts_master
    ZSpec['VHSpec']['rg'] = rg_master
    ZSpec['VHSpec']['var'] = interp_var
    ZSpec['VHSpec']['mask'] = interp_mask
    ZSpec['VHSpec']['dimlabel'] = ['time', 'range', 'vel', 'channel']


    ############################################################################################################################################################
    #   _    ____ ____ ___     ___ ____ ____ _ _  _ _ _  _ ____ ____ ____ ___
    #   |    |  | |__| |  \     |  |__/ |__| | |\ | | |\ | | __ [__  |___  |
    #   |___ |__| |  | |__/     |  |  \ |  | | | \| | | \| |__] ___] |___  |
    #
    config_global_model = toml.load(voodoo_path + ann_settings_file)
    USE_MODEL = config_global_model['tensorflow']['USE_MODEL']

    features, targets, masked = load_features_and_labels(
        ZSpec['VHSpec'],
        cloudnet_variables['CLASS'],
        USE_MODEL=USE_MODEL,
        **config_global_model['feature']
    )

    ############################################################################################################################################################
    #   ____ ____ _  _ ____    ___  ____ ____ ____    ____ _ _    ____ ____
    #   [__  |__| |  | |___      /  |__| |__/ |__/    |___ | |    |___ [__
    #   ___] |  |  \/  |___     /__ |  | |  \ |  \    |    | |___ |___ ___]
    #
    if save:
        # save features (subfolders for different tensor dimension)
        ds._add_coordinate({'nsamples':  'Number of samples'}, '-', np.arange(features.shape[0]))
        ds._add_coordinate({'nvelocity': 'Number of velocity bins'}, '-', np.arange(features.shape[1]))
        ds._add_coordinate({'nchannels': 'Number of stacked spectra'}, '-', np.arange(features.shape[2]))

        ds._add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels'), features, **{})
        ds._add_nD_variable('targets', ('nsamples'), targets, **{})
        ds._add_nD_variable('masked', ('ts', 'rg'), masked, **{})

        h.change_dir(f'{data_path}/{cloudnet}/xarray/')

        FILE_NAME = f'{data_path}/{cloudnet}/xarray/{dt_string}_{system}.zarr'
        try:
            ds.to_zarr(store=FILE_NAME, mode='w', compute=True)
            logger.info(f'save :: {FILE_NAME}')
        except Exception as e:
            logger.info('Data too large?', e)
        finally:
            logger.critical(f'DONE :: {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} zarr files generated, elapsed time = '
                            f'{timedelta(seconds=int(time() - start_time))} min')

    return features, targets, masked, cloudnet_variables['CLASS'], cloudnet_variables['detection_status'], cloudnet_model['T']


########################################################################################################################
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#

if __name__ == '__main__':

    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
    DATA_PATH = '/home/sdig/code/larda3/voodoo/data/'
    CASE_LIST = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'
    ANN_INI_FILE = 'ann_model_setting3.toml'

    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    case_string = kwargs['case'] if 'case' in kwargs else '20190801-01'
    load_from_toml = True if 'case' in kwargs else False

    # load case information
    if load_from_toml:
        case = Utils.load_case_list(CASE_LIST, case_string)
        TIME_SPAN_ = [datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
    else:
        if 'dt_start' in kwargs and 't_train' in kwargs:
            dt_begin = datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d-%H%M')
            dt_end   = dt_begin + timedelta(minutes=float(kwargs['t_train']))
            TIME_SPAN_ = [dt_begin, dt_end]
        else:
            dt_begin = datetime.strptime('20200-1900', '%Y%m%d-%H%M')
            dt_end   = dt_begin + timedelta(minutes=60.0)
            TIME_SPAN_ = [dt_begin, dt_end]
            #raise ValueError('Wrong dt_begin or dt_end')

    dt_string = f'{TIME_SPAN_[0]:%Y%m%d}_{TIME_SPAN_[0]:%H%M}-{TIME_SPAN_[1]:%H%M}'

    try:
        features, targets, masked, cn_class, cn_status, cn_temperature = load_features_from_nc(
            time_span=TIME_SPAN_,
            voodoo_path=VOODOO_PATH,
            data_path=DATA_PATH,
            kind=kwargs['kind'] if 'kind' in kwargs else 'HSI',
            system=kwargs['system'] if 'system' in kwargs else 'limrad94',
            cloudnet=kwargs['cnet'] if 'cnet' in kwargs else 'CLOUDNETpy94',
            save=True,
            n_channels=kwargs['nchannels'] if 'nchannels' in kwargs else _DEFAULT_CHANNELS,
            ann_settings_file=ANN_INI_FILE,
            site=kwargs['site'] if 'site' in kwargs else 'limtower',
        )

    except Exception:
        Utils.traceback_error(TIME_SPAN_)
