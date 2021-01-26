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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pyLARDA
import pyLARDA.SpectraProcessing as sp
import pyLARDA.Transformations as tr
import pyLARDA.helpers as h
from larda.pyLARDA.Transformations import plot_timeheight
from larda.pyLARDA.SpectraProcessing import spectra2moments, load_spectra_rpgfmcw94

import voodoo.libVoodoo.Plot as Plot
import voodoo.libVoodoo.Utils as Utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.2.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

lidar = 'POLLY'
cloudnet_vars = ['CLASS', 'detection_status', 'Z', 'VEL', 'VEL_sigma', 'width', 'Tw', 'insect_prob', 'beta', 'category_bits', 'quality_bits']
cloudnet_ts_vars = ['LWP']
model_vars = ['T', 'P', 'q', 'UWIND', 'VWIND']
lidar_vars = ['attbsc1064', 'attbsc532', 'depol']
larda_params = ['filename', 'system', 'colormap', 'rg_unit', 'var_unit']


# resolution of python version is master
class VoodooXR(xr.Dataset):

    def __init__(self, _time, _range, *_vel):
        # build xarray dataset
        super().__init__()

        # set metadata
        if _time is not None:
            self.attrs['ts_unit'], self.attrs['ts_unit_long'] = 'sec', 'Unix time, seconds since Jan 1. 1979'
            self.attrs['dt_unit'], self.attrs['dt_unit_long'] = 'date', 'Datetime format'
            self.coords['ts'] = _time
            self.coords['dt'] = [h.ts_to_dt(ts) for ts in _time]
        if _range is not None:
            self.attrs['rg_unit'], self.attrs['rg_unit_long'] = 'm', 'Meter'
            self.coords['rg'] = _range

        # use cloudnet time and range resolution as default
        if len(_vel) > 0: self.coords['vel'] = _vel[0]
        if len(_vel) > 1: self.coords.update({f'vel_{ic + 1}': _vel[ic] for ic in range(1, len(_vel))})

    def _add_coordinate(self, name, unit):
        """
        Adding a coordinate to an xarray structure.
        Args:
            name (dict): key = variable name of the new coordinate, item = long name of the variable
            unit (string): variable unit
            val (numpy.array): values

        """
        for key, item in name.items():
            #self.attrs[key] = item
            self.attrs[f'{key}_unit'] = unit
            self.coords[key] = item

    def add_nD_variable(self, name, dims, val, **kwargs):
        self[name] = (dims, val)
        for key, item in kwargs.items():
            self[name].attrs[key] = item


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


def load_features_and_labels(spectra, classes, category_bits, **feature_info):
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

     clounet bitmask
        :comment = "This variable contains information on the nature of the targets at each pixel,
        thereby facilitating the application of algorithms that work with only one type of target.
        The information is in the form of an array of bits, each of which states either whether a
        certain type of particle is present (e.g. aerosols), or the whether some of the target
        particles have a particular property. The definitions of each bit are given in the definition
        attribute. Bit 0 is the least significant.";
        :definition = "
        Bit 0: Small liquid droplets are present.
        Bit 1: Falling hydrometeors are present; if Bit 2 is set then these are most likely ice particles, otherwise they are drizzle or rain drops.
        Bit 2: Wet-bulb temperature is less than 0 degrees C, implying the phase of Bit-1 particles.
        Bit 3: Melting ice particles are present.
        Bit 4: Aerosol particles are present and visible to the lidar.
        Bit 5: Insects are present and visible to the radar.";
    """
    t0 = time()


    if len(spectra['var'].shape) == 5:
        n_time, n_range, n_Dbins, n_chan, n_pol = spectra['var'].shape
        dual_pol = True
    elif len(spectra['var'].shape) == 4:
        n_time, n_range, n_Dbins, n_chan = spectra['var'].shape
        dual_pol = False
    else:
        raise ValueError('Spectra has wrong dimension!', spectra['var'].shape)

    if classes is None:
        add_labels = False
    else:
        add_labels = True
        bits_uint = category_bits['var'].astype(np.uint8)

    spectra_Ndim = spectra['var'].astype('float32')
    spectra_mask = spectra['mask']
    MASK = np.all(np.all(spectra_mask, axis=3), axis=2)
    spectra_lims = np.array(feature_info['VSpec']['var_lims'])
    # convert to logarithmic units
    if 'var_converter' in feature_info['VSpec'] and 'lin2z' in feature_info['VSpec']['var_converter']:
        spectra_Ndim = h.get_converter_array('lin2z')[0](spectra_Ndim)
        spectra_lims = h.get_converter_array('lin2z')[0](feature_info['VSpec']['var_lims'])

    # load scaling functions
    spectra_scaler = scaling(strat='normalize')
    spectra_scaled = spectra_scaler(spectra_Ndim, spectra_lims[0], spectra_lims[1])
    feature_list, target_labels, multitarget_labels = [], [], []

    logger.info(f'\nConv2D Feature Extraction......')
    iterator = range(n_time) if logger.level > 20 else tqdm(range(n_time))
    for iT in iterator:
        for iR in range(n_range):
            if MASK[iT, iR]: continue  # skip MASK values
            if dual_pol:
                feature_list.append(spectra_scaled[iT, iR, :, :, :])
            else:
                feature_list.append(spectra_scaled[iT, iR, :, :])

            if add_labels:
                target_labels.append(classes['var'][iT, iR])  # sparse one hot encoding
                multitarget_labels.append(np.unpackbits(bits_uint[iT, iR])[2:])

    Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    FEATURES = np.array(feature_list, dtype=np.float32)
    LABELS = np.array(target_labels, dtype=np.float32)
    MULTILABELS = np.array(multitarget_labels, dtype=np.float32)

    logger.debug(f'min/max value in features = {np.min(FEATURES)},  maximum = {np.max(FEATURES)}')
    if add_labels:
        logger.debug(f'min/max value in targets  = {np.min(LABELS)},  maximum = {np.max(LABELS)}')

    return FEATURES, LABELS, MULTILABELS, MASK


def load_data(larda_connected, system, time_span, var_list):
    data = {}
    for i, var in enumerate(var_list):
        try:
            var_info = larda_connected.read(system, var, time_span, [0, 'max'])
            var_info['n_ts'] = var_info['ts'].size if 'ts' in var_info.keys() else None
            var_info['n_rg'] = var_info['rg'].size if 'rg' in var_info.keys() else None
            data.update({var: var_info})
        except Exception as e:
            logger.warning(f'WARNING :: Skipped {system} Data {var} --> set 30 sec time res. as master')
            cloudnet_ts_vars.pop(i)
    return data


def hyperspectralimage(ts, var, msk, **kwargs):
    new_ts = kwargs['new_time']
    n_channels = kwargs['n_channels'] if 'n_channels' in kwargs else _DEFAULT_CHANNELS
    n_ts_new = len(new_ts) if len(new_ts) > 0 else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = var.shape
    mid = n_channels // 2

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


def hyperspectralimage2(ts, vhspec, hspec, msk, **kwargs):
    new_ts = kwargs['new_time']
    n_channels = kwargs['n_channels'] if 'n_channels' in kwargs else _DEFAULT_CHANNELS
    n_ts_new = len(new_ts) if len(new_ts) > 0 else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = vhspec.shape
    mid = n_channels // 2

    ip_var = np.full((n_ts_new, n_rg, n_vel, n_channels, 2), fill_value=-999.0, dtype=np.float32)
    ip_msk = np.full((n_ts_new, n_rg, n_vel, n_channels), fill_value=True)

    logger.info(f'\nConcatinate {n_channels} spectra to 1 sample:\n'
                f'    --> resulting tensor dimension (n_samples, n_velocity_bins, n_channels, 2) = (????, 256, 32, {n_channels}, 2) ......')
    # for iBin in range(n_vel):
    iterator = range(n_vel) if logger.level > 20 else tqdm(range(n_vel))
    for iBin in iterator:
        for iT_cn in range(n_ts_new):
            iT_rd0 = h.argnearest(ts, new_ts[iT_cn])

            for itmp in range(-mid, mid):
                iTdiff = itmp if iT_rd0 + itmp < n_ts else 0
                ip_var[iT_cn, :, iBin, iTdiff + mid, 0] = vhspec[iT_rd0 + iTdiff, :, iBin]
                ip_var[iT_cn, :, iBin, iTdiff + mid, 1] = hspec[iT_rd0 + iTdiff, :, iBin]
                ip_msk[iT_cn, :, iBin, iTdiff + mid] = msk[iT_rd0 + iTdiff, :, iBin]

    return ip_var, ip_msk


def load_features_from_nc(
        time_span,
        voodoo_path='',
        data_path='',
        system='limrad94',
        cloudnet='CLOUDNETpy94',
        interp='rectbivar',
        ann_settings_file='ann_model_settings.toml',
        save=True,
        site='lacros_dacapo_gpu',
        dual_polarization=False,
        **kwargs
):
    def quick_check(path):
        if spec_settings['quick_check']:
            radarMoments = sp.spectra2moments(ZSpec, larda_connected.connectors['LIMRAD94'].system_info['params'], **spec_settings['VSpec'])

            radarMoments['Ze']['var_unit'] = 'dBZ'
            fig, _ = plot_timeheight(
                radarMoments['Ze'],
                var_converter='lin2z',
                title=f'Ze for testing features {dt_string[:8]}'
            )  # , **plot_settings)
            Plot.save_figure(fig, name=f'{path}/limrad_0ZE_{dt_string}.png', dpi=400)

            fig, _ = plot_timeheight(
                radarMoments['VEL'],
                title=f'MDV for testing features {dt_string[:8]}'
            )  # , **plot_settings)
            Plot.save_figure(fig, name=f'{path}/limrad_1MDV_{dt_string}.png', dpi=400)

            fig, _ = plot_timeheight(
                radarMoments['sw'],
                title=f'WIDTH for testing features {dt_string[:8]}'
            )  # , **plot_settings)
            Plot.save_figure(fig, name=f'{path}/limrad_2WIDTH_{dt_string}.png', dpi=400)

            fig, _ = plot_timeheight(
                radarMoments['skew'],
                title=f'SKEW for testing features {dt_string[:8]}'
            )  # , **plot_settings)
            Plot.save_figure(fig, name=f'{path}/limrad_3SKW_{dt_string}.png', dpi=400)

            fig, _ = plot_timeheight(
                radarMoments['kurt'],
                title=f'KURT for testing features {dt_string[:8]}'
            )  # , **plot_settings)
            Plot.save_figure(fig, name=f'{path}/limrad_4KRT_{dt_string}.png', dpi=400)


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
    ZSpec['VHSpec']['var'] = replace_fill_value(ZSpec['VHSpec']['var'], ZSpec['SLv']['var'])
    ZSpec['HSpec']['var'] = replace_fill_value(ZSpec['HSpec']['var'], ZSpec['SLh']['var'])
    logger.info(f'\nloaded :: {TIME_SPAN_RADAR[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_RADAR[1]:%H:%M:%S} of {system} VHSpectra')

    #quick_check(QUICKLOOK_PATH)

    ########################################################################################################################################################
    #
    #   _    ____ ____ ___     ____ _    ____ _  _ ___  _  _ ____ ___    ___  ____ ___ ____
    #   |    |  | |__| |  \    |    |    |  | |  | |  \ |\ | |___  |     |  \ |__|  |  |__|
    #   |___ |__| |  | |__/    |___ |___ |__| |__| |__/ | \| |___  |     |__/ |  |  |  |  |
    #
    try:
        cloudnet_variables = load_data(larda_connected, cloudnet, TIME_SPAN_, cloudnet_vars)
        cloudnet_ts_variables = load_data(larda_connected, cloudnet, TIME_SPAN_, cloudnet_ts_vars)
        cloudnet_model = load_data(larda_connected, cloudnet, TIME_SPAN_MODEL, model_vars)
        ts_master, rg_master = cloudnet_variables['CLASS']['ts'], cloudnet_variables['CLASS']['rg']
        cn_available = True

        CNclass = cloudnet_variables['CLASS']
        CNbits = cloudnet_variables['category_bits']

        # Create a new xarray dataset
        ds = VoodooXR(ts_master, rg_master)

        # Add cloudnet data if available
        if cn_available:
            for ivar in cloudnet_vars:
                ds.add_nD_variable(ivar, ('ts', 'rg'), cloudnet_variables[ivar]['var'], **{key: cloudnet_variables[ivar][key] for key in larda_params})
            for ivar in cloudnet_ts_vars:
                ds.add_nD_variable(
                    ivar,
                    ('ts'),
                    cloudnet_ts_variables[ivar]['var'],
                    **{
                        key: cloudnet_ts_variables[ivar][key] for key in larda_params if key in cloudnet_ts_variables[ivar].keys()
                    }
                )
            for ivar in model_vars:
                cloudnet_model[ivar] = tr.interpolate2d(cloudnet_model[ivar], new_time=ts_master, new_range=rg_master)
                ds.add_nD_variable(ivar, ('ts', 'rg'), cloudnet_model[ivar]['var'], **{key: cloudnet_model[ivar][key] for key in larda_params})

    except Exception as e:
        CNclass, CNbits = None, None
        time_step = 30
        #ts_master, rg_master = cloudnet_variables['CLASS']['ts'], cloudnet_variables['CLASS']['rg']
        ts_master = np.arange(ZSpec['VHSpec']['ts'][0], ZSpec['VHSpec']['ts'][-1], time_step)
        rg_master = ZSpec['VHSpec']['rg']
        logger.warning('WARNING :: Skipped CLoudnet Data --> set 30 sec time.')

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
            ds.add_nD_variable(ivar, ('ts', 'rg'), polly_variables[ivar]['var'], **{key: polly_variables[ivar][key] for key in larda_params})

    except Exception as e:
        logger.warning('WARNING :: Skipped Lidar Data --> set 30 sec time.')

    ########################################################################################################################################################
    #
    #   ___  ____ ____ ___  ____ ____ ____    ____ ___  ____ ____ ___ ____ ____
    #   |__] |__/ |___ |__] |__| |__/ |___    [__  |__] |___ |     |  |__/ |__|
    #   |    |  \ |___ |    |  | |  \ |___    ___] |    |___ |___  |  |  \ |  |
    #

    # average N time-steps of the radar spectra over the cloudnet time resolution (~30 sec)
    if dual_polarization:
        interp_var, interp_mask = hyperspectralimage2(
            ZSpec['VHSpec']['ts'],
            ZSpec['VHSpec']['var'],
            ZSpec['HSpec']['var'],
            ZSpec['VHSpec']['mask'],
            new_time=ts_master,
            n_channels=kwargs['n_channels']
        )
        ZSpec['VHSpec']['dimlabel'] = ['time', 'range', 'vel', 'channel', 'pol']
    else:
        interp_var, interp_mask = hyperspectralimage(
            ZSpec['VHSpec']['ts'],
            ZSpec['VHSpec']['var'],
            ZSpec['VHSpec']['mask'],
            new_time=ts_master,
            n_channels=kwargs['n_channels']
        )
        ZSpec['VHSpec']['dimlabel'] = ['time', 'range', 'vel', 'channel']

    ZSpec['VHSpec']['ts'] = ts_master
    ZSpec['VHSpec']['rg'] = rg_master
    ZSpec['VHSpec']['var'] = interp_var
    ZSpec['VHSpec']['mask'] = interp_mask

    ############################################################################################################################################################
    #   _    ____ ____ ___     ___ ____ ____ _ _  _ _ _  _ ____ ____ ____ ___
    #   |    |  | |__| |  \     |  |__/ |__| | |\ | | |\ | | __ [__  |___  |
    #   |___ |__| |  | |__/     |  |  \ |  | | | \| | | \| |__] ___] |___  |
    #
    config_global_model = toml.load(voodoo_path + ann_settings_file)

    features, targets, multitargets, masked = load_features_and_labels(
        ZSpec['VHSpec'],
        CNclass, CNbits,
        **config_global_model['feature']
    )

    ############################################################################################################################################################
    #   ____ ____ _  _ ____    ___  ____ ____ ____    ____ _ _    ____ ____
    #   [__  |__| |  | |___      /  |__| |__/ |__/    |___ | |    |___ [__
    #   ___] |  |  \/  |___     /__ |  | |  \ |  \    |    | |___ |___ ___]
    #
    if save:
        # save features (subfolders for different tensor dimension)
        ds._add_coordinate({'nsamples': 'Number of samples'}, '-', np.arange(features.shape[0]))
        ds._add_coordinate({'nvelocity': 'Number of velocity bins'}, '-', np.arange(features.shape[1]))
        ds._add_coordinate({'nchannels': 'Number of stacked spectra'}, '-', np.arange(features.shape[2]))
        ds._add_coordinate({'nbits': 'Number of Cloudnet category bits'}, '-', np.arange(6))

        if dual_polarization:
            ds._add_coordinate({'npol': 'Number of polarizations'}, '-', np.arange(features.shape[3]))
            ds.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels', 'npol'), features, **{})
        else:
            ds.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels'), features, **{})
        ds.add_nD_variable('targets', ('nsamples'), targets, **{})
        ds.add_nD_variable('multitargets', ('nsamples', 'nbits'), multitargets, **{})
        ds.add_nD_variable('masked', ('ts', 'rg'), masked, **{})

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

    return features, targets, multitargets, masked, CNclass, ts_master, rg_master


########################################################################################################################
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
if __name__ == '__main__':

    _DEFAULT_CHANNELS = 12
    _DEFAULT_DOPPBINS = 256

    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
    DATA_PATH = f'{VOODOO_PATH}/data_24channel/'
    QUICKLOOK_PATH = f'{VOODOO_PATH}/data/plots/quickcheck/'
    CASE_LIST = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'
    ANN_INI_FILE = 'HP_12chdp.toml'
    #ANN_INI_FILE = 'ann_model_setting.toml'


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
            dt_end = dt_begin + timedelta(minutes=float(kwargs['t_train']))
            TIME_SPAN_ = [dt_begin, dt_end]
        else:
            dt_begin = datetime.strptime('20190801-0500', '%Y%m%d-%H%M')
            dt_end = dt_begin + timedelta(minutes=60.0)
            TIME_SPAN_ = [dt_begin, dt_end]
            # raise ValueError('Wrong dt_begin or dt_end')

    dt_string = f'{TIME_SPAN_[0]:%Y%m%d}_{TIME_SPAN_[0]:%H%M}-{TIME_SPAN_[1]:%H%M}'

    try:
        features, targets, multitargets, masked, classes, ts, rg = load_features_from_nc(
            time_span=TIME_SPAN_,
            voodoo_path=VOODOO_PATH,
            data_path=DATA_PATH,
            system=kwargs['system'] if 'system' in kwargs else 'limrad94',
            cloudnet=kwargs['cnet'] if 'cnet' in kwargs else 'CLOUDNETpy94',
            save=True,
            n_channels=_DEFAULT_CHANNELS,
            ann_settings_file=ANN_INI_FILE,
            site=kwargs['site'] if 'site' in kwargs else 'lacros_dacapo_gpu',
            dual_polarization=True,
        )

    except Exception:
        Utils.traceback_error(TIME_SPAN_)
