"""
This module contains routines for loading and preprocessing cloud radar and lidar data.

"""

import logging
import sys
import time
from datetime import timedelta, datetime

import numpy as np
import toml
import xarray as xr
from tqdm.auto import tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from .Utils import interpolate2d, ts_to_dt, lin2z, argnearest

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2021, The Voodoo Project"
__credits__ = ["Willi Schimmel"]
__license__ = "MIT"
__version__ = "1.1.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

_DEFAULT_CHANNELS = 12
_DEFAULT_TIME_RES = 30

preproc_ini = toml.load('preprocessor_ini.toml')


# xarray datastructure, quickly generating multidim arrays in time, range and velocity
class VoodooXR(xr.Dataset):

    def __init__(self, _time, _range, *_vel):
        # build xarray dataset
        super(VoodooXR, self).__init__()

        # set metadata
        if _time is not None:
            self.attrs['ts_unit'], self.attrs['ts_unit_long'] = 'sec', 'Unix time, seconds since Jan 1. 1979'
            self.attrs['dt_unit'], self.attrs['dt_unit_long'] = 'date', 'Datetime format'
            self.coords['ts'] = _time
            self.coords['dt'] = [ts_to_dt(ts) for ts in _time]
        if _range is not None:
            self.attrs['rg_unit'], self.attrs['rg_unit_long'] = 'm', 'Meter'
            self.coords['rg'] = _range

        # use cloudnet time and range resolution as default
        if len(_vel) > 0: self.coords['vel'] = _vel[0]
        if len(_vel) > 1: self.coords.update({f'vel_{ic + 1}': _vel[ic] for ic in range(1, len(_vel))})

    def add_coordinate(self, name, unit):
        """
        Adding a coordinate to an xarray structure.
        Args:
            name (dict): key = variable name of the new coordinate, item = long name of the variable
            unit (string): variable unit
            val (numpy.array): values

        """
        for key, item in name.items():
            # self.attrs[key] = item
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

    n_time, n_range = spectra['var'].shape[:2]

    # convert to logarithmic units
    if 'var_converter' in feature_info and 'lin2z' in feature_info['var_converter']:
        spectra_scaled = lin2z(spectra['var'].astype('float32'))
        spectra_lims = lin2z(feature_info['var_lims'])
    else:
        spectra_scaled = spectra['var'].astype('float32')
        spectra_lims = np.array(feature_info['var_lims'])

    # load scaling functions
    scaler = scaling(strat=feature_info['scaling'])
    spectra_scaled = scaler(spectra_scaled, *spectra_lims)
    masked = np.all(np.all(spectra['mask'], axis=3), axis=2)

    logger.info(f'\nConv2D Feature Extraction......')

    # add features
    feature_list = []
    iterator = range(n_time) if logger.level > 20 else tqdm(range(n_time))
    for iT in iterator:
        for iR in range(n_range):
            if masked[iT, iR]: continue  # skip MASK values
            feature_list.append(spectra_scaled[iT, iR, ...])

    # add targets
    target_labels, multitarget_labels = [], []
    if classes is not None:
        bits_uint = category_bits['var'].astype(np.uint8)
        for iT in iterator:
            for iR in range(n_range):
                if masked[iT, iR]: continue  # skip MASK values
                target_labels.append(classes['var'][iT, iR])  # sparse one hot encoding
                multitarget_labels.append(np.unpackbits(bits_uint[iT, iR])[2:])

    feature_list = np.array(feature_list, dtype=np.float32)
    target_labels = np.array(target_labels, dtype=np.float32)
    multitarget_labels = np.array(multitarget_labels, dtype=np.float32)

    return feature_list, target_labels, multitarget_labels, masked


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
    return data


def hyperspectralimage(ts, vhspec, hspec, msk, n_channels, new_ts):
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
            iT_rd0 = argnearest(ts, new_ts[iT_cn])

            for itmp in range(-mid, mid):
                iTdiff = itmp if iT_rd0 + itmp < n_ts else 0
                ip_var[iT_cn, :, iBin, iTdiff + mid, 0] = vhspec[iT_rd0 + iTdiff, :, iBin]
                ip_var[iT_cn, :, iBin, iTdiff + mid, 1] = hspec[iT_rd0 + iTdiff, :, iBin]
                ip_msk[iT_cn, :, iBin, iTdiff + mid] = msk[iT_rd0 + iTdiff, :, iBin]

    return ip_var, ip_msk


def features_from_nc(
        time_span,
        data_path='',
        system='limrad94',
        cloudnet='CLOUDNETpy94',
        ann_settings_file='',
        save=True,
        site='lacros_dacapo_gpu',
        dual_polarization=False,
        **kwargs
):
    sys.path.append(preproc_ini['larda']['path'])
    from pyLARDA import LARDA
    import pyLARDA.SpectraProcessing as sp
    import pyLARDA.helpers as h

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    start_time = time.time()

    feature_settings = toml.load(ann_settings_file)

    # Load LARDA
    larda_connected = LARDA().connect(site, build_lists=True)

    TIME_SPAN_ = time_span

    TIME_SPAN_RADAR = [TIME_SPAN_[0] - timedelta(seconds=35.0), TIME_SPAN_[1] + timedelta(seconds=35.0)]
    TIME_SPAN_LIDAR = [TIME_SPAN_[0] - timedelta(seconds=60.0), TIME_SPAN_[1] + timedelta(seconds=60.0)]
    TIME_SPAN_MODEL = [datetime(TIME_SPAN_[0].year, TIME_SPAN_[0].month, TIME_SPAN_[0].day) + timedelta(minutes=1),
                       datetime(TIME_SPAN_[0].year, TIME_SPAN_[0].month, TIME_SPAN_[0].day) + timedelta(minutes=1439)]

    begin_dt, end_dt = TIME_SPAN_
    dt_string = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}'

    # load radar dara
    if system == 'limrad94':
        ZSpec = sp.load_spectra_rpgfmcw94(larda_connected, TIME_SPAN_RADAR, **feature_settings['feature']['Spec'])
    else:
        raise ValueError('Unknown system.', system)

    # replace fill values with sensitivity limit or zeros?
    #fill_ = np.full(ZSpec['SLv']['var'].shape, 1.0e-7)
    #ZSpec['VHSpec']['var'] = replace_fill_value(ZSpec['VHSpec']['var'], fill_)
    #ZSpec['HSpec']['var'] = replace_fill_value(ZSpec['HSpec']['var'], fill_)
    ZSpec['VHSpec']['var'] = replace_fill_value(ZSpec['VHSpec']['var'], ZSpec['SLv']['var'])
    ZSpec['HSpec']['var'] = replace_fill_value(ZSpec['HSpec']['var'], ZSpec['SLh']['var'])
    logger.info(f'\nloaded :: {TIME_SPAN_RADAR[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_RADAR[1]:%H:%M:%S} of {system} VHSpectra')

    # input categorize data
    try:
        cloudnet_variables = load_data(larda_connected, cloudnet, TIME_SPAN_, preproc_ini['instruments']['cloudnet'])
        cloudnet_ts_variables = load_data(larda_connected, cloudnet, TIME_SPAN_, preproc_ini['instruments']['cloudnet_ts'])
        cloudnet_model = load_data(larda_connected, cloudnet, TIME_SPAN_MODEL, preproc_ini['instruments']['model'])
        ts_main, rg_main = cloudnet_variables['CLASS']['ts'], cloudnet_variables['CLASS']['rg']
        cn_available = True
        CNclass = cloudnet_variables['CLASS']
        CNbits = cloudnet_variables['category_bits']
    except Exception as e:
        cn_available = False
        CNclass, CNbits = None, None
        ts_main = np.arange(ZSpec['VHSpec']['ts'][0], ZSpec['VHSpec']['ts'][-1], _DEFAULT_TIME_RES)
        rg_main = ZSpec['VHSpec']['rg']
        logger.warning(f'WARNING :: Skipped CLoudnet Data --> set {_DEFAULT_TIME_RES} sec time.')

    # input polly data
    try:
        polly_variables = load_data(larda_connected, 'POLLY', TIME_SPAN_LIDAR, preproc_ini['instruments']['lidar'])
        lidar_available = True
    except Exception as e:
        logger.warning('WARNING :: Skipped Lidar Data --> set 30 sec time.')
        lidar_available = False

    # preprocess spectra
    interp_var, interp_mask = hyperspectralimage(
        ZSpec['VHSpec']['ts'],
        ZSpec['VHSpec']['var'],
        ZSpec['HSpec']['var'],
        ZSpec['VHSpec']['mask'],
        kwargs['n_channels'],
        ts_main
    )

    ZSpec['VHSpec']['dimlabel'] = ['time', 'range', 'vel', 'channel', 'pol']
    ZSpec['VHSpec']['ts'] = ts_main
    ZSpec['VHSpec']['rg'] = rg_main
    ZSpec['VHSpec']['var'] = interp_var
    ZSpec['VHSpec']['mask'] = interp_mask

    # reshape spectra from (ts, rg, vel, pol) --> feature dimension (samples, pol, channels, vel)
    #
    features, targets, multitargets, masked = load_features_and_labels(
        ZSpec['VHSpec'], CNclass, CNbits, **feature_settings['feature']['Spec']
    )

    ############################################################################################################################################################
    #   ____ ____ _  _ ____    ___  ____ ____ ____    ____ _ _    ____ ____
    #   [__  |__| |  | |___      /  |__| |__/ |__/    |___ | |    |___ [__
    #   ___] |  |  \/  |___     /__ |  | |  \ |  \    |    | |___ |___ ___]
    #
    if save:
        h.change_dir(f'{data_path}')

        # Add cloudnet data if available
        if cn_available:
            ds = VoodooXR(ts_main, rg_main)
            # all 2D variables
            for ivar in preproc_ini['instruments']['cloudnet']:
                ds.add_nD_variable(
                    ivar, ('ts', 'rg'), cloudnet_variables[ivar]['var'],
                    **{key: cloudnet_variables[ivar][key] for key in preproc_ini['larda']['params']}
                )
            # all 1D time series data
            for ivar in preproc_ini['instruments']['cloudnet_ts']:
                ds.add_nD_variable(
                    ivar, ('ts'), cloudnet_ts_variables[ivar]['var'],
                    **{key: cloudnet_ts_variables[ivar][key] for key in preproc_ini['larda']['params']
                        if key in preproc_ini['instruments']['cloudnet_ts']}
                )
            # all 2D model data
            for ivar in preproc_ini['instruments']['model']:
                cloudnet_model[ivar] = interpolate2d(cloudnet_model[ivar], new_time=ts_main, new_range=rg_main)
                ds.add_nD_variable(
                    ivar, ('ts', 'rg'), cloudnet_model[ivar]['var'],
                    **{key: cloudnet_model[ivar][key] for key in preproc_ini['larda']['params']}
                )

            if lidar_available:
                for ivar in preproc_ini['instruments']['lidar']:
                    polly_variables[ivar] = interpolate2d(polly_variables[ivar], new_time=ts_main, new_range=rg_main)
                    ds.add_nD_variable(
                        ivar, ('ts', 'rg'), polly_variables[ivar]['var'],
                        **{key: polly_variables[ivar][key] for key in preproc_ini['larda']['params']}
                    )
            try:
                FILE_NAME = f'{dt_string}_{system}-{cloudnet}-2D.zarr'
                ds.to_zarr(store=FILE_NAME, mode='w', compute=True)
                logger.info(f'SAVE FEATURES :: {FILE_NAME}')
            except Exception as e:
                logger.info('Data too large?', e)
            finally:
                logger.critical(f'DONE :: {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} zarr files generated, elapsed time = '
                                f'{timedelta(seconds=int(time.time() - start_time))} min')

        assert targets.shape[0] == features.shape[0], \
            f'Features (N={features.shape[0]}) or Labels (N={targets.shape[0]}) missing!'

        ds_spec = VoodooXR(None, None)
        # save features (subfolders for different tensor dimension)
        ds_spec.add_coordinate({'nsamples': np.arange(features.shape[0])}, 'Number of samples')
        ds_spec.add_coordinate({'nvelocity': np.arange(features.shape[1])}, 'Number of velocity bins')
        ds_spec.add_coordinate({'nchannels': np.arange(features.shape[2])}, 'Number of stacked spectra')
        ds_spec.add_coordinate({'nbits': np.arange(6)}, 'Number of Cloudnet category bits')

        if dual_polarization:
            ds_spec.add_coordinate({'npol': np.arange(features.shape[3])}, 'Number of polarizations')
            ds_spec.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels', 'npol'), features, **{})
        else:
            ds_spec.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels'), features, **{})
        ds_spec.add_nD_variable('targets', ('nsamples'), targets, **{})
        ds_spec.add_nD_variable('multitargets', ('nsamples', 'nbits'), multitargets, **{})
        ds_spec.add_nD_variable('masked', ('ts', 'rg'), masked, **{})

        FILE_NAME = f'{dt_string}_{system}-{cloudnet}-ND.zarr'
        try:
            ds_spec.to_zarr(store=FILE_NAME, mode='w', compute=True)
            logger.info(f'SAVE TARGETS :: {FILE_NAME}')
        except Exception as e:
            logger.info('Data too large?', e)
        finally:
            logger.critical(f'DONE :: {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} zarr files generated, elapsed time = '
                            f'{timedelta(seconds=int(time.time() - start_time))} min')

    return features, targets, multitargets, masked, CNclass, ts_main, rg_main



def spectra_fold_to_zarr(args):
    xr_ds = VoodooXR(None, None)
    # add coordinates
    xr_ds.add_coordinate({'nsamples': np.arange(args[0].shape[0])}, 'Number of training samples')
    xr_ds.add_coordinate({'nvelocity': np.arange(args[0].shape[1])}, 'Number of velocity bins')
    xr_ds.add_coordinate({'nchannels': np.arange(args[0].shape[2])}, 'Number of stacked spectra')
    xr_ds.add_coordinate({'npolarization': np.arange(args[0].shape[3])}, 'vertical(co) and horizontal(cx) polarization')
    xr_ds.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels', 'npolarization'), args[0], **{})
    xr_ds.add_nD_variable('targets', ('nsamples'), args[1], **{})

    return xr_ds


def validation_fold_to_zarr(args):
    xr_ds2D = VoodooXR(args[12], args[13])
    xr_ds2D.add_nD_variable(
        'classes', ('ts', 'rg'), args[3],
        **{'colormap': 'cloudnet_target_new',
           'rg_unit': 'km',
           'var_unit': '',
           'system': 'Cloudnetpy',
           'var_lims': [0, 10]}
    )
    xr_ds2D.add_nD_variable(
        'status', ('ts', 'rg'), args[4],
        **{'colormap': 'cloudnetpy_detection_status',
           'rg_unit': 'km',
           'var_unit': '',
           'system': 'Cloudnetpy',
           'var_lims': [0, 7]}
    )
    xr_ds2D.add_nD_variable(
        'Ze', ('ts', 'rg'), args[17],
        **{'colormap': 'jet',
           'rg_unit': 'km',
           'var_unit': 'dBZ',
           'system': 'Cloudnetpy',
           'var_lims': [-50, 20]}
    )
    xr_ds2D.add_nD_variable(
        'mask', ('ts', 'rg'), args[8],
        **{'colormap': 'coolwarm',
           'rg_unit': 'km', 'var_unit': '',
           'system': 'Cloudnetpy',
           'var_lims': [0, 1]}
    )



