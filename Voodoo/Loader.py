import glob
import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
from typing import List
from itertools import product
import numpy as np
import xarray as xr
from datetime import timedelta, datetime
import torch
import toml
from scipy.interpolate import interp1d

from .Utils import decimalhour2unix, argnearest, set_intersection, ts_to_dt, lin2z, traceback_error, load_training_mask, reshape
from .TorchModel import VoodooNet

DEFAULT_TIME_RES = 30.0  # sec
MODEL_SETUP_FILE = f'model/VnetSettings.toml'

INSTRUMENTS = {

    'cloudnet': ['CLASS', 'detection_status', 'Z', 'VEL', 'VEL_sigma',
                 'width', 'Tw', 'insect_prob', 'beta', 'category_bits', 'quality_bits'],

    'cloudnet_ts': ['LWP'],

    'model': ['T', 'P', 'q', 'UWIND', 'VWIND'],
}

PARAMS = ['filename', 'system', 'colormap', 'rg_unit', 'var_unit']


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


def open_xarray_datasets(path):
    import re
    ds = xr.open_mfdataset(path, parallel=True, decode_times=False, )
    x = re.findall("\d{8}", path)[0]
    # convert time to unix
    ds = ds.assign_coords(time=decimalhour2unix(str(x), ds['time'].values))
    ds['time'].attrs['units'] = 'UTC'
    return ds


def hyperspectralimage(ts: np.array, new_ts: np.array, vhspec: np.array, msk: np.array, n_channels: int=6, n_stride: int=1):

    n_ts_new = len(new_ts) if len(new_ts) > 0 else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = vhspec.shape
    mid = n_stride*n_channels // 2

    ip_var = np.full((n_ts_new, n_rg, n_vel, n_channels), fill_value=-999.0, dtype=np.float32)
    ip_msk = np.full((n_ts_new, n_rg, n_vel, n_channels), fill_value=True, dtype=bool)

    # for iBin in range(n_vel):
    for iBin in range(n_vel):
        for iT_cn in range(n_ts_new):
            iT_rd0 = argnearest(ts, new_ts[iT_cn])
            for i, itmp in enumerate(range(-mid, mid, n_stride)):
                iTdiff = itmp if iT_rd0 + itmp < n_ts else 0
                ip_var[iT_cn, :, iBin, i] = vhspec[iT_rd0 + iTdiff, :, iBin]
                ip_msk[iT_cn, :, iBin, i] = msk[iT_rd0 + iTdiff, :, iBin]

    return ip_var, ip_msk


def VoodooPredictor(X):
    """
    Predict probability disribution over discrete set of 3 classes (dummy, CD, non-CD) .
    
    Args:
        X: list of [256, 6, 1] time spectrograms, dimensitons: (N_Doppler_bins, N_time_steps, N_range_gate)
        
        
    Return:
        predicitons: list of predictions 
    """

    # load architecture
    trained_model = 'model/Vnet0x615580bf-fn1-gpu0-VN.pt'

    torch_settings = toml.load(os.path.join(MODEL_SETUP_FILE))['pytorch']
    torch_settings.update({'dev': 'cpu'})

    # (n_samples, n_Doppler_bins, n_time_steps)
    X = X[:, :, :, np.newaxis]
    X = X.transpose(0, 3, 2, 1)
    X_test = torch.Tensor(X)

    model = VoodooNet(X_test.shape, 3, **torch_settings)
    model.load_state_dict(torch.load(trained_model, map_location=model.device)['state_dict'])

    prediction = model.predict(X_test, batch_size=256)
    prediction = prediction.to('cpu')
    return prediction


def scaling(X, Xlim, strat='none'):
    def ident(x):
        return x

    def norm(x, min_, max_):
        x[x < min_] = min_
        x[x > max_] = max_
        return (x - min_) / max(1.e-15, max_ - min_)

    def minmaxscaler(x, min_, max_):
        x[x < min_] = min_
        x[x > max_] = max_
        return (x - min_) / max(1.e-15, max_ - min_)

    if strat == 'normalize':
        return norm(X, *Xlim)
    elif strat == 'minmaxscaler':
        return minmaxscaler(X, *Xlim)
    else:
        return ident


def load_features_and_labels(spectra, mask, classification, **feature_info):
    n_time, n_range = spectra.shape[:2]
    masked = np.all(np.all(mask, axis=3), axis=2)

    # convert to logarithmic units
    if 'lin2z' in feature_info['var_converter']:
        spectra_scaled = lin2z(spectra.astype('float32'))
        spectra_lims = lin2z(feature_info['var_lims'])
    else:
        spectra_scaled = spectra.astype('float32')
        spectra_lims = np.array(feature_info['var_lims'])

    # load scaling functions
    spectra_scaled = scaling(spectra_scaled, spectra_lims, strat=feature_info['scaling'])

    # add features
    feature_list = []
    for ind_time in range(n_time):
        for ind_range in range(n_range):
            if masked[ind_time, ind_range]:
                continue  # skip MASK values
            feature_list.append(spectra_scaled[ind_time, ind_range, ...])

    # add targets
    target_labels = []
    for ind_time in range(n_time):
        for ind_range in range(n_range):
            if masked[ind_time, ind_range]:
                continue  # skip MASK values
            target_labels.append(classification[ind_time, ind_range])  # sparse one hot encoding

    feature_list = np.array(feature_list, dtype=np.float32)
    target_labels = np.array(target_labels, dtype=np.float32)

    return feature_list, target_labels, masked


def features_from_nc(
        time_span: List[datetime],
        hourly_path='',
        system='limrad94',
        cloudnet: str='CLOUDNETpy94',
        larda_path: str='',
        save: bool=True,
        site: str='lacros_dacapo_rs01',
        build_lists: bool=True,
        n_channels: int=6,
        n_stride: int=1,
        **kwargs
):
    sys.path.append(larda_path)
    import pyLARDA
    import pyLARDA.SpectraProcessing as sp
    import pyLARDA.helpers as h
    import logging
    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    start_time = time.time()
    ds, ds_spec = None, None

    feature_settings = toml.load(MODEL_SETUP_FILE)
    # Load LARDA
    larda_connected = pyLARDA.LARDA().connect(site, build_lists=build_lists)

    TIME_SPAN_ = time_span
    TIME_SPAN_RADAR = [TIME_SPAN_[0] - timedelta(seconds=35.0), TIME_SPAN_[1] + timedelta(seconds=35.0)]

    begin_dt, end_dt = TIME_SPAN_
    dt_string = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}'

    # load radar dara
    ZSpec = {}
    if system == 'limrad94':
        ZSpec = sp.load_spectra_rpgfmcw94(larda_connected, TIME_SPAN_RADAR, **feature_settings['feature']['Spec'])
        ts, rg = ZSpec['VHSpec']['ts'], ZSpec['VHSpec']['rg']
        spectrum = ZSpec['VHSpec']['var']
        mask = ZSpec['mask']['var']
        SL = ZSpec['SLv']['var']
        n_stride = 1

    elif system == 'KAZR':
        ZSpec = larda_connected.read("KAZR", 'specco', TIME_SPAN_RADAR, [100, 8000])
        ts, rg, = ZSpec['ts'], ZSpec['rg']
        spectrum = ZSpec['var']
        mask = ZSpec['mask']
        noise_est = sp.noise_estimation_uncompressed_data(ZSpec, n_std=6.0)
        SL = noise_est['mean']

        old = np.arange(spectrum.shape[2])
        f = interp1d(old, spectrum, axis=2, bounds_error=False, fill_value=-999., kind='linear')
        spectrum = f(np.linspace(old[np.argmin(old)], old[np.argmax(old)], 256))
        n_stride = 2

    else:
        raise ValueError('Unknown system.', system)

    spectrum = replace_fill_value(spectrum, SL)
    #print(f'\nloaded :: {TIME_SPAN_RADAR[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_RADAR[1]:%H:%M:%S} of {system} VHSpectra')

    fig, ax = plt.subplots(figsize=(14, 5))
    ZZ = 10 * np.log10(np.sum(spectrum, axis=(1, 2)))
    #ZZ = reshape(ZZ, _masked_ND)
    ax.pcolormesh(ZZ.T)
    fig.savefig('test.png')

    try:
        cn_class = larda_connected.read(cloudnet, 'CLASS', time_span, [100, 8000])
        cn_stat = larda_connected.read(cloudnet, 'STATUS', time_span, [100, 8000])
        validation_mask = load_training_mask(cn_class['var'], cn_stat['var'])
        ts_main, rg_main = cn_class['ts'], cn_class['rg']

    except Exception as e:
        cn_class = None
        validation_mask = None
        ts_main = np.arange(ts[0], ts[-1], DEFAULT_TIME_RES)
        rg_main = rg
        print(f'WARNING :: Skipped CLoudnet Data --> set {DEFAULT_TIME_RES} sec time.')

    # preprocess spectra
    interp_var, interp_mask = hyperspectralimage(
        ts,
        ts_main,
        spectrum,
        mask,
        n_channels,
        n_stride,
    )

    # reshape spectra from (ts, rg, vel) --> feature dimension (samples, channels, vel)
    features, targets, masked = load_features_and_labels(
        interp_var, interp_mask, cn_class['var'], **feature_settings['feature']['Spec']
    )

    assert features.shape[0] == targets.shape[0], \
        f'No spectra (n_feat={features.shape[0]}) or Cloundet (n_label={targets.shape[0]}) data available!'

    # save features, targets, multitargets, masked to ND.zarr
    ds_spec = VoodooXR(ts_main, rg_main)
    # save features (subfolders for different tensor dimension)
    ds_spec.add_coordinate({'nsamples': np.arange(features.shape[0])}, 'Number of samples')
    ds_spec.add_coordinate({'nvelocity': np.arange(features.shape[1])}, 'Number of velocity bins')
    ds_spec.add_coordinate({'nchannels': np.arange(features.shape[2])}, 'Number of stacked spectra')

    ds_spec.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels'), features, **{})
    ds_spec.add_nD_variable('targets', ('nsamples'), targets, **{})
    ds_spec.add_nD_variable('masked', ('ts', 'rg'), masked, **{})
    ds_spec.add_nD_variable('validation_mask', ('ts', 'rg'), validation_mask, **{})

    if save:
        h.change_dir(hourly_path + f'/{begin_dt.year}/')
        FILE_NAME = f'{dt_string}_voodoo_Xy.zarr'
        try:
            ds_spec.to_zarr(store=FILE_NAME, mode='w', compute=True)
            savednd = '√'
        except Exception as e:
            savednd = 'x'
            print('Data too large?', e)
            print(traceback_error(TIME_SPAN_))

        print(f'DONE   :: {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} zarr files generated, elapsed time = '
              f'{timedelta(seconds=int(time.time() - start_time))} min - F/L-data [{savednd}] ')

    return ds, ds_spec


def dataset_from_zarr_new(data_list, PLOT=False, TASK='train', **kwargs):
    N_NOT_AVAILABLE, N2D_NOT_AVAILABLE = 0, 0

    X, y = [], []
    NA_LIST = []

    for ihour, zarr_file in tqdm(enumerate(data_list), total=len(data_list), unit='files', ncols=100):

        # check if a mat files is available
        try:
            with xr.open_zarr(zarr_file, consolidated=False) as zarr_data:
                _featSPC = zarr_data['features'].values
                _targCLS = zarr_data['targets'].values
                _masked_ND = zarr_data['masked'].values
                _valid = zarr_data['validation_mask'].values

        except:
            N_NOT_AVAILABLE += 1
            NA_LIST.append(zarr_file)
            continue


        if _masked_ND.all():
            continue  # if there are no data points
        if (_targCLS == -999.0).all():
            continue  # if there are no labels available

        # apply training mask
        idx_valid_samples = set_intersection(_masked_ND, _valid)
        if len(idx_valid_samples) < 1:
            continue

        X.append(_featSPC[idx_valid_samples, ...])
        y.append(_targCLS[idx_valid_samples, ...])

    return X, y
