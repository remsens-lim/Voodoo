"""
This module contains routines for loading and preprocessing cloud radar and lidar data.

"""

import logging
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import toml
import torch
import xarray as xr
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
warnings.simplefilter(action='ignore', category=FutureWarning)

from .Utils import ts_to_dt, lin2z, argnearest, decimalhour2unix
from .Utils import set_intersection, cloudnetpy_classes
from .TorchModel import VoodooNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)

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
    elif strat == 'minmaxscaler':
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler
    else:
        return ident


def scaling2(X, Xlim, strat='none'):
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


def load_training_mask(classes, status):
    """
    classes
        "Value 0: Clear sky.\n",
        "Value 1: Cloud liquid droplets only.\n",
        "Value 2: Drizzle or rain.\n",
        "Value 3: Drizzle or rain coexisting with cloud liquid droplets.\n",
        "Value 4: Ice particles.\n",
        "Value 5: Ice coexisting with supercooled liquid droplets.\n",
        "Value 6: Melting ice particles.\n",
        "Value 7: Melting ice particles coexisting with cloud liquid droplets.\n",
        "Value 8: Aerosol particles, no cloud or precipitation.\n",
        "Value 9: Insects, no cloud or precipitation.\n",
        "Value 10: Aerosol coexisting with insects, no cloud or precipitation." ;

    status
        "Value 0: Clear sky.\n",
        "Value 1: Lidar echo only.\n",
        "Value 2: Radar echo but reflectivity may be unreliable as attenuation by rain, melting\n",
        "         ice or liquid cloud has not been corrected.\n",
        "Value 3: Good radar and lidar echos.\n",
        "Value 4: No radar echo but rain or liquid cloud beneath mean that attenuation that would\n",
        "         be experienced is unknown.\n",
        "Value 5: Good radar echo only.\n",
        "Value 6: No radar echo but known attenuation.\n",
        "Value 7: Radar echo corrected for liquid attenuation using microwave radiometer data.\n",
        "Value 8: Radar ground clutter.\n",
        "Value 9: Lidar clear-air molecular scattering." ;
    """
    # create mask
    valid_samples = np.full(status.shape, False)
    valid_samples[status == 3] = True  # add good radar radar & lidar

    valid_samples[classes == 1] = True  # add cloud droplets only class
    #valid_samples[classes == 2] = True  # add drizzle/rain
    valid_samples[classes == 3] = True  # add cloud droplets + drizzle/rain
    valid_samples[classes == 5] = True  # add mixed-phase class pixel
    valid_samples[classes == 6] = True  # add melting layer
    valid_samples[classes == 7] = True  # add melting layer + SCL class pixel

    # at last, remove lidar only pixel caused by adding cloud droplets only class
    valid_samples[status == 1] = False

    return ~valid_samples


def load_features_and_labels(spectra, mask, classes, **feature_info):
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
    spectra_scaled = scaling2(spectra_scaled, spectra_lims, strat=feature_info['scaling'])

    logger.info(f'\nConv2D Feature Extraction......')
    # add features
    feature_list = []
    target_labels = []
    iterator = range(n_time) if logger.level > 20 else tqdm(range(n_time))
    for ind_time in iterator:
        for ind_range in range(n_range):
            if masked[ind_time, ind_range]:
                continue  # skip MASK values
            feature_list.append(spectra_scaled[ind_time, ind_range, ...])
            target_labels.append(classes[ind_time, ind_range])  # sparse one hot encoding

    feature_list = np.array(feature_list, dtype=np.float32)
    target_labels = np.array(target_labels, dtype=np.float32)

    return feature_list, target_labels, masked


def load_data(larda_connected, system, time_span, var_list):
    data = {}
    for i, var in enumerate(var_list):
        data.update({var: larda_connected.read(system, var, time_span, [0, 'max'])})
    return data


def hyperspectralimage_old(ts, vhspec, msk, n_channels, new_ts):
    n_ts_new = len(new_ts) if len(new_ts) > 0 else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = vhspec.shape
    mid = n_channels // 2

    ip_var = np.full((n_ts_new, n_rg, n_vel, n_channels), fill_value=-999.0, dtype=np.float32)
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
                ip_var[iT_cn, :, iBin, iTdiff + mid] = vhspec[iT_rd0 + iTdiff, :, iBin]
                ip_msk[iT_cn, :, iBin, iTdiff + mid] = msk[iT_rd0 + iTdiff, :, iBin]

    return ip_var, ip_msk

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


def features_from_nc(
        time_span,
        system='limrad94',
        cloudnet='CLOUDNETpy94',
        site='lacros_dacapo_gpu',
        larda_path='',
        build_lists=True,
        feature_settings=None,
        default_time_res=30.0,
):
    sys.path.append(larda_path)

    import pyLARDA
    import pyLARDA.SpectraProcessing as sp

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    if feature_settings is None:
        feature_settings = {
            'var_lims': [1.0e-5, 1.0e2],  # linear units mm6/m3
            'var_converter': 'lin2z',  # conversion to dBZ
            'scaling': 'normalize',  # between 0 and 1 using var_lims
            'channels': 6,
            'n_stride': 1,
        }

    # Load LARDA
    larda_connected = pyLARDA.LARDA().connect(site, build_lists=build_lists)
    time_span_radar = [time_span[0] - timedelta(seconds=35.0), time_span[1] + timedelta(seconds=35.0)]

    # load radar data
    ZSpec = {}
    if system == 'limrad94':
        ZSpec = sp.load_spectra_rpgfmcw94(larda_connected, time_span_radar, **feature_settings)
        ts, rg = ZSpec['VHSpec']['ts'], ZSpec['VHSpec']['rg']
        spectrum = ZSpec['VHSpec']['var']
        mask = ZSpec['VHSpec']['mask']
        SL = ZSpec['SLv']['var']

    elif system == 'KAZR':
        ZSpec = larda_connected.read("KAZR", 'specco', time_span_radar, [0, 'max'])
        ts, rg, = ZSpec['ts'], ZSpec['rg']
        spectrum = ZSpec['var']
        mask = ZSpec['mask']
        noise_est = sp.noise_estimation_uncompressed_data(ZSpec, n_std=6.0)
        SL = noise_est['mean']

        old = np.arange(spectrum.shape[2])
        f = interp1d(old, spectrum, axis=2, bounds_error=False, fill_value=-999., kind='linear')
        spectrum = f(np.linspace(old[np.argmin(old)], old[np.argmax(old)], 256))

    else:
        raise ValueError('Unknown system.', system)

    spectrum = replace_fill_value(spectrum, SL)

    try:
        cn_class = larda_connected.read(cloudnet, 'CLASS', time_span, [0, 'max'])
        cn_stat = larda_connected.read(cloudnet, 'detection_status', time_span, [0, 'max'])
    except Exception as e:
        ts_main = np.arange(ts[0], ts[-1], default_time_res)
        cn_class = {'var': np.zeros((ts_main.size, rg.size)), 'ts': ts_main, 'rg': rg}
        cn_stat = {'var': np.zeros((ts_main.size, rg.size)), 'ts': ts_main, 'rg': rg}
        print(f'WARNING :: Skipped CLoudnet Data --> set {default_time_res} sec time.')

    ts_main, rg_main = cn_class['ts'], cn_class['rg']
    # preprocess spectra
    interp_var, interp_mask = hyperspectralimage(
        ts,
        ts_main,
        spectrum,
        mask,
        feature_settings['channels'],
        feature_settings['n_stride'],
    )

    # reshape spectra from (ts, rg, vel) --> feature dimension (samples, channels, vel)
    features, targets, masked = load_features_and_labels(
        interp_var, interp_mask, cn_class['var'], **feature_settings
    )

    # create xarray from features and labels
    assert features.shape[0] * targets.shape[0] > 0, \
        f'No spectra (n_feat={features.shape[0]}) or Cloundet (n_label={targets.shape[0]}) data available!'

    ds_spec = VoodooXR(ts_main, rg_main)
    ds_spec.add_coordinate({'nsamples': np.arange(features.shape[0])}, 'Number of samples')
    ds_spec.add_coordinate({'nvelocity': np.arange(features.shape[1])}, 'Number of velocity bins')
    ds_spec.add_coordinate({'nchannels': np.arange(features.shape[2])}, 'Number of stacked spectra')
    ds_spec.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels'), features, **{})
    ds_spec.add_nD_variable('targets', ('nsamples'), targets, **{})
    ds_spec.add_nD_variable('masked', ('ts', 'rg'), masked, **{})
    ds_spec.add_nD_variable('class', ('ts', 'rg'), cn_class['var'], **{})
    ds_spec.add_nD_variable('status', ('ts', 'rg'), cn_stat['var'], **{})

    return ds_spec


def tensor_from_hourly_zarr(data_list, task=''):
    N_NOT_AVAILABLE, N2D_NOT_AVAILABLE = 0, 0

    X, y, Time, mask, status = [], [], [], [], []
    NA_LIST = []

    for ihour, zarr_file in tqdm(enumerate(data_list), total=len(data_list), unit='files', ncols=100):

        # check if a mat files is available
        try:
            with xr.open_zarr(zarr_file, consolidated=False) as zarr_data:
                features = zarr_data['features'].values
                targets = zarr_data['targets'].values
                msk = zarr_data['masked'].values
                cl = zarr_data['class'].values
                st = zarr_data['status'].values
                ts = zarr_data['ts'].values

        except:
            N_NOT_AVAILABLE += 1
            NA_LIST.append(zarr_file)
            continue
 
        if msk.all():
            N_NOT_AVAILABLE += 1
            NA_LIST.append(zarr_file)
            continue  # if there are no data points

        if (targets == -999.0).all():
            N_NOT_AVAILABLE += 1
            NA_LIST.append(zarr_file)
            continue  # if there are no labels available

        # apply training mask
        if task == 'train':
            valid = load_training_mask(cl, st)
            idx_valid_samples = set_intersection(msk, valid)
            if len(idx_valid_samples) < 1:
                N_NOT_AVAILABLE += 1
                NA_LIST.append(zarr_file)
                continue

            features = features[idx_valid_samples, ...]
            targets = targets[idx_valid_samples, ...]

            # remove samples with low signal to noise
            mean = np.mean(features, axis=(1, 2))
            idx_valid_samples = np.argwhere(mean > 0.01)[:, 0]
            if len(idx_valid_samples) == 0:
                N_NOT_AVAILABLE += 1
                NA_LIST.append(zarr_file)
                continue

            features = features[idx_valid_samples]
            targets = targets[idx_valid_samples]

            # remove samples to high values
            sum = np.sum(np.mean(features, axis=2), axis=1)
            idx_valid_samples = np.argwhere(sum > 300)[:, 0]
            if len(idx_valid_samples) > 0:
                features = np.delete(features, idx_valid_samples, axis=0)
                targets = np.delete(targets, idx_valid_samples, axis=0)

        X.append(features)
        y.append(targets)
        Time.append(ts)
        status.append(st)
        mask.append(msk)

    assert len(X) > 0 and len(y) > 0, f'No features could be loaded. Check zarr files: {data_list}'

    X = np.concatenate(X)
    y = np.concatenate(y)
    Time = np.concatenate(Time)
    status = np.concatenate(status, axis=0)
    mask = np.concatenate(mask, axis=0)

    numbers = dict(zip(np.arange(len(cloudnetpy_classes)), np.zeros(len(cloudnetpy_classes))))
    unique, counts = np.unique(y, return_counts=True)
    numbers.update(dict(zip(unique, counts)))
    print(numbers)

    return torch.tensor(X), torch.tensor(y), Time, status, mask


def cn_from_hourly_zarr(data_list):
    N_NOT_AVAILABLE, N2D_NOT_AVAILABLE = 0, 0

    cl, st, msk = [], [], []
    NA_LIST = []

    for ihour, zarr_file in tqdm(enumerate(data_list), total=len(data_list), unit='files', ncols=100):

        # check if a mat files is available
        try:
            with xr.open_zarr(zarr_file, consolidated=False) as zarr_data:
                masked_ND = zarr_data['masked'].values
                classes = zarr_data['class'].values
                status = zarr_data['status'].values

        except:
            N_NOT_AVAILABLE += 1
            NA_LIST.append(zarr_file)
            continue

        if masked_ND.all():
            continue  # if there are no data points

        cl.append(classes)
        st.append(status)
        msk.append(masked_ND)

    cl = np.concatenate(cl)
    st = np.concatenate(st)
    msk = np.concatenate(msk)

    return cl, st, msk


def spectra_debug_ql(spec, spectra2, mask, vlim=(0, 1), path=''):
    from itertools import product
    from .Plot import load_xy_style, load_cbar_style
    icnt = 0
    for ind_time, ind_range in product(range(spec.shape[0]), range(spec.shape[1])):
        if mask[ind_time, ind_range]:
            continue
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        pmesh = ax[0, 0].pcolormesh(spec[ind_time, ind_range, :, :, 0].T, vmin=vlim[0], vmax=vlim[1], cmap='coolwarm')
        ax[1, 0].pcolormesh(spec[ind_time, ind_range, :, :, 1].T, vmin=vlim[0], vmax=vlim[1], cmap='coolwarm')
        pmesh2 = ax[0, 1].pcolormesh(spectra2[ind_time, ind_range, :, :, 0].T, vmin=0, vmax=1, cmap='coolwarm')
        ax[1, 1].pcolormesh(spectra2[ind_time, ind_range, :, :, 1].T, vmin=0, vmax=1, cmap='coolwarm')
        cbar = fig.colorbar(pmesh, ax=ax[:, 0], shrink=0.95, anchor=(1.05, -0.11))
        cbar2 = fig.colorbar(pmesh2, ax=ax[:, 1], shrink=0.95, anchor=(1.05, -0.11))
        for i, j in product(range(2), range(2)):
            load_xy_style(ax[i, j], xlabel='Doppler bins [-]', ylabel='Time Steps [-]')
        load_cbar_style(cbar, cbar_label='')
        load_cbar_style(cbar2, cbar_label='')
        fig.savefig(f'{path}/spec_{icnt}.png', dpi=200)
        icnt += 1
        print(f':::DBG:::: {path}/spec_{icnt}-ind_time{ind_time}-ind_range{ind_range}.png')


def VoodooPredictor(X, setup_file, model_file):
    """
    Predict probability disribution over discrete set of 3 classes (dummy, CD, non-CD) .

    Args:
        X: list of [256, 6, 1] time spectrograms, dimensitons: (N_Doppler_bins, N_time_steps, N_range_gate)


    Return:
        predicitons: list of predictions
    """
    import torch
    import toml

    # load architecture
    torch_settings = toml.load(setup_file)['pytorch']
    torch_settings.update({'dev': 'cpu', 'task': 'test'})

    # (n_samples, n_Doppler_bins, n_time_steps)
    n_classes = 2
    X = torch.tensor(X)
    print(X.shape)
    X = torch.unsqueeze(X, dim=1)
    X = torch.transpose(X, 3, 2)
    print(X.shape)

    model = VoodooNet(X.shape, n_classes, **torch_settings)
    model.load_state_dict(torch.load(model_file, map_location=model.device)['state_dict'])

    prediction = model.predict(X, batch_size=256)
    prediction = prediction.to('cpu')
    return prediction

def open_xarray_datasets(path):
    import re
    ds = xr.open_mfdataset(path, parallel=True, decode_times=False, )
    x = re.findall("\d{8}", path)[0]
    # convert time to unix
    ds = ds.assign_coords(time = decimalhour2unix(str(x), ds['time'].values))
    ds['time'].attrs['units'] = 'UTC'
    return ds
