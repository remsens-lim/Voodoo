#!/home/sdig/anaconda3/bin/python3
"""
Short description:
    Creating a .mat file containing input features and labels for the VOOODOO neural network.
"""

import sys
from datetime import timedelta, datetime

sys.path.append('../larda/')
sys.path.append('.')

from time import time

from scipy.io import savemat
from scipy.interpolate import RectBivariateSpline, LinearNDInterpolator, interp2d, NearestNDInterpolator

import logging
import toml
from scipy import signal
import numpy as np
from tqdm.auto import tqdm

import pyLARDA
import pyLARDA.SpectraProcessing as sp
import pyLARDA.Transformations as tr
import pyLARDA.helpers as h
from larda.pyLARDA.Transformations import plot_timeheight
from larda.pyLARDA.SpectraProcessing import spectra2moments, load_spectra_rpgfmcw94

import voodoo.libVoodoo.Plot   as Plot
import libVoodoo.Plot as Plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.2.2"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def load_case_list(path, case_name):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case'][case_name]


def load_case_file(path):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case']



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


def load_training_mask(classes, status):
    # cn_good radar and lidar =  3
    # cnpy_good radar and lidar =  1
    # cn_good lidar only =  1
    # cnpy_good lidar only =  3
    # create mask
    valid_samples = np.full(status['var'].shape, False)
    valid_samples[status['var'] == 1] = True  # add good radar radar & lidar
    # valid_samples[status['var'] == 2]  = True   # add good radar only
    valid_samples[classes['var'] == 5] = True  # add mixed-phase class pixel
    # valid_samples[classes['var'] == 6] = True   # add melting layer class pixel
    # valid_samples[classes['var'] == 7] = True   # add melting layer + SCL class pixel
    valid_samples[classes['var'] == 1] = True  # add cloud droplets only class

    # at last, remove lidar only pixel caused by adding cloud droplets only class
    valid_samples[status['var'] == 3] = False
    return ~valid_samples

def load_features_and_labels_conv1d(spectra, classes, **feature_info):
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

    masked = np.all(np.all(spectra_mask, axis=3), axis=2)
    #masked = np.all(np.any(spectra_mask, axis=3), axis=2)
    spec_params = feature_info['VSpec']

    quick_check = feature_info['quick_check'] if 'quick_check' in feature_info else False
    if quick_check:
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
        ZE['mask'] = masked

        fig, _ = tr.plot_timeheight(ZE, var_converter='lin2z', title='bla inside wavelett')  # , **plot_settings)
        Plot.save_figure(fig, name=f'limrad_pseudoZe.png', dpi=200)

    spectra_lims = np.array(spec_params['var_lims'])
    # convert to logarithmic units
    if 'var_converter' in spec_params and 'lin2z' in spec_params['var_converter']:
        spectra_3d = h.get_converter_array('lin2z')[0](spectra_3d)
        spectra_lims = h.get_converter_array('lin2z')[0](spec_params['var_lims'])

    # load scaling functions
    spectra_scaler = scaling(strat='normalize')
    cnt = 0
    feature_list = []
    target_labels = []
    spc_matrix = np.zeros((n_Dbins, n_chan), dtype=np.float32)
    window_fcn = np.kaiser(n_Dbins, 5.0)
    print('\nConv1d Feature Extraction......')
    for iT in tqdm(range(n_time)):
        for iR in range(n_range):
            if masked[iT, iR]: continue  # skip masked values

            spc_list = []
            for iCh in range(n_chan):
                spc_matrix[:, iCh] = spectra_scaler(spectra_3d[iT, iR, :, iCh], spectra_lims[0], spectra_lims[1]) * window_fcn
                spc_list.append(spc_matrix[:, iCh])

            #feature_list.append(np.stack([np.mean(spc_matrix, axis=1), np.std(spc_matrix, axis=1)], axis=1))
            feature_list.append(np.stack(spc_list, axis=1))

            # one hot encoding
            if classes['var'][iT, iR] in [8, 9, 10]:
                target_labels.append([0, 0, 0, 0, 0, 0, 0, 0, 1])  # Insects or ground clutter.
            elif classes['var'][iT, iR] == 7:
                target_labels.append([0, 0, 0, 0, 0, 0, 0, 1, 0])  # Melting ice particles coexisting with cloud liquid droplets.
            elif classes['var'][iT, iR] == 6:
                target_labels.append([0, 0, 0, 0, 0, 0, 1, 0, 0])  # Melting ice particles.
            elif classes['var'][iT, iR] == 5:
                target_labels.append([0, 0, 0, 0, 0, 1, 0, 0, 0])  # Ice coexisting with supercooled liquid droplets.
            elif classes['var'][iT, iR] == 4:
                target_labels.append([0, 0, 0, 0, 1, 0, 0, 0, 0])  # Ice particles.
            elif classes['var'][iT, iR] == 3:
                target_labels.append([0, 0, 0, 1, 0, 0, 0, 0, 0])  # Drizzle or rain coexisting with cloud liquid droplets.
            elif classes['var'][iT, iR] == 2:
                target_labels.append([0, 0, 1, 0, 0, 0, 0, 0, 0])  # Drizzle or rain.
            elif classes['var'][iT, iR] == 1:
                target_labels.append([0, 1, 0, 0, 0, 0, 0, 0, 0])  # Cloud liquid droplets only
            else:
                target_labels.append([1, 0, 0, 0, 0, 0, 0, 0, 0])  # Clear-sky
            cnt += 1

    Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    FEATURES = np.array(feature_list, dtype=np.float32)
    LABELS = np.array(target_labels, dtype=np.float32)

    logger.debug(f'min/max value in features = {np.min(FEATURES)},  maximum = {np.max(FEATURES)}')
    logger.debug(f'min/max value in targets  = {np.min(LABELS)},  maximum = {np.max(LABELS)}')

    return FEATURES, LABELS, masked

def load_features_and_labels_conv2d(spectra, classes, **feature_info):
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

    masked = np.all(np.all(spectra_mask, axis=3), axis=2)
    #masked = np.all(np.any(spectra_mask, axis=3), axis=2)
    spec_params = feature_info['VSpec']
    cwt_params = feature_info['cwt']
    fft_params = feature_info['fft']

    assert len(cwt_params['scales']) == 3, 'The list of scaling parameters 3 values!'
    n_cwt_scales = int(cwt_params['scales'][2])
    scales = np.linspace(cwt_params['scales'][0], cwt_params['scales'][1], n_cwt_scales)

    quick_check = feature_info['quick_check'] if 'quick_check' in feature_info else False
    if quick_check:
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
        ZE['mask'] = masked

        fig, _ = tr.plot_timeheight(ZE, var_converter='lin2z', title='bla inside wavelett')  # , **plot_settings)
        Plot.save_figure(fig, name=f'limrad_pseudoZe.png', dpi=200)

    spectra_lims = np.array(spec_params['var_lims'])
    # convert to logarithmic units
    if 'var_converter' in spec_params and 'lin2z' in spec_params['var_converter']:
        spectra_3d = h.get_converter_array('lin2z')[0](spectra_3d)
        spectra_lims = h.get_converter_array('lin2z')[0](spec_params['var_lims'])

    # load scaling functions
    spectra_scaler = scaling(strat='normalize')
    cwt_scaler = scaling(strat='normalize')
    fft_scaler = scaling(strat='normalize')
    cnt = 0
    feature_list = []
    target_labels = []
    cwt_matrix = np.zeros((n_Dbins, n_cwt_scales, n_chan), dtype=np.float32)
    window_fcn = np.kaiser(n_Dbins, 5.0)
    print('\nConv2d Feature Extraction......')
    for iT in tqdm(range(n_time)):
        for iR in range(n_range):
            if masked[iT, iR]: continue  # skip masked values

            for iCh in range(n_chan):
                spc_matrix = spectra_scaler(spectra_3d[iT, iR, :, iCh], spectra_lims[0], spectra_lims[1]) * window_fcn
                cwtmatr = signal.cwt(spc_matrix, signal.ricker, scales)
                cwt_matrix[:, :, iCh] = cwt_scaler(cwtmatr, cwt_params['var_lims'][0], cwt_params['var_lims'][1]).T

            #_mean = np.mean(cwt_matrix, axis=2)
            #_std  = np.std(cwt_matrix, axis=2)
            #_std[_std < 1.0e-7] = 1.0e-7
            #cwt_tensor = [np.min(cwt_matrix, axis=2), np.max(cwt_matrix, axis=2), np.mean(cwt_matrix, axis=2), np.std(cwt_matrix, axis=2)]
            #cwt_tensor = [np.resize(np.mean(cwt_matrix, axis=2), (n_cwt_scales, n_cwt_scales)),
            #              np.resize(np.std(cwt_matrix, axis=2), (n_cwt_scales, n_cwt_scales))]
            #for iCh in range(n_chan):
            #    cwt_matrix[:, :, iCh] = (cwt_matrix[:, :, iCh] - _mean)/_std

            #cwt_tensor = [cwt_matrix[:, :, icwt] for icwt in range(n_chan)]

            cwt_tensor = [np.mean(cwt_matrix, axis=2), np.std(cwt_matrix, axis=2)]
            feature_list.append(np.stack(cwt_tensor, axis=2))
#
#            # TRY THIS IF CWT WITH MINIMUM; MAXIMUM; MEAN; STD FAILS
#            # cut of second half because the real part is symmetric
#            fft_tensor = [fft_scaler(np.abs(np.fft.fft(spc)[:n_Dbins // 2]).T, fft_params[0], fft_params[1]) for spc in spc_tensor]
#            cwtfft_tensor = [fft_scaler(np.abs(np.fft.fft2(cwt)), fft_params['var_lims'][0], fft_params['var_lims'][1]) for cwt in cwt_tensor]


            # one hot encoding
            if classes['var'][iT, iR] in [8, 9, 10]:
                target_labels.append([0, 0, 0, 0, 0, 0, 0, 0, 1])  # Insects or ground clutter.
            elif classes['var'][iT, iR] == 7:
                target_labels.append([0, 0, 0, 0, 0, 0, 0, 1, 0])  # Melting ice particles coexisting with cloud liquid droplets.
            elif classes['var'][iT, iR] == 6:
                target_labels.append([0, 0, 0, 0, 0, 0, 1, 0, 0])  # Melting ice particles.
            elif classes['var'][iT, iR] == 5:
                target_labels.append([0, 0, 0, 0, 0, 1, 0, 0, 0])  # Ice coexisting with supercooled liquid droplets.
            elif classes['var'][iT, iR] == 4:
                target_labels.append([0, 0, 0, 0, 1, 0, 0, 0, 0])  # Ice particles.
            elif classes['var'][iT, iR] == 3:
                target_labels.append([0, 0, 0, 1, 0, 0, 0, 0, 0])  # Drizzle or rain coexisting with cloud liquid droplets.
            elif classes['var'][iT, iR] == 2:
                target_labels.append([0, 0, 1, 0, 0, 0, 0, 0, 0])  # Drizzle or rain.
            elif classes['var'][iT, iR] == 1:
                target_labels.append([0, 1, 0, 0, 0, 0, 0, 0, 0])  # Cloud liquid droplets only
            else:
                target_labels.append([1, 0, 0, 0, 0, 0, 0, 0, 0])  # Clear-sky
            cnt += 1

    Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    FEATURES = np.array(feature_list, dtype=np.float32)
    LABELS = np.array(target_labels, dtype=np.float32)

    logger.debug(f'min/max value in features = {np.min(FEATURES)},  maximum = {np.max(FEATURES)}')
    logger.debug(f'min/max value in targets  = {np.min(LABELS)},  maximum = {np.max(LABELS)}')

    return FEATURES, LABELS, masked


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


def hyperspectralimage(ts, var, msk, **kwargs):

    new_ts = kwargs['new_time']
    n_channels = kwargs['n_channels'] if 'n_channels' in kwargs else 4
    n_ts_new = len(new_ts) if len(new_ts) > 0  else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = var.shape
    mid = n_channels//2

    ip_var = np.zeros((n_ts_new, n_rg, n_vel, n_channels), dtype=np.float32)
    ip_msk = np.empty((n_ts_new, n_rg, n_vel, n_channels), dtype=np.bool)

    print(f'\nConcatinate {n_channels} spectra to 1 sample:\n'
          f'    --> resulting tensor dimension (n_samples, n_velocity_bins, n_cwt_scales, n_channels) = (????, 256, 32, {n_channels}) ......')
    # for iBin in range(n_vel):
    for iBin in tqdm(range(n_vel)):
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
        time_span,
        voodoo_path='',
        data_path='',
        kind='1HSI',
        system='limra94',
        cloudnet='CLOUDNET',
        interp='rectbivar',
        save=True,
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

    spec_settings = toml.load(voodoo_path + 'ann_model_setting.toml')['feature']['info']

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda_connected = pyLARDA.LARDA().connect('lacros_dacapo_gpu')

    TIME_SPAN_ = time_span

    TIME_SPAN_RADAR = [TIME_SPAN_[0] - timedelta(seconds=35.0), TIME_SPAN_[1] + timedelta(seconds=35.0)]
    TIME_SPAN_MODEL = [datetime(TIME_SPAN_[0].year, TIME_SPAN_[0].month, TIME_SPAN_[0].day) + timedelta(minutes=1),
                       datetime(TIME_SPAN_[0].year, TIME_SPAN_[0].month, TIME_SPAN_[0].day) + timedelta(minutes=1439)]

    begin_dt, end_dt = TIME_SPAN_
    dt_string = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}'

    ########################################################################################################################################################
    #
    #   _    ____ ____ ___    / ____ ____ _  _ ____    ____ _    ____ _  _ ___  _  _ ____ ___    ___  ____ ___ ____
    #   |    |  | |__| |  \  /  [__  |__| |  | |___    |    |    |  | |  | |  \ |\ | |___  |     |  \ |__|  |  |__|
    #   |___ |__| |  | |__/ /   ___] |  |  \/  |___    |___ |___ |__| |__| |__/ | \| |___  |     |__/ |  |  |  |  |
    #                      /
    #

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

    cn_variables = load_data(larda_connected, cloudnet, TIME_SPAN_, ['CLASS', 'detection_status'])
    ts_cnpy94, rg_cnpy94 = cn_variables['CLASS']['ts'], cn_variables['CLASS']['rg']

    if save:
        h.change_dir(f'{data_path}/cloudnet/')
        savemat(f'{dt_string}_{cloudnet}_class.mat', cn_variables['CLASS'])
        savemat(f'{dt_string}_{cloudnet}_status.mat', cn_variables['detection_status'])
        print(f'\nloaded :: {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} of cLoudnetpy94 Class & Status\n')

    # temperature and pressure
    cnpy94_model = load_data(larda_connected, cloudnet, TIME_SPAN_MODEL, ['T', 'P'])

    cnpy94_model['T'] = tr.interpolate2d(cnpy94_model['T'], new_time=cn_variables['CLASS']['ts'], new_range=cn_variables['CLASS']['rg'])
    cnpy94_model['P'] = tr.interpolate2d(cnpy94_model['P'], new_time=cn_variables['CLASS']['ts'], new_range=cn_variables['CLASS']['rg'])

    if save:
        h.change_dir(f'{data_path}/cloudnet/')
        savemat(f'{dt_string}_{cloudnet}_model_T.mat', cnpy94_model['T'])
        savemat(f'{dt_string}_{cloudnet}_model_P.mat', cnpy94_model['P'])
        print(f'save :: !interpolated! {dt_string}_{cloudnet}_model_T/P')

    ########################################################################################################################################################
    #   _    ____ ____ ___     ____ ____ ___  ____ ____    ___  ____ ___ ____
    #   |    |  | |__| |  \    |__/ |__| |  \ |__| |__/    |  \ |__|  |  |__|
    #   |___ |__| |  | |__/    |  \ |  | |__/ |  | |  \    |__/ |  |  |  |  |
    #
    #

    # add more radar data lodaer later on
    if system == 'limrad94':
        ZSpec = sp.load_spectra_rpgfmcw94(larda_connected, TIME_SPAN_RADAR, **spec_settings['VSpec'])
    else:
        raise ValueError('Unknown system.', system)

    # interpolate time dimension of spectra
    #artificial_minimum = np.full(ZSpec['SLv']['var'].shape, fill_value=1.e-7)
    #ZSpec['VHSpec']['var'] = replace_fill_value(ZSpec['VHSpec']['var'], artificial_minimum)
    ZSpec['VHSpec']['var'] = replace_fill_value(ZSpec['VHSpec']['var'], ZSpec['SLv']['var'])
    print(f'\nloaded :: {TIME_SPAN_RADAR[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_RADAR[1]:%H:%M:%S} of {system} VHSpectra')

    quick_check(ZSpec['SLv'], f'pseudoZe-{kind}-High-res', '/home/sdig/code/larda3/voodoo/plots/training')

    if kind == 'HSI':
        ZSpec['VHSpec'] = interpolate3d(
            ZSpec['VHSpec'],
            new_time=ZSpec['VHSpec']['ts'],
            new_range=rg_cnpy94,
            method=interp
        )

        quick_check(ZSpec['SLv'], 'pseudoZe_3spec-range_interp', '/home/sdig/code/larda3/voodoo/plots/training')

        # average N time-steps of the radar spectra over the cloudnet time resolution (~30 sec)
        interp_var, interp_mask = hyperspectralimage(
            ZSpec['VHSpec']['ts'],
            ZSpec['VHSpec']['var'],
            ZSpec['VHSpec']['mask'],
            new_time=ts_cnpy94,
            n_channels=kwargs['n_channels']
        )

        ZSpec['VHSpec']['ts'] = ts_cnpy94
        ZSpec['VHSpec']['rg'] = rg_cnpy94
        ZSpec['VHSpec']['var'] = interp_var
        ZSpec['VHSpec']['mask'] = interp_mask
        ZSpec['VHSpec']['dimlabel'] = ['time', 'range', 'vel', 'channel']

        quick_check(cn_variables['CLASS'], 'pseudoZe_3spec-time-range-interp', '/home/sdig/code/larda3/voodoo/plots/training')

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

        quick_check(cn_variables['CLASS'], 'pseudoZe_avg30spec-interp', '/home/sdig/code/larda3/voodoo/plots/training')

        ZSpec['VHSpec'] = interpolate3d(ZSpec['VHSpec'], new_time=ts_cnpy94, new_range=rg_cnpy94, method=interp)
        quick_check(cn_variables['CLASS'], 'pseudoZe_avg30spec-range-interp', '/home/sdig/code/larda3/voodoo/plots/training')

    else:
        raise ValueError('Unknown KIND of preprocessing.', kind)

    ############################################################################################################################################################
    #   _    ____ ____ ___     ___ ____ ____ _ _  _ _ _  _ ____ ____ ____ ___
    #   |    |  | |__| |  \     |  |__/ |__| | |\ | | |\ | | __ [__  |___  |
    #   |___ |__| |  | |__/     |  |  \ |  | | | \| | | \| |__] ___] |___  |
    #

    config_global_model = toml.load(voodoo_path + 'ann_model_setting.toml')
    USE_MODEL = config_global_model['tensorflow']['USE_MODEL']

    if USE_MODEL == 'conv2d':
        features, targets, masked = load_features_and_labels_conv2d(
            ZSpec['VHSpec'],
            cn_variables['CLASS'],
            **config_global_model['feature']['info']
        )
    elif USE_MODEL == 'conv1d':
        features, targets, masked = load_features_and_labels_conv1d(
            ZSpec['VHSpec'],
            cn_variables['CLASS'],
            **config_global_model['feature']['info']
        )
    else:
        raise ValueError(f'Unkown "USE_MODEL" variable = {USE_MODEL}')

    if save:
        h.change_dir(f'{data_path}/features/{kind}/{USE_MODEL}/')
        # save features (subfolders for different tensor dimension)
        FILE_NAME_1 = f'{dt_string}_{system}'
        try:
            savemat(f'{FILE_NAME_1}_features_{kind}.mat', {'features': features})
        except Exception as e:
            print('Data too large?', e)

        # same labels for different tensor dimensions
        h.change_dir(f'{data_path}/labels/')
        savemat(f'{FILE_NAME_1}_labels.mat', {'labels': targets})
        savemat(f'{FILE_NAME_1}_masked.mat', {'masked': masked})
        print(f'save :: {FILE_NAME_1}_limrad94_{kind}_features/labels.mat')

    return features, targets, masked, cn_variables['CLASS'], cn_variables['detection_status'], cnpy94_model['T']




########################################################################################################################
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#

if __name__ == '__main__':
    import traceback
    start_time = time()

    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
    DATA_PATH = '/home/sdig/code/larda3/voodoo/data/'
    CASE_LIST = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'

    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    SYSTEM = kwargs['system'] if 'system' in kwargs else 'limrad94'
    CLOUDNET = kwargs['cnet'] if 'cnet' in kwargs else 'CLOUDNETpy94'
    KIND = kwargs['kind'] if 'kind' in kwargs else 'HSI'
    case_string = kwargs['case'] if 'case' in kwargs else '20190801-01'
    n_channels_ = 6 if 'HSI' in KIND else 1
    load_from_toml = True if 'case' in kwargs else False

    # load case information
    if load_from_toml:
        case = load_case_list(CASE_LIST, case_string)
        TIME_SPAN_ = [datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
    else:
        if 'dt_start' in kwargs and 't_train' in kwargs:
            dt_begin = datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d-%H%M')
            dt_end   = dt_begin + timedelta(minutes=float(kwargs['t_train']))
            TIME_SPAN_ = [dt_begin, dt_end]
        else:
            raise ValueError('Wrong dt_begin or dt_end')

    dt_string = f'{TIME_SPAN_[0]:%Y%m%d}_{TIME_SPAN_[0]:%H%M}-{TIME_SPAN_[1]:%H%M}'

    try:
        features, targets, masked, cn_class, cn_status, cn_temperature = load_features_from_nc(
            time_span=TIME_SPAN_,
            voodoo_path=VOODOO_PATH,
            data_path=DATA_PATH,
            kind=KIND,
            system=SYSTEM,
            cloudnet=CLOUDNET,
            save=True,
            n_channels=n_channels_
        )

    except Exception:
        exc_type, exc_value, exc_tb = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_tb)
        logger.error(ValueError(f'Something went wrong with this interval: {TIME_SPAN_}'))

    logger.critical(f'\n    {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} mat files generated')
