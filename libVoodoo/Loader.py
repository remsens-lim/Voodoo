"""
This module contains routines for loading and preprocessing cloud radar and lidar data.

"""

import sys
import datetime
import numpy as np
import time
from copy import deepcopy
from numba import jit
from scipy import signal
from scipy.interpolate import interp1d
import concurrent.futures

import pyLARDA.helpers as h
import pyLARDA.Transformations as trf
from larda.pyLARDA.spec2mom_limrad94 import spectra2moments, build_extended_container

import libVoodoo.Multiscatter as Multiscatter
import libVoodoo.Plot as Plot

__author__      = "Willi Schimmel"
__copyright__   = "Copyright 2019, The Voodoo Project"
__credits__     = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__     = "MIT"
__version__     = "0.0.1"
__maintainer__  = "Willi Schimmel"
__email__       = "willi.schimmel@uni-leipzig.de"
__status__      = "Prototype"

def load_case_list(excel_sheet_path):
    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)
    case_list = []
    if 'date' in kwargs:
        date = str(kwargs['date'])
        if 'begin' in kwargs:
            begin_dt = datetime.datetime.strptime(date + ' ' + kwargs['begin'], '%Y%m%d %H:%M:%S')
            end_dt = datetime.datetime.strptime(date + ' ' + kwargs['end'], '%Y%m%d %H:%M:%S')
        else:
            date = '20190908'
            begin_dt = datetime.datetime.strptime(date + ' 01:00:05', '%Y%m%d %H:%M:%S')
            end_dt = datetime.datetime.strptime(date + ' 09:58:55', '%Y%m%d %H:%M:%S')

        case_list.append({'begin_dt': begin_dt, 'end_dt': end_dt, 'plot_range': [0, 12000], 'notes': '-'})
    else:
        case_list = h.extract_case_from_excel_sheet(excel_sheet_path, sheet_nr=0, kind='ann_input')

    return case_list


def scaling(data, strat='none', **kwargs):
    if 'var_lims' in kwargs:
        assert len(kwargs['var_lims']) == 2, 'Error while loading data, wrong number of var_lims set for scaling!'
    if strat == 'standartize':
        mean = np.nanmean(data)
        std = np.nanstd(data)
        output = standart(data, mean, std)
    elif strat == 'normalize':
        output = norm(data, kwargs['var_lims'][0], kwargs['var_lims'][1])
    else:
        output = data * 1.0
    return output

def ldr2cdr(ldr):
    ldr = np.array(ldr)
    return np.log10(-2.0 * ldr / (ldr - 1.0)) / (np.log10(2) + np.log10(5))

def cdr2ldr(cdr):
    cdr = np.array(cdr)
    return np.power(10.0, cdr) / (2 + np.power(10.0, cdr))


def load_trainingset(spec, mom, lidar, task, **kwargs):
    # normalize the input data
    n_time        = kwargs['n_time']       if 'n_time'      in kwargs else 0
    n_range       = kwargs['n_range']      if 'n_range'     in kwargs else 0
    n_Dbins       = kwargs['n_Dbins']      if 'n_Dbins'     in kwargs else 0
    print_cwt     = kwargs['print_cwt']    if 'print_cwt'   in kwargs else False
    add_moments   = kwargs['add_moments']  if 'add_moments' in kwargs else True
    add_spectra   = kwargs['add_spectra']  if 'add_spectra' in kwargs else True
    add_cwt       = kwargs['add_cwt']      if 'add_cwt'     in kwargs else True
    cwt_params    = kwargs['cwt']          if 'cwt'         in kwargs else {'none': -999}
    # list of features and label
    feature_list  = kwargs['feature_list'] if 'feature_list' in kwargs else ['Ze', 'sw']
    label_list    = kwargs['label_list']   if 'label_list'   in kwargs else ['attbsc1064_ip', 'dpl']
    # normalization boundaries and other variables
    radar_info    = kwargs['feature_info'] if 'feature_info' in kwargs else {'Ze_lims': [1.e-7, 1.e3],
                                                                             'VEL_lims': [-5, 5],
                                                                             'sw_lims': [0, 3],
                                                                             'skew_lims': [-3, 3],
                                                                             'kurt_lims': [0, 3],
                                                                             'normalization': 'none'}
    lidar_info   = kwargs['label_info'] if 'label_info' in kwargs else {'bsc_lims': [1.0e-7, 1.0e-3],
                                                                        'dpl_lims': [1.0e-7, 0.3],
                                                                        'bsc_converter': 'none',
                                                                        'dpl_converter': 'none',
                                                                        'bsc_shift': 0,
                                                                        'dpl_shift': 0,
                                                                        'normalization': 'none'}

    assert n_time*n_range*n_Dbins > 0, f'Error while loading data, n_time(={n_time}) AND n_range(={n_range}) AND n_Dbins(={n_Dbins}) must be larger than 0!'

    # load dimensions,
    add_lidar  = True
    rg_offsets = spec[0]['rg_offsets']
    n_chirps   = len(spec)
    Times, Heights, moments_list, train_set  = [], [], [], []

    ####################################################################################################################################
    #  ____ ____ ___    ___  ____ ___ ____    _  _ ____ ____ _  _
    #  | __ |___  |     |  \ |__|  |  |__|    |\/| |__| [__  |_/
    #  |__] |___  |     |__/ |  |  |  |  |    |  | |  | ___] | \_
    #
    masked = np.invert(np.hstack([np.invert(spec[ic]['mask']).any(axis=2) for ic in range(n_chirps)]))     # default
    if task == 'train_radar_lidar':

        if 'attbsc1064_ip' in lidar:
            masked = np.logical_or(lidar['attbsc1064_ip']['mask'], masked)
            masked[lidar['attbsc1064_ip']['var'] <= lidar_info['attbsc1064_lims'][0]] = True

            if kwargs['label_info']['bsc_converter'] == 'log':
                lidar['attbsc1064_ip']['var'][lidar['attbsc1064_ip']['var'] < lidar_info['attbsc1064_lims'][0]] = lidar_info['attbsc1064_lims'][0]
                lidar['attbsc1064_ip']['var'] = np.log10(lidar['attbsc1064_ip']['var']) + lidar_info['bsc_shift']
                kwargs['label_info']['attbsc1064_lims'] = np.log10(kwargs['label_info']['attbsc1064_lims']) + lidar_info['bsc_shift']

        if 'voldepol532_ip' in lidar:
            masked = np.logical_or(lidar['voldepol532_ip']['mask'], masked)
            masked[lidar['voldepol532_ip']['var'] <= lidar_info['voldepol532_lims'][0]] = True
            if kwargs['label_info']['dpl_converter'] == 'ldr2cdr':
                lidar['voldepol532_ip']['var'][lidar['voldepol532_ip']['var'] >= 1] = lidar_info['voldepol532_lims'][1]
                lidar['voldepol532_ip']['var'] = ldr2cdr(lidar['voldepol532_ip']['var']) + lidar_info['dpl_shift']
                kwargs['label_info']['voldepol532_lims'] = ldr2cdr(kwargs['label_info']['voldepol532_lims']) + lidar_info['dpl_shift']

    n_samples = np.size(masked) - np.count_nonzero(masked)
    print(f'Number of samples in feature set = {n_samples}')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    ____ ____ ___  ____ ____    _  _ ____ _  _ ____ _  _ ___ ____
    #  |__| |  \ |  \ | |\ | | __    |__/ |__| |  \ |__| |__/    |\/| |  | |\/| |___ |\ |  |  [__
    #  |  | |__/ |__/ | | \| |__]    |  \ |  | |__/ |  | |  \    |  | |__| |  | |___ | \|  |  ___]
    #
    if add_moments:
        t0 = time.time()
        features = {ivar: scaling(mom[ivar]['var'], strat=radar_info['normalization'], var_lims=radar_info[f'{ivar}_lims']) for ivar in feature_list}

        for iT in range(n_time):
            print(f'Timesteps converted :: {iT+1:5d} of {n_time}', end='\r')
            for iH in range(n_range):
                if not masked[iT, iH]:
                    moments_list.append(np.array([features[feat][iT, iH] for feat in feature_list]))

        train_set = np.array(moments_list, dtype=np.float32)
        Plot.print_elapsed_time(t0, '\nAdded radar moments to features, elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    _    _ ___  ____ ____    _  _ ____ ____ _ ____ ___  _    ____ ____
    #  |__| |  \ |  \ | |\ | | __    |    | |  \ |__| |__/    |  | |__| |__/ | |__| |__] |    |___ [__
    #  |  | |__/ |__/ | | \| |__]    |___ | |__/ |  | |  \     \/  |  | |  \ | |  | |__] |___ |___ ___]
    #
    if add_lidar:
        t0 = time.time()
        train_label = []
        labels = {ivar: scaling(lidar[f'{ivar}_ip']['var'], strat=lidar_info['normalization'], var_lims=lidar_info[f'{ivar}_lims']) for ivar in label_list}

        for iT in range(n_time):
            print(f'Timesteps converted :: {iT+1:5d} of {n_time}', end='\r')
            for iH in range(n_range):
                if not masked[iT, iH]:
                    Times.append(iT)
                    Heights.append(iH)
                    train_label.append(np.array([labels[label][iT, iH] for label in label_list]))

        train_label = np.array(train_label, dtype=np.float32)
        Plot.print_elapsed_time(t0, '\nAdded lidar variables to targets, elapsed time = ')

    ####################################################################################################################################
    #  _ _  _ ___ ____ ____ ___  ____ _    ____ ___ ____    ___ _  _ _ ____ ___     ____ _  _ _ ____ ___
    #  | |\ |  |  |___ |__/ |__] |  | |    |__|  |  |___     |  |__| | |__/ |  \    |    |__| | |__/ |__]
    #  | | \|  |  |___ |  \ |    |__| |___ |  |  |  |___     |  |  | | |  \ |__/    |___ |  | | |  \ |
    #
    # interpolating 3rd chirp of limrad94 spectra to 256 bins
    if add_spectra or add_cwt:
        t0 = time.time()
        spectra_interp = np.zeros((n_time, spec[2]['rg'].size, 256))
        for iH in range(spec[2]['rg'].size):
            print(f'Interpolation spectra(ichirp=3) :: {iH + 1:5d} of {spec[2]["rg"].size}', end='\r')
            for iT in range(n_time):
                spcij = spec[2]['var'][iT, iH, :]
                if not masked[iT, iH + rg_offsets[2]]:
                    f = interp1d(spec[2]['vel'], spcij, kind='linear')
                    spectra_interp[iT, iH, :] = f(np.linspace(spec[2]['vel'][0], spec[2]['vel'][-1], 256))

        Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (multi-core), elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    ___  ____ ___  ___  _    ____ ____    ____ ___  ____ ____ ___ ____ ____
    #  |__| |  \ |  \ | |\ | | __    |  \ |  | |__] |__] |    |___ |__/    [__  |__] |___ |     |  |__/ |__|
    #  |  | |__/ |__/ | | \| |__]    |__/ |__| |    |    |___ |___ |  \    ___] |    |___ |___  |  |  \ |  |
    #
    if add_spectra:
        t0 = time.time()
        spectra_list = np.zeros((n_samples, 256), dtype=np.float32)
        i_sample = 0
        for iT in range(n_time):
            for ic in range(n_chirps):
                for iH in range(spec[ic]['rg'].size):
                    print(f'Timesteps spectra(ichirp={ic + 1}) added :: {iH + 1:5d} of {spec[ic]["rg"].size}', end='\r')
                    if not masked[iT, iH + rg_offsets[ic]]:
                        # assign radar moments reflectivity, mean doppler velocity, spectral width, linear deplo ratio
                        spcij = spectra_interp[iT, iH, :] if ic == 2 else spec[ic]['var'][iT, iH, :]
                        spectra_list[i_sample, :] = scaling(spcij, strat=radar_info['normalization'], var_lims=radar_info[f'spec_lims'])
                        i_sample += 1

        if len(train_set) == 0:
            train_set = spectra_list.astype(np.float32)
        else:
            train_set = np.concatenate((train_set, spectra_list.astype(np.float32)), axis=1)
        Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (multi-core), elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    _ _ _ ____ _  _ ____ _    ____ ___ ____    _  _ _  _ _    ___ _    ____ ____ ____ ____
    #  |__| |  \ |  \ | |\ | | __    | | | |__| |  | |___ |    |___  |  [__     |\/| |  | |     |  | __ |    |  | |__/ |___
    #  |  | |__/ |__/ | | \| |__]    |_|_| |  |  \/  |___ |___ |___  |  ___]    |  | |__| |___  |  |    |___ |__| |  \ |___
    #
    # add continuous wavelet transformation to list
    add_cwt_multi = False
    if add_cwt_multi:
        t0 = time.time()
        spec[2]['var'] = spectra_interp
        assert 'sfacs' in cwt_params['sfacs'], 'The CWT needs scaling factors! No scaling factors were given'
        n_cwt_scales = len(cwt_params['sfacs'])
        assert n_cwt_scales > 0, 'The list of scaling factors has to be positive!'

        cnt = 0
        cwt_list = []
        keywargs = {'scales': cwt_params['sfacs'], 'n_Dbins': n_Dbins, 'var_lims': radar_info['spec_lims']}
        for ic in range(n_chirps):
            print(f'Timesteps cwt(ichirp={ic + 1}) added')
            with concurrent.futures.ProcessPoolExecutor() as executor:
                cwt_list.append(executor.map(multiprocess_cwt,
                                             spec[ic]['var'][np.where(masked[:, rg_offsets[ic]:rg_offsets[ic+1]]==False)],
                                             keywargs))
                cnt += 1

        Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (multi-core), elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    _ _ _ ____ _  _ ____ _    ____ ___ ____    ____ _ _  _ ____ _    ____    ____ ____ ____ ____
    #  |__| |  \ |  \ | |\ | | __    | | | |__| |  | |___ |    |___  |  [__     [__  | |\ | | __ |    |___ __ |    |  | |__/ |___
    #  |  | |__/ |__/ | | \| |__]    |_|_| |  |  \/  |___ |___ |___  |  ___]    ___] | | \| |__] |___ |___    |___ |__| |  \ |___
    #
    if add_cwt:
        t0 = time.time()
        assert 'sfacs' in cwt_params, 'The CWT needs scaling factors! No scaling factors were given'
        n_cwt_scales = len(cwt_params['sfacs'])
        assert n_cwt_scales > 0, 'The list of scaling factors has to be positive!'

        N_cwt_flat = n_cwt_scales * n_Dbins
        cnt = 0
        cwt_list = []
        for iT in range(n_time):
            for ic in range(n_chirps):
                print(f'Timesteps cwt(ichirp={ic + 1}) added :: {iT + 1:5d} of {n_time}', end='\r')
                for iH in range(spec[ic]['rg'].size):
                    if not masked[iT, iH + rg_offsets[ic]]:
                        # assign radar moments reflectivity, mean doppler velocity, spectral width, linear deplo ratio
                        spcij = spectra_interp[iT, iH, :] if ic == 2 else spec[ic]['var'][iT, iH, :]

                        # spcij_minmax = (spcij - radar_info['spec_lims'][0]) / max((radar_info['spec_lims'][1] - radar_info['spec_lims'][0]), 1.e-9)
                        spcij_minmax = scaling(spcij, strat=radar_info['normalization'], var_lims=radar_info[f'spec_lims'])
                        cwtmatr      = signal.cwt(spcij_minmax, signal.ricker, cwt_params['sfacs'])
                        cwt_norm     = scaling(cwtmatr, strat=radar_info['normalization'], var_lims=[np.min(cwtmatr), np.max(cwtmatr)])

                        if cwt_params['dim'] == '1d':
                            cwt_list.append(np.reshape(cwt_norm, N_cwt_flat))
                        else:
                            cwt_list.append(np.reshape(cwt_norm, (n_cwt_scales, n_Dbins, 1)))
                        cnt += 1

                        if print_cwt:  # and spec[ic]['rg'][iH+rg_offsets[ic]] > 3500:
                            z_lims = [0, 1]
                            x_lims = [-6, 6]
                            # show spectra, normalized spectra and wavlet transformation
                            fig, ax = trf.plot_spectra_cwt(spec[ic], cwt_norm, iT, iH,
                                                           vspec_norm=spcij_minmax,
                                                           # features=cwt_features,
                                                           # mira_spec=MIRA_Zspec,
                                                           z_lim=z_lims,
                                                           x_lim=x_lims,
                                                           scales=cwt_params['sfacs'],
                                                           z_converter='lin2z',
                                                           colormap='jet',
                                                           fig_size=[7, 6]
                                                           )
                            fig.tight_layout()
                            fig_name = f'limrad_cwt_{str(cnt).zfill(4)}_iT-iH_{str(iT).zfill(4)}-{str(iH + rg_offsets[ic]).zfill(4)}.png'
                            fig.savefig(fig_name, dpi=150)
                            print(fig_name)

        if len(train_set) == 0:
            train_set = np.array(cwt_list, dtype=np.float32)
        else:
            train_set = np.concatenate((train_set, np.array(cwt_list, dtype=np.float32)), axis=1)
        Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    return train_set, train_label, Times, Heights


def rescale(data, load):
    from sklearn.externals import joblib
    scaler = joblib.load(load)
    return scaler.inverse_transform(data)


@jit(nopython=True, fastmath=True)
def standart(x, mean, std):
    return (x - mean) / std


# @jit(nopython=True, fastmath=True)
def norm(x, mini, maxi):
    x[x < mini] = mini
    x[x > maxi] = maxi
    return (x - mini) / max(1.e-15, maxi - mini)

def multiprocess_cwt(spec, **kwargs):
    scales = kwargs['scales']
    n_Dbins = kwargs['n_Dbins']
    n_cwt_scales = len(scales)
    lims = kwargs['var_lims']

    # assign radar moments reflectivity, mean doppler velocity, spectral width, linear deplo ratio
    spcij_minmax = spec.copy()
    mini, maxi = lims[0], lims[1]

    spcij_minmax[spcij_minmax < mini] = mini
    spcij_minmax[spcij_minmax > maxi] = maxi
    spcij_minmax = (spcij_minmax - mini) / max(1.e-15, maxi - mini)

    cwtmatr = signal.cwt(spcij_minmax, signal.ricker, scales)

    mini_cwt = np.min(cwtmatr)
    maxi_cwt = np.max(cwtmatr)
    cwt_norm = (cwtmatr - mini_cwt) / max(1.e-15, maxi_cwt - mini_cwt)

    return np.reshape(cwt_norm, (n_cwt_scales, n_Dbins, 1))


def equalize_radar_chirps(spec):
    t0 = time.time()
    n_chirps = len(spec)
    Dbins_max  = np.max([len(spec[ic]['vel']) for ic in range(n_chirps)])

    f = interp1d(spec[2]['vel'], spec[2]['var'], axis=2, kind='nearest', bounds_error=False, fill_value=-999.)
    spectra_interp = f(np.linspace(spec[2]['vel'][0], spec[2]['vel'][-1], Dbins_max))
    spectra_interp[spectra_interp < 0.0] = 0.0
    Plot.print_elapsed_time(t0, f'Interpolation of 3rd chirp to {Dbins_max} Doppler bins, elapsed time = ')

    # this will work if n_chirps=3 and n_Dbins(ichirp=3) < n_Dbins(ichirp=2) = n_Dbins(ichirp=1)
    varstack = np.concatenate((spec[0]['var'], spec[1]['var'], spectra_interp), axis=1)
    varstack = h.lin2z(varstack)

    new_spec = h.put_in_container(varstack, spec[0], name='VSpec', mask=np.ma.getmask(varstack))
    new_spec['rg'] = np.hstack((spec[ic]['rg'] for ic in range(n_chirps))).ravel()

    return new_spec

def load_radar_data(larda, begin_dt, end_dt, **kwargs):

    rm_prcp_ghst = kwargs['rm_precip_ghost']  if 'rm_precip_ghost'  in kwargs else False
    rm_crtn_ghst = kwargs['rm_curtain_ghost'] if 'rm_curtain_ghost' in kwargs else False
    dspckl       = kwargs['do_despeckle']     if 'do_despeckle'     in kwargs else False
    dspckl3d     = kwargs['do_despeckle3d']   if 'do_despeckle3d'   in kwargs else 95.
    est_noise    = kwargs['estimate_noise']   if 'estimate_noise'   in kwargs else False
    NF           = kwargs['noise_factor']     if 'noise_factor'     in kwargs else 6.0
    main_peak    = kwargs['main_peak']        if 'main_peak'        in kwargs else True

    start_time = time.time()
    LIMRAD_Zspec = build_extended_container(larda, 'VSpec', begin_dt, end_dt,
                                            rm_precip_ghost=rm_prcp_ghst, do_despeckle3d=dspckl3d,
                                            estimate_noise=est_noise,     noise_factor=NF
                                            )

    LIMRAD94_moments = spectra2moments(LIMRAD_Zspec, larda.connectors['LIMRAD94'].system_info['params'],
                                       despeckle=dspckl, main_peak=main_peak, filter_ghost_C1=rm_crtn_ghst)

    for ic in range(len(LIMRAD_Zspec)):
        LIMRAD_Zspec[ic]['var'][np.isnan(LIMRAD_Zspec[ic]['var'])] = LIMRAD_Zspec[ic]['var_lims'][0]
        for it in range(LIMRAD_Zspec[ic]['ts'].size):
            for ih in range(LIMRAD_Zspec[ic]['rg'].size):
                LIMRAD_Zspec[ic]['var'][it, ih, LIMRAD_Zspec[ic]['var'][it, ih, :] <= LIMRAD_Zspec[ic]['var_lims'][0]] = np.min(
                    LIMRAD_Zspec[ic]['var'][it, ih, :])

    print(f'Read radar data = {datetime.timedelta(seconds=int(time.time()-start_time))} [hour:min:sec]')
    return {'spectra': LIMRAD_Zspec, 'moments': LIMRAD94_moments}


def load_lidar_data(larda, var_list, begin_dt, end_dt, plot_range, **kwargs):
    start_time = time.time()

    lidar_var = {var: larda.read("POLLY", var, [begin_dt, end_dt], plot_range) for var in var_list}

    # remove multiple scattering effects caused by large field of view
    if 'msf' in kwargs and kwargs['msf']:
        assert len(lidar_var) < 2, 'multiple scattering filter needs both attbsc1064 and voldepol532'
        lidar_var['attbsc1064']['var'], lidar_var['voldepol532']['var'] = Multiscatter.apply_filter(
            lidar_var['attbsc1064'], lidar_var['voldepol532'], despeckle=True)

    for var in var_list:
        lidar_var[var]['var'][np.isnan(lidar_var[var]['var'])] = 0.0
        mask1 = lidar_var[var]['var'] <= 0.0
        mask2 = lidar_var[var]['var'] > 0.0
        lidar_var[var]['mask'][mask1] = True
        lidar_var[var]['mask'][mask2] = False

    print(f'Read lidar data = {datetime.timedelta(seconds=int(time.time() - start_time))} [hour:min:sec]')
    return lidar_var


def make_container_from_prediction(pred_list, list_time, list_range, paraminfo, ts, rg, **kwargs):
    pred_var = np.full((ts.size, rg.size), fill_value=-999.0)
    cnt = 0
    for iT, iR in zip(list_time, list_range):
        iT, iR = int(iT), int(iR)
        pred_var[iT, iR] = pred_list[cnt]
        # print(iT, iR, pred_list[cnt], pred_var[iT, iR])
        cnt += 1

    mask = np.full((ts.size, rg.size), fill_value=False)
    mask[pred_var <= -999.0] = True
    pred_var = np.ma.masked_less_equal(pred_var, -999.0)

    container = {'dimlabel': ['time', 'range'],
                 'filename': [],
                 'paraminfo': deepcopy(paraminfo),
                 'rg_unit': paraminfo['rg_unit'],
                 'colormap': paraminfo['colormap'],
                 'var_unit': paraminfo['var_unit'],
                 'var_lims': paraminfo['var_lims'],
                 'system': paraminfo['system'],
                 'name': paraminfo['paramkey'],
                 'rg': rg * 1.0,
                 'ts': ts * 1.0,
                 'mask': mask,
                 'var': pred_var * 1.0
                 }

    return container


# def make_container_from_prediction_spectra(radar, pred_list, list_time, list_range, paraminfo, ts=0, rg=0, vel=0, **kwargs):
#    pred_var = np.full((ts.size, rg.size, vel.size), fill_value=-999.0)
#    cnt = 0
#    for iT, iR in zip(list_time, list_range):
#        iT, iR = int(iT), int(iR)
#        pred_var[iT, iR, :] = pred_list[cnt, :]
#        # print(iT, iR, pred_list[cnt], pred_var[iT, iR])
#        cnt += 1
#
#    mask = np.full((ts.size, rg.size, vel.size), fill_value=False)
#    mask[pred_var <= -999.0] = True
#    pred_var = np.ma.masked_less_equal(pred_var, -999.0)
#
#    container = {'dimlabel': ['time', 'range', 'vel'],
#                 'filename': [],
#                 'paraminfo': copy.deepcopy(paraminfo),
#                 'rg_unit': paraminfo['rg_unit'],
#                 'colormap': paraminfo['colormap'],
#                 'var_unit': paraminfo['var_unit'],
#                 'var_lims': paraminfo['var_lims'],
#                 'system': paraminfo['system'],
#                 'name': paraminfo['paramkey'],
#                 'rg': rg.copy(),
#                 'ts': ts.copy(),
#                 'vel': vel.copy(),
#                 'mask': mask,
#                 'var': pred_var}
#
#    return container

def predict2container(pred, pred_list, dimensions, param_info):
    predictions = {}
    if 'attbsc1064' in pred_list:
        bsc_shift = dimensions['label_info']['bsc_shift'] if 'bsc_shift' in dimensions else 0.0
        bsc_lims = dimensions['label_info']['attbsc1064_lims']

        # make prediction larda container
        attbsc1064_pred = make_container_from_prediction(pred[:, 0], dimensions['list_ts'], dimensions['list_rg'],
                                                     param_info['attbsc1064'],
                                                     dimensions['ts_radar'], dimensions['rg_radar'])

        if dimensions['label_info']['normalization'] == 'normalize':
            attbsc1064_pred['var'] = attbsc1064_pred['var'] * (bsc_lims[1] - bsc_lims[0]) + bsc_lims[0]

        if dimensions['label_info']['bsc_converter'] == 'log':
            attbsc1064_pred['var'] = np.power(10., attbsc1064_pred['var'] - bsc_shift)

        attbsc1064_pred['var_lims'] = bsc_lims

        predictions.update({'attbsc1064_pred': attbsc1064_pred})


    if 'voldepol532' in pred_list:
        dpl_shift = dimensions['label_info']['dpl_shift'] if 'dpl_shift' in dimensions else 0.0
        dpl_lims = dimensions['label_info']['voldepol532_lims']


        voldepol532_pred = make_container_from_prediction(pred[:, 1], dimensions['list_ts'], dimensions['list_rg'],
                                                   param_info['voldepol532'],
                                                   dimensions['ts_radar'], dimensions['rg_radar'])


        if dimensions['label_info']['normalization'] == 'normalize':
            voldepol532_pred['var'] = voldepol532_pred['var'] * (dpl_lims[1] - dpl_lims[0]) + dpl_lims[0]

        if dimensions['label_info']['dpl_converter'] == 'ldr2cdr':
            voldepol532_pred['var'] = cdr2ldr(voldepol532_pred['var'] - dpl_shift)

        voldepol532_pred['var_lims'] = dpl_lims

        predictions.update({'voldepol532_pred': voldepol532_pred})

    return predictions

#
