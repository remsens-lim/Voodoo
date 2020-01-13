"""
This module contains routines for loading and preprocessing cloud radar and lidar data.

"""

import sys
from copy import deepcopy

import datetime
import libVoodoo.Multiscatter as Multiscatter
import libVoodoo.Plot as Plot
import numpy as np
import pyLARDA.helpers as h
import time
from numba import jit
from scipy import signal
from scipy.interpolate import interp1d

import logging

logger = logging.getLogger(__name__)

from larda.pyLARDA.spec2mom_limrad94 import spectra2moments, build_extended_container

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


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


def get_mask(spec, lidar, task='training'):
    """This routine generates a 2d mask, where 1 is set if a specific spectrum contains exclusively fill values, otherwise 0.

    Args:
        - spec (list) : list of spectra dictionaries (len(spec)==n_chirps)
        - lidar (dict) : dictionary containing lidar variables

    Kwargs:
        - task (string) : specifies the task for which the mask is generated, either: "training" or "prediction"

    Return:
        - masked (np.array) : binary numpy array, where values==1 are masked elements and values==0 are signals
    """
    if task == 'training':
        if 'attbsc1064_ip' in lidar:
            return np.logical_or(lidar['attbsc1064_ip']['mask'], spec['mask'].all(axis=2))
        if 'voldepol532_ip' in lidar:
            return np.logical_or(lidar['voldepol532_ip']['mask'], spec['mask'].all(axis=2))
    else:
        return spec['mask'].all(axis=2)


def is_key_in_dict(dictionary, var_list):
    """Function that returns true when a dictionary contains variable name from a list that will be used by the ANN."""
    for key, val in dictionary.items():
        if key in var_list and dictionary[key]['used']:
            return True
    return False


def load_trainingset(spec, mom, lidar, **kwargs):
    # normalize the input data
    n_time = kwargs['n_time'] if 'n_time' in kwargs else 0
    n_range = kwargs['n_range'] if 'n_range' in kwargs else 0
    n_Dbins = kwargs['n_Dbins'] if 'n_Dbins' in kwargs else 0
    task = kwargs['task'] if 'task' in kwargs else 'predict'

    # list of feature settings
    feature_info = kwargs['feature_info'] if 'feature_info' in kwargs else False
    # list of target settings
    output_format = kwargs['output_format'] if 'output_format' in kwargs else 'regression'
    target_info = kwargs['target_info'] if 'target_info' in kwargs else False

    # quick check if any dimensions are positiv
    assert n_time * n_range * n_Dbins > 0, f'Error while loading data, n_time(={n_time}) AND n_range(={n_range}) AND n_Dbins(={n_Dbins}) must be larger than 0!'
    # assert feature_info is dict(), 'Settings for features is missing!'
    # assert target_info is dict(), 'Settings for targets is missing!'

    add_moments = is_key_in_dict(feature_info, ['Ze', 'VEL', 'sw', 'skew', 'kurt'])
    add_spectra = is_key_in_dict(feature_info, ['VSpec', 'HSpec', 'Zspec'])
    add_cwt = is_key_in_dict(feature_info, ['cwt'])

    # load dimensions,
    n_chirps = len(spec)
    Times, Heights, moments_list, feature_set = [], [], [], []
    feature_list = [var for var in feature_info.keys() if feature_info[var]['used']]
    target_list = [var for var in target_info.keys() if target_info[var]['used']]

    ####################################################################################################################################
    #  ____ ____ ___    ___  ____ ___ ____    _  _ ____ ____ _  _
    #  | __ |___  |     |  \ |__|  |  |__|    |\/| |__| [__  |_/
    #  |__] |___  |     |__/ |  |  |  |  |    |  | |  | ___] | \_
    #
    masked = get_mask(spec, lidar, task=task)

    if task == 'train':
        if 'attbsc1064_ip' in lidar:
            if target_info['attbsc1064']['var_converter'] == 'log':
                lidar['attbsc1064_ip']['var'][lidar['attbsc1064_ip']['var'] < target_info['attbsc1064']['var_lims'][0]] = target_info['attbsc1064']['var_lims'][
                    0]
                lidar['attbsc1064_ip']['var'] = np.log10(lidar['attbsc1064_ip']['var'])
                target_info['attbsc1064']['var_lims'] = np.log10(target_info['attbsc1064']['var_lims'])

        if 'depol_ip' in lidar:
            if target_info['depol']['var_converter'] == 'ldr2cdr':
                lidar['depol_ip']['var'][lidar['depol_ip']['var'] >= 1] = target_info['depol']['var_lims'][1]
                lidar['depol_ip']['var'] = ldr2cdr(lidar['depol_ip']['var'])
                target_info['depol']['var_lims'] = ldr2cdr(target_info['depol']['var_lims'])

    n_samples = np.size(masked) - np.count_nonzero(masked)
    print(f'Number of samples in feature set = {n_samples}')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    ____ ____ ___  ____ ____    _  _ ____ _  _ ____ _  _ ___ ____
    #  |__| |  \ |  \ | |\ | | __    |__/ |__| |  \ |__| |__/    |\/| |  | |\/| |___ |\ |  |  [__
    #  |  | |__/ |__/ | | \| |__]    |  \ |  | |__/ |  | |  \    |  | |__| |  | |___ | \|  |  ___]
    #
    if add_moments:
        t0 = time.time()
        features = {ivar: scaling(mom[ivar]['var'], strat=feature_info['scaling'], var_lims=feature_info[f'{ivar}_lims']) for ivar in feature_list}

        for iT in range(n_time):
            print(f'Timesteps converted :: {iT + 1:5d} of {n_time}', end='\r')
            for iH in range(n_range):
                if not masked[iT, iH]:
                    moments_list.append(np.array([features[feat][iT, iH] for feat in feature_list]))

        feature_set = np.array(moments_list, dtype=np.float32)
        Plot.print_elapsed_time(t0, '\nAdded radar moments to features, elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    _    _ ___  ____ ____    _  _ ____ ____ _ ____ ___  _    ____ ____
    #  |__| |  \ |  \ | |\ | | __    |    | |  \ |__| |__/    |  | |__| |__/ | |__| |__] |    |___ [__
    #  |  | |__/ |__/ | | \| |__]    |___ | |__/ |  | |  \     \/  |  | |  \ | |  | |__] |___ |___ ___]
    #
    if output_format == 'regression':
        t0 = time.time()
        target_set = []
        labels = {ivar: scaling(lidar[f'{ivar}_ip']['var'], strat=target_info[ivar]['scaling'],
                                var_lims=target_info[ivar][f'var_lims']) for ivar in target_list}

        for iT in range(n_time):
            print(f'Timesteps converted :: {iT + 1:5d} of {n_time}', end='\r')
            for iH in range(n_range):
                if not masked[iT, iH]:
                    Times.append(iT)
                    Heights.append(iH)
                    target_set.append(np.array([labels[label][iT, iH] for label in target_list]))

        target_set = np.array(target_set, dtype=np.float32)
        Plot.print_elapsed_time(t0, '\nAdded lidar variables to targets, elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    _    _ ___  ____ ____    ___  _ _  _ ____ ____ _   _ 
    #  |__| |  \ |  \ | |\ | | __    |    | |  \ |__| |__/    |__] | |\ | |__| |__/  \_/  
    #  |  | |__/ |__/ | | \| |__]    |___ | |__/ |  | |  \    |__] | | \| |  | |  \   |   
    #
    if output_format == 'classification':
        t0 = time.time()
        target_set = []
        for iT, ts in enumerate(spec['ts']):
            print(f'Timesteps converted :: {iT + 1:5d} of {n_time}', end='\r')
            iT_nearest = h.argnearest(lidar['attbsc1064']['ts'], ts)
            for iH, rg in enumerate(spec['rg']):
                if not masked[iT, iH]:
                    iH_nearest = h.argnearest(lidar['attbsc1064']['rg'], rg)
                    Times.append(iT)
                    Heights.append(iH)
                    val = lidar['attbsc1064']['flags'][iT_nearest, iH_nearest]
                    if val in [0, 1, 3]:
                        target_set.append([0])
                    if val == 2:
                        target_set.append([1])

        target_set = np.array(target_set, dtype=np.float32)
        Plot.print_elapsed_time(t0, '\nAdded lidar variables to targets, elapsed time = ')

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
            for iH in range(spec['rg'].size):
                print(f'Timesteps spectra added :: {iH + 1:5d} of {n_range}', end='\r')
                if not masked[iT, iH]:
                    # assign radar moments reflectivity, mean doppler velocity, spectral width, linear deplo ratio
                    spectra_list[i_sample, :] = scaling(spec['var'][iT, iH, :],
                                                        strat=feature_info['Vspec']['scaling'],
                                                        var_lims=feature_info['Vspec'][f'var_lims'])
                    i_sample += 1

        if len(feature_set) == 0:
            feature_set = spectra_list.astype(np.float32)
        else:
            feature_set = np.concatenate((feature_set, spectra_list.astype(np.float32)), axis=1)
        Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (multi-core), elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    _ _ _ ____ _  _ ____ _    ____ ___ ____    _  _ _  _ _    ___ _    ____ ____ ____ ____
    #  |__| |  \ |  \ | |\ | | __    | | | |__| |  | |___ |    |___  |  [__     |\/| |  | |     |  | __ |    |  | |__/ |___
    #  |  | |__/ |__/ | | \| |__]    |_|_| |  |  \/  |___ |___ |___  |  ___]    |  | |__| |___  |  |    |___ |__| |  \ |___
    #
    # add continuous wavelet transformation to list
    #    add_cwt_multi = False
    #    if add_cwt_multi:
    #        t0 = time.time()
    #        assert 'sfacs' in cwt_params['sfacs'], 'The CWT needs scaling factors! No scaling factors were given'
    #        n_cwt_scales = len(cwt_params['sfacs'])
    #        assert n_cwt_scales > 0, 'The list of scaling factors has to be positive!'
    #
    #        cnt = 0
    #        cwt_list = []
    #        keywargs = {'scales': cwt_params['sfacs'], 'n_Dbins': n_Dbins, 'var_lims': feature_info['spec_lims']}
    #        print(f'Timesteps cwt(ichir + 1}) added')
    #        with concurrent.futures.ProcessPoolExecutor() as executor:
    #            cwt_list.append(executor.map(multiprocess_cwt,
    #                                         spec[ic]['var'][np.where(masked[:, rg_offsets[ic]:rg_offsets[ic+1]]==False)],
    #                                         keywargs))
    #            cnt += 1
    #
    #        Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (multi-core), elapsed time = ')

    ####################################################################################################################################
    #  ____ ___  ___  _ _  _ ____    _ _ _ ____ _  _ ____ _    ____ ___ ____    ____ _ _  _ ____ _    ____    ____ ____ ____ ____
    #  |__| |  \ |  \ | |\ | | __    | | | |__| |  | |___ |    |___  |  [__     [__  | |\ | | __ |    |___ __ |    |  | |__/ |___
    #  |  | |__/ |__/ | | \| |__]    |_|_| |  |  \/  |___ |___ |___  |  ___]    ___] | | \| |__] |___ |___    |___ |__| |  \ |___
    #
    if add_cwt:
        cwt_list = wavlet_transformation(spec, masked, **feature_info['cwt'])

        if len(feature_set) == 0:
            feature_set = np.array(cwt_list, dtype=np.float32)
        else:
            feature_set = np.concatenate((feature_set, np.array(cwt_list, dtype=np.float32)), axis=1)

    return feature_set, target_set, Times, Heights


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
    spcij_scaled = spec.copy()
    mini, maxi = lims[0], lims[1]

    spcij_scaled[spcij_scaled < mini] = mini
    spcij_scaled[spcij_scaled > maxi] = maxi
    spcij_scaled = (spcij_scaled - mini) / max(1.e-15, maxi - mini)

    cwtmatr = signal.cwt(spcij_scaled, signal.ricker, scales)

    mini_cwt = np.min(cwtmatr)
    maxi_cwt = np.max(cwtmatr)
    cwt_scaled = (cwtmatr - mini_cwt) / max(1.e-15, maxi_cwt - mini_cwt)

    return np.reshape(cwt_scaled, (n_cwt_scales, n_Dbins, 1))


def wavlet_transformation(spectra, masked, **cwt_params):
    t0 = time.time()
    assert 'scales' in cwt_params, 'The CWT needs scaling factors! No scaling factors were given'
    assert len(cwt_params['scales']) > 0, 'The list of scaling factors has to be positive!'

    n_time, n_range, n_Dbins = spectra['var'].shape
    n_cwt_scales = len(cwt_params['scales'])
    spectra_unit = spectra['var_unit']
    spectra_lims = np.array(spectra['var_lims'])
    spectra_3d = spectra['var'].copy()
    ts_list = spectra['ts']
    rg_list = spectra['rg']
    vel_list = spectra['vel']

    # convert to logarithmic units
    if 'var_converter' in cwt_params and 'lin2z' in cwt_params['var_converter']:
        spectra_unit = 'dBZ'
        spectra_3d = h.get_converter_array('lin2z')[0](spectra_3d)
        spectra_lims = h.get_converter_array('lin2z')[0](spectra_lims)

    cnt = 0
    cwt_list = []
    for iT in range(n_time):
        print(f'Timestep cwt added :: {iT + 1:5d} of {n_time}', end='\r')
        for iH in range(n_range):
            if not masked[iT, iH]:
                spcij_scaled = scaling(spectra_3d[iT, iH, :], strat=cwt_params['scaling'], var_lims=spectra_lims)
                cwtmatr = signal.cwt(spcij_scaled, signal.ricker, cwt_params['scales'])

                if 'chsgn' in cwt_params['var_converter']: cwtmatr *= -1.0
                cwtmatr[cwtmatr < cwt_params['var_lims'][0]] = cwt_params['var_lims'][0]
                cwtmatr[cwtmatr > cwt_params['var_lims'][1]] = cwt_params['var_lims'][1]
                cwt_scaled = scaling(cwtmatr, strat=cwt_params['scaling'], var_lims=cwt_params['var_lims'])
                # cwt_scaled = scaling(cwtmatr, strat=cwt_params['scaling'], var_lims=[np.min(cwtmatr), np.max(cwtmatr)])

                cwt_list.append(np.reshape(cwt_scaled, (n_cwt_scales, n_Dbins, 1)))
                cnt += 1

                if 'plot_cwt' in cwt_params and cwt_params['plot_cwt']:
                    velocity_lims = [-6, 6]
                    # show spectra, normalized spectra and wavlet transformation
                    fig, ax = Plot.spectra_wavelettransform(vel_list, spcij_scaled, cwt_scaled,
                                                            idxts=iT, ts=ts_list[iT],
                                                            idxrg=iH, rg=rg_list[iH],
                                                            colormap='cloudnet_jet',
                                                            x_lim=velocity_lims,
                                                            scales=cwt_params['scales'],
                                                            fig_size=[7, 4]
                                                            )
                    # fig, (top_ax, bottom_left_ax, bottom_right_ax) = Plot.spectra_wavelettransform2(vel_list, spcij_scaled, cwt_params['scales'])
                    Plot.save_figure(fig, name=f'limrad_cwt_{str(cnt).zfill(4)}_iT-iH_{str(iT).zfill(4)}-{str(iH).zfill(4)}.png', dpi=300)

    Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    return cwt_list


def equalize_rpg_radar_chirps(spec, **kwargs):
    """This routine takes a list of larda spectrum containers and equalizes the velocity dimensions.

    Args:
        - spec (list) : spetrum container of len(spec)==n_chirps, and type(spec[i])==dict(), for n_chirps see RPG-FMCW94 Cloud Radar manual.

    **Kwargs:
        - interp_method (string) : kind of the interpolation, see scipy.inperolation.interp1d(), default: nearest
        - fill_value (float) : noise will be set to this value, default: -999.0

    Return:
        - new_spec (dict) : spectrum containing the concatinated chirps, unified to the maximum number of Doppler bins.
    """

    t0 = time.time()
    method = kwargs['interp_method'] if 'interp_method' in kwargs else 'nearest'
    fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else -999.0

    # at first find the maximums Doppler velocity fo all chrips
    n_chirps = len(spec)
    N_Dbins_max = np.max([len(spec[ic]['vel']) for ic in range(n_chirps)])
    V_Dbins_max = np.max([spec[ic]['vel'][-1] for ic in range(n_chirps)])
    I_Dbins_max = np.argmax([spec[ic]['vel'][-1] for ic in range(n_chirps)])

    # most likely the first chirp has the Nyquist velocity
    varstack = spec[0]['var'].copy()
    for ic in range(n_chirps):
        if ic != I_Dbins_max:
            f = interp1d(spec[ic]['vel'], spec[ic]['var'], axis=2, kind=method, bounds_error=False, fill_value=fill_value)
            spectra_interp = f(np.linspace(-V_Dbins_max, V_Dbins_max, N_Dbins_max))
            varstack = np.concatenate((varstack, spectra_interp), axis=1)

    new_mask = np.ma.getmask(np.ma.masked_less_equal(varstack, 0.0))
    new_spec = h.put_in_container(varstack, spec[0], name='VSpec', mask=new_mask)
    new_spec['rg'] = np.hstack([spec[ic]['rg'] for ic in range(n_chirps)])

    Plot.print_elapsed_time(t0, f'Interpolation of 3rd chirp to {N_Dbins_max} Doppler bins, elapsed time = ')

    return new_spec


def load_radar_data(larda, begin_dt, end_dt, **kwargs):
    """ This routine loads the radar spectra from an RPG cloud radar and caluclates the radar moments.

    Args:
        - larda (larda object) : the class for reading NetCDF files
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

    rm_prcp_ghst = kwargs['rm_precip_ghost'] if 'rm_precip_ghost' in kwargs else False
    rm_crtn_ghst = kwargs['rm_curtain_ghost'] if 'rm_curtain_ghost' in kwargs else False
    dspckl = kwargs['do_despeckle'] if 'do_despeckle' in kwargs else False
    dspckl3d = kwargs['do_despeckle3d'] if 'do_despeckle3d' in kwargs else 95.
    est_noise = kwargs['estimate_noise'] if 'estimate_noise' in kwargs else False
    NF = kwargs['noise_factor'] if 'noise_factor' in kwargs else 6.0
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else True
    fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else -999.0

    t0 = time.time()
    radar_spectra = build_extended_container(larda, 'VSpec', begin_dt, end_dt,
                                             rm_precip_ghost=rm_prcp_ghst, do_despeckle3d=dspckl3d,
                                             estimate_noise=est_noise, noise_factor=NF)

    radar_moments = spectra2moments(radar_spectra, larda.connectors['LIMRAD94'].system_info['params'],
                                    despeckle=dspckl, main_peak=main_peak, filter_ghost_C1=rm_crtn_ghst)

    # radar_spectra = remove_noise_from_spectra(radar_spectra)

    # replace NaN values with fill_value
    for ic in range(len(radar_spectra)):
        radar_spectra[ic]['var'][np.isnan(radar_spectra[ic]['var'])] = fill_value

    Plot.print_elapsed_time(t0, f'Reading spectra + moment calculation, elapsed time = ')
    return {'spectra': radar_spectra, 'moments': radar_moments}


def load_lidar_data(larda, var_list, begin_dt, end_dt, **kwargs):
    t0 = time.time()

    lidar_var = {var: larda.read("POLLY", var, [begin_dt, end_dt], [0, 12000]) for var in var_list}

    # remove multiple scattering effects caused by large field of view
    if 'msf' in kwargs and kwargs['msf']:
        assert len(lidar_var) == 2, 'multiple scattering filter needs both attbsc1064 and depol'

        status_flags = Multiscatter.get_filter_mask(lidar_var['attbsc1064'], lidar_var['depol'], despeckle=True)

        logger.info(f'Number of non-liquid pixel = {np.sum(status_flags == 1)}')
        logger.info(f'Number of liquid pixel     = {np.sum(status_flags == 2)}')
        logger.info(f'Number of attenuated pixel = {np.sum(status_flags == 3)}')
        lidar_var['depol']['var'], lidar_var['depol']['mask'] = Multiscatter.apply_filter_mask(status_flags, lidar_var['depol'],
                                                                                               mask_attenuated=True, mask_liq=True)
        lidar_var['attbsc1064']['var'], lidar_var['attbsc1064']['mask'] = Multiscatter.apply_filter_mask(status_flags, lidar_var['attbsc1064'],
                                                                                                         mask_attenuated=True)
        lidar_var['attbsc1064'].update({'flags': status_flags})
        lidar_var['depol'].update({'flags': status_flags})

    for var in var_list:
        lidar_var[var]['var'][np.isnan(lidar_var[var]['var'])] = 0.0
        mask1 = lidar_var[var]['var'] <= 0.0
        mask2 = lidar_var[var]['var'] > 0.0
        lidar_var[var]['mask'][mask1] = True
        lidar_var[var]['mask'][mask2] = False

    Plot.print_elapsed_time(t0, f'Read lidar data, elapsed time = ')
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
        bsc_lims = dimensions['target_info']['attbsc1064']['var_lims']

        # make prediction larda container
        attbsc1064_pred = make_container_from_prediction(pred[:, 0], dimensions['list_ts'], dimensions['list_rg'],
                                                         param_info['attbsc1064'], dimensions['ts_radar'], dimensions['rg_radar'])

        if dimensions['target_info']['attbsc1064']['scaling'] == 'normalize':
            attbsc1064_pred['var'] = attbsc1064_pred['var'] * (bsc_lims[1] - bsc_lims[0]) + bsc_lims[0]

        if dimensions['target_info']['attbsc1064']['var_converter'] == 'log':
            attbsc1064_pred['var'] = np.power(10., attbsc1064_pred['var'])

        attbsc1064_pred['var_lims'] = bsc_lims

        predictions.update({'attbsc1064_pred': attbsc1064_pred})

    if 'depol' in pred_list:
        dpl_lims = dimensions['target_info']['depol']['var_lims']
        depol_pred = make_container_from_prediction(pred[:, 1], dimensions['list_ts'], dimensions['list_rg'],
                                                          param_info['depol'], dimensions['ts_radar'], dimensions['rg_radar'])

        if dimensions['target_info']['depol']['scaling'] == 'normalize':
            depol_pred['var'] = depol_pred['var'] * (dpl_lims[1] - dpl_lims[0]) + dpl_lims[0]

        if dimensions['target_info']['depol']['var_converter'] == 'ldr2cdr':
            depol_pred['var'] = cdr2ldr(depol_pred['var'])

        depol_pred['var_lims'] = dpl_lims

        predictions.update({'depol_pred': depol_pred})

    return predictions

#
