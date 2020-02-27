"""
This module contains routines for loading and preprocessing cloud radar and lidar data.

"""

import sys
from copy import deepcopy

import datetime
import toml
import libVoodoo.Plot as Plot
import numpy as np
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import time
from numba import jit
from scipy import signal
from scipy.interpolate import interp1d
from itertools import product

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.2.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def load_case_list(path, case_name):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case'][case_name]


def scaling(data, strat='none', **kwargs):
    if 'var_lims' in kwargs:
        assert len(kwargs['var_lims']) == 2, 'Error while loading data, wrong number of var_lims set for scaling!'
    if strat == 'normalize':
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


# @jit(nopython=True, fastmath=True)
def norm(x, mini, maxi):
    x[x < mini] = mini
    x[x > maxi] = maxi
    return (x - mini) / max(1.e-15, maxi - mini)


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


def load_trainingset(spectra, classes, masked, **feature_info):
    t0 = time.time()

    n_time, n_range, n_Dbins = spectra['var'].shape
    spectra_unit = spectra['var_unit']
    spectra_3d = spectra['var'].astype('float32')
    ts_list = spectra['ts']
    rg_list = spectra['rg']
    vel_list = spectra['vel']

    spec_params = feature_info['VSpec']
    cwt_params = feature_info['cwt']

    assert len(cwt_params['scales']) == 3, 'The list of scaling parameters 3 values!'
    n_cwt_scales = int(cwt_params['scales'][2])
    scales = np.linspace(cwt_params['scales'][0], cwt_params['scales'][1], n_cwt_scales)

    quick_check = False
    if quick_check:
        ZE = np.sum(spectra_3d, axis=2)
        ZE = h.put_in_container(ZE, cwt_params['SLv'])#, **kwargs)
        ZE['dimlabel'] = ['time', 'range']
        ZE['name'] = ZE['name'][0]
        ZE['joints'] = ZE['joints'][0]
        ZE['rg_unit'] = ZE['rg_unit'][0]
        ZE['colormap'] = ZE['colormap'][0]
        #ZE['paraminfo'] = dict(ZE['paraminfo'][0])
        ZE['system'] = ZE['system'][0]
        ZE['ts'] = np.squeeze(ZE['ts'])
        ZE['rg'] = np.squeeze(ZE['rg'])
        #ZE['var_lims'] = [ZE['var'].min(), ZE['var'].max()]
        ZE['var_lims'] = [-60, 20]
        ZE['var_unit'] = 'dBZ'
        ZE['mask'] = masked

        fig, _ = tr.plot_timeheight(ZE, var_converter='lin2z', title='bla inside wavelett')#, **plot_settings)
        Plot.save_figure(fig, name=f'limrad_cwtest.png', dpi=200)

    spectra_lims = np.array(spec_params['var_lims'])
    # convert to logarithmic units
    if 'var_converter' in spec_params and 'lin2z' in spec_params['var_converter']:
        spectra_unit = 'dBZ'
        spectra_3d = h.get_converter_array('lin2z')[0](spectra_3d)
        spectra_lims = h.get_converter_array('lin2z')[0](spec_params['var_lims'])


    cnt = 0
    cwt_list = []
    target_labels = []
    for iT in range(n_time):
        print(f'Timestep cwt added :: {iT + 1:5d} of {n_time}')
        for iR in range(n_range):
            if masked[iT, iR]: continue
            spcij_scaled = scaling(spectra_3d[iT, iR, :], strat=spec_params['scaling'], var_lims=spectra_lims)
            cwtmatr = signal.cwt(spcij_scaled, signal.ricker, scales)

            if 'chsgn' in cwt_params['var_converter']: cwtmatr *= -1.0
            if 'normalize' in cwt_params['scaling']: cwtmatr = scaling(cwtmatr, strat='normalize', var_lims=cwt_params['var_lims'])

            cwt_list.append(np.reshape(cwtmatr, (n_Dbins, n_cwt_scales, 1)))
            # one hot encodeing
            if classes[iT, iR]:
                target_labels.append([0, 1])    # "contains liquid droplets"
            else:
                target_labels.append([1, 0])    # "non-droplet class"
            cnt += 1

            if 'plot' in cwt_params and cwt_params['plot']:
                velocity_lims = [-6, 6]

                # show spectra, normalized spectra and wavlet transformation
                fig, ax = Plot.spectra_wavelettransform(vel_list[0], spcij_scaled, cwtmatr,
                                                        ts=ts_list[0, iT],
                                                        rg=rg_list[0, iR],
                                                        colormap='cloudnet_jet',
                                                        x_lims=velocity_lims,
                                                        v_lims=cwt_params['var_lims'],
                                                        scales=scales,
                                                        hydroclass=classes[iT, iR],
                                                        fig_size=[7, 4]
                                                        )
                # fig, (top_ax, bottom_left_ax, bottom_right_ax) = Plot.spectra_wavelettransform2(vel_list, spcij_scaled, cwt_params['scales'])
                Plot.save_figure(fig, name=f'limrad_cwt_{str(cnt).zfill(4)}_iT-iR_{str(iT).zfill(4)}-{str(iR).zfill(4)}.png', dpi=300)

    Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    return np.array(cwt_list, dtype=np.float32), np.array(target_labels, dtype=np.float32)

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

    time_span = [begin_dt, end_dt]

    """old reader
    from larda.pyLARDA.spec2mom_limrad94 import spectra2moments, build_extended_container

    radar_spectra = build_extended_container(larda, 'VSpec', time_span,
                                             rm_precip_ghost=rm_prcp_ghst, do_despeckle3d=dspckl3d, estimate_noise=est_noise, noise_factor=NF)

    radar_moments = spectra2moments(radar_spectra, larda.connectors['LIMRAD94'].system_info['params'],
                                    despeckle=dspckl, main_peak=main_peak, filter_ghost_C1=rm_crtn_ghst)
    """

    from larda.pyLARDA.SpectraProcessing import spectra2moments, load_spectra_rpgfmcw94
    radar_spectra = load_spectra_rpgfmcw94(larda, time_span, rm_precip_ghost=rm_prcp_ghst, do_despeckle3d=dspckl3d, estimate_noise=est_noise, noise_factor=NF)
    radar_moments = spectra2moments(radar_spectra, larda.connectors['LIMRAD94'].system_info['params'], despeckle=dspckl, main_peak=main_peak,
                                    filter_ghost_C1=rm_crtn_ghst)

    # radar_spectra = remove_noise_from_spectra(radar_spectra)

    # replace NaN values with fill_value
    #for ic in range(len(radar_spectra)):
    #    radar_spectra[ic]['var'][np.isnan(radar_spectra[ic]['var'])] = fill_value

    Plot.print_elapsed_time(t0, f'Reading spectra + moment calculation, elapsed time = ')
    return {'spectra': radar_spectra, 'moments': radar_moments}

