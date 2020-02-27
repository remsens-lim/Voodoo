#!/usr/bin/env python3
"""
Short description:
    Creating a .mat file containing radar spectra, lidar variables and target classifications.
"""

import datetime
import sys

sys.path.append('../larda/')
sys.path.append('.')

import logging
import numpy as np
import time

from scipy.io import savemat

import scipy.interpolate

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpectraProcessing as sp
from larda.pyLARDA.Transformations import interpolate2d

import voodoo.libVoodoo.Loader_v2 as Loader
import voodoo.libVoodoo.Plot   as Plot

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def load_data(system, var_list):
    data = {}
    for i, var in enumerate(var_list):
        var_info = larda.read(system, var, TIME_SPAN_, [0, 'max'])
        var_info['n_ts'] = var_info['ts'].size
        var_info['n_rg'] = var_info['rg'].size
        data.update({var: var_info})
    return data

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
    print('var min {}'.format(data['var'][~data['mask']].min()))
    method = kwargs['method'] if 'method' in kwargs else 'rectbivar'

    new_time = data['ts'] if not 'new_time' in kwargs else kwargs['new_time']
    new_range = data['rg'] if not 'new_range' in kwargs else kwargs['new_range']

    for iBin in range(data['vel'].size):
        var, mask = all_var_bins[:, :, iBin], all_mask_bins[:, :, iBin]
        if method == 'rectbivar':
            kx, ky = 1, 1
            interp_var = scipy.interpolate.RectBivariateSpline(data['ts'], data['rg'], var, kx=kx, ky=ky)
            interp_mask = scipy.interpolate.RectBivariateSpline(data['ts'], data['rg'], mask.astype(np.float), kx=kx, ky=ky)
            args_to_pass = {"grid":True}
        elif method == 'linear':
            interp_var = scipy.interpolate.interp2d(data['rg'], data['ts'], var, fill_value=-999.0)
            interp_mask = scipy.interpolate.interp2d(data['rg'], data['ts'], mask.astype(np.float))
            args_to_pass = {}
        elif method == 'nearest':
            points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
            interp_var = scipy.interpolate.NearestNDInterpolator(points, var.flatten())
            interp_mask = scipy.interpolate.NearestNDInterpolator(points, (mask.flatten()).astype(np.float))

        if not method == "nearest":
            new_var = interp_var(new_time, new_range, **args_to_pass)
            new_mask = interp_mask(new_time, new_range, **args_to_pass)
        else:
            new_points = np.array(list(zip(np.repeat(new_time, len(new_range)), np.tile(new_range, len(new_time)))))
            new_var = interp_var(new_points).reshape((len(new_time), len(new_range)))
            new_mask = interp_mask(new_points).reshape((len(new_time), len(new_range)))

        new_mask[new_mask > mask_thres] = 1
        new_mask[new_mask < mask_thres] = 0

        var_interp[:, :, iBin] = new_var if method in ['nearest', 'rectbivar'] else np.transpose(new_var)
        mask_interp[:, :, iBin] = new_mask if method in ['nearest', 'rectbivar'] else np.transpose(new_mask)
        print('iBin = ', iBin)

    # deepcopy to keep data immutable
    interp_data = {**data}

    interp_data['ts'] = new_time
    interp_data['rg'] = new_range
    interp_data['var'] = var_interp
    interp_data['mask'] = mask_interp
    print("interpolated shape: time {} range {} var {} mask {}".format(
        new_time.shape, new_range.shape, var_interp.shape, mask_interp.shape))

    return interp_data
########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
if __name__ == '__main__':

    CASE_LIST =  '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'

    spec_settings = {

        'despeckle2D': True,  # 2D convolution (5x5 window), removes single non-zero values,

        'main_peak': True,  #

        'ghost_echo_1': True,  # reduces the domain (Nyquist velocitdy) by ± 2.5 [m/s], when signal > 0 [dBZ] within 200m above antenna

        'ghost_echo_2': True,  #

    }

    PATH = '/home/sdig/code/larda3/voodoo/plots/spectra2moments_v2/'
    NCPATH = '/home/sdig/code/larda3/voodoo/nc-files/spectra/'

    start_time = time.time()

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu')

    # load case information
    #case_string = '20190801-01'
    #case_string = '20190410-02'
    case_string = '20190904-03'
    case = Loader.load_case_list(CASE_LIST, case_string)

    TIME_SPAN_ = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
    begin_dt, end_dt = TIME_SPAN_

    h.change_dir(NCPATH)

    #
    # load & store cloudnet information
    #
    cnpy94_model = load_data('CLOUDNETpy94', ['T', 'P'])
    savemat(f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_cloudnetpy94_model_T.mat', cnpy94_model.pop('T'))
    savemat(f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_cloudnetpy94_model_P.mat', cnpy94_model.pop('P'))
    print(f'save :: {begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_cloudnetpy94_model_T/P')

    """
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
    cnpy94_class = load_data('CLOUDNETpy94', ['CLASS', 'detection_status'])
    ts_cnpy94, rg_cnpy94 = cnpy94_class['CLASS']['ts'], cnpy94_class['CLASS']['rg']
    savemat(f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_cloudnetpy94_class.mat', cnpy94_class.pop('CLASS'))
    savemat(f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_cloudnetpy94_status.mat', cnpy94_class.pop('detection_status'))
    print(f'save :: {begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_cloudnetpy94_class/status')

    #
    # load & store pollynet information
    #
    pollynet_class = load_data('POLLYNET', ['CLASS_v2'])
    FILE_NAME_3 = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_pollynet_class.mat'
    savemat(FILE_NAME_3, pollynet_class.pop('CLASS_v2'))
    print(f'save :: {FILE_NAME_3}')

    #
    # load & store radar information
    #
    limrad94_Zspec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_, **spec_settings)

    # interpolate time dimension of spectra
    limrad94_Zspec['VHSpec']['var'] = Loader.replace_fill_value(limrad94_Zspec['VHSpec']['var'], limrad94_Zspec['SLv']['var'])

    limrad94_Zspec['SLv']    = interpolate2d(limrad94_Zspec['SLv'], new_time=ts_cnpy94, new_range=rg_cnpy94, method='rectbivar')
    limrad94_Zspec['VHSpec'] = interpolate3d(limrad94_Zspec['VHSpec'], new_time=ts_cnpy94, new_range=rg_cnpy94, method='rectbivar')

    FILE_NAME_1 = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_limrad94_spectra.mat'
    savemat(f'{FILE_NAME_1}', limrad94_Zspec.pop('VHSpec'))  # store spectra separately from other arrays
    print(f'save :: {FILE_NAME_1}')

    #limrad94_Zspec
    [limrad94_Zspec.pop(ivar) for ivar in ['ge1_mask', 'ge2_mask', 'dspkl_mask', 'edges', 'var_max', 'Vnoise', 'variance', 'mean', 'thresh']]
    limrad94_Zspec['n_ts'] = ts_cnpy94.size
    limrad94_Zspec['n_rg'] = rg_cnpy94.size

    FILE_NAME_n = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_limrad94_spectra_SLv.mat'
    savemat(f'{FILE_NAME_n}', limrad94_Zspec.pop('SLv'))  # store spectra separately from other arrays
    print(f'save :: {FILE_NAME_n}')

    FILE_NAME_2 = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_limrad94_spectra_extra.mat'
    savemat(FILE_NAME_2, limrad94_Zspec)
    print(f'save :: {FILE_NAME_2}')



    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))
