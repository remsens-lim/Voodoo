#!/usr/bin/env python3
"""
Short description:
    This module will correct the linear volume depolarization of Polly XT polarization lidar system of TROPOS for multiple scattering effects in liquid cloud
    layers. The module was developed to correct the data for the application to an artificial neural network.


Long description:
    Patrics Description:  With increasing penetration depth into
    the cloud the ratio of photons scattered back from single-scattering events to the photons scattered back
    after multiple scattering events decreases. I.e., the deeper you look into a cloud the higher is the fraction
    of multiply scattered light. And because scattering events at angles deviating from exactly 180°
    (backscattering) cause strong depolarization, the measured depolarization increases with increasing
    penetration depth (since multiple-scattering events by definition involve scattering processes at angles
    not equal 180°)

    I cannot give you a definite number at which distance from cloud base the multiple-scattering dominates the
    signal and causes significant values of MS-related depolarization which will cause the miss-classification
    you show in the slides you sent. Characteristic is the rather linear increase of the depolarization ratio
    with penetration depth into the cloud.

    How it works:
        In Cloudnet they are using a threshold to identify a liquid cloud besides the depolarization
        characterization. In there, a liquid bit is only set if the signal decreases by at least a factor
        of 10 within 250 m of vertical distance.  Starting to look for
        new liquid cloud layers only at a defined distance (e.g., at 500 m) above the lower liquid layer base.
"""

import os
import sys
import logging
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# just needed to find pyLARDA from this location
sys.path.append('../../larda/')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import pyLARDA.spec2mom_limrad94 as s2m

import libVoodoo.Plot as Plot

__author__      = "Willi Schimmel"
__copyright__   = "Copyright 2019, The Voodoo Project"
__credits__     = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__     = "MIT"
__version__     = "0.0.1"
__maintainer__  = "Willi Schimmel"
__email__       = "willi.schimmel@uni-leipzig.de"
__status__      = "Prototype"


def apply_filter(bsc, depol, **kwargs):
    """
    Patric:  With increasing penetration depth into
    the cloud the ratio of photons scattered back from single-scattering events to the photons scattered back
    after multiple scattering events decreases. I.e., the deeper you look into a cloud the higher is the fraction
    of multiply scattered light. And because scattering events at angles deviating from exactly 180°
    (backscattering) cause strong depolarization, the measured depolarization increases with increasing
    penetration depth (since multiple-scattering events by definition involve scattering processes at angles
    not equal 180°)

    I cannot give you a definite number at which distance from cloud base the multiple-scattering dominates the
    signal and causes significant values of MS-related depolarization which will cause the miss-classification
    you show in the slides you sent. Characteristic is the rather linear increase of the depolarization ratio
    with penetration depth into the cloud.

    How it works:
        In Cloudnet they are using a threshold to identify a liquid cloud besides the depolarization
        characterization. In there, a liquid bit is only set if the signal decreases by at least a factor
        of 10 within 250 m of vertical distance.  Starting to look for
        new liquid cloud layers only at a defined distance (e.g., at 500 m) above the lower liquid layer base.


    Args:
        bsc (dict): container of backscatter values
        depol (dict): container of depolarization values

    Keyword Args:
        **bsc_thresh (list, float): list containing [min_bsc_thresh, max_bsc_thresh], values above and below are set to the respective value; default: [1.e-6, 1.e-3]
        **depol_thresh (list, float): list containing [min_depol_thresh, max_depol_thresh], values above and below are set to the respective value; default: [0.0, 0.5]
        **thresh_fac (float): minimum decrease of the signal in powers of 10; default: 10
        **n_smoothing_ts (integer): number of time steps for smoothing; default: 0
        **n_smoothing_rg (integer): number of range bins for smoothing; default: 8
        **n_times_smoothing (integer): controls how many times the smoothing is applied; default: 1
        **n_bins (integer): a liquid bit is only set if the signal decreases by at least a factor of 10 within N meter of vertical distance, where 7.5*n_bins = N
        **plot_profiles(bool): if True, save all profile analysis plots; default: False
        **despeckle (bool): if True, mask single pixels; default: True

    Returns:
        bsc_out, depol_out (2D - numpy.arrays): containing corrected backscatter and depolarization matrices, NOTE: The mask of the input container is updated!
    """

    min_bsc_thresh = kwargs['bsc_thresh'][0] if 'bsc_thresh' in kwargs else 1.e-6
    max_bsc_thresh = kwargs['bsc_thresh'][1] if 'bsc_thresh' in kwargs else 1.e-3
    min_depol_thresh = kwargs['depol_thresh'][0] if 'depol_thresh' in kwargs else 0.0
    max_depol_thresh = kwargs['depol_thresh'][1] if 'depol_thresh' in kwargs else 0.5
    n_smoothing_ts = kwargs['n_smoothing_ts'] if 'n_smoothing_ts' in kwargs else 0
    n_smoothing_rg = kwargs['n_smoothing_rg'] if 'n_smoothing_rg' in kwargs else 8
    n_times_smoothing = kwargs['n_times_smoothing'] if 'n_times_smoothing' in kwargs else 1
    n_bins_250m = kwargs['n_bins'] if 'n_bins' in kwargs else 35  # equivivalent to 262.5m
    factor_thresh = kwargs['thresh_fac'] if 'thresh_fac' in kwargs else 1.e-1
    plot_profiles = kwargs['plot_profiles'] if 'plot_profiles' in kwargs else False
    do_despeckle = kwargs['despeckle'] if 'despeckle' in kwargs else True

    bsc_var_lims_orig, depol_var_lims_orig = bsc['var_lims'], depol['var_lims']

    bsc_cpy, depol_cpy = bsc['var'].copy(), depol['var'].copy()
    bsc_out, depol_out = bsc['var'].copy(), depol['var'].copy()
    n_rg = len(depol['rg'])

    # set all nan to 0 for smoothing
    bsc_cpy[np.isnan(bsc_cpy)] = min_bsc_thresh  # max invalid values in bsc and depol
    depol_cpy[np.isnan(depol_cpy)] = min_depol_thresh  # max invalid values in bsc and depol
    bsc_cpy[bsc_cpy < min_bsc_thresh] = min_bsc_thresh  # remove values beyond minimum and maximum bsc thresh
    bsc_cpy[bsc_cpy > max_bsc_thresh] = max_bsc_thresh  # set maximum bsc value to threshold
    depol_cpy[bsc_cpy < min_bsc_thresh] = min_depol_thresh  # remove values beyond minimum and maximum bsc thresh
    depol_cpy[depol_cpy > max_depol_thresh] = max_depol_thresh  # set maximum depol value to thresholdh
    # update mask for mask non-signal values
    bsc_mask_cpy = np.logical_or(bsc_cpy <= 0., bsc['mask'])
    depol_mask_cpy = np.logical_or(bsc_cpy <= 0., bsc['mask'])

    if n_smoothing_rg > 0:
        for i_smth in range(n_times_smoothing):
            for i in range(len(bsc['ts'])):
                bsc_cpy[i, :] = tr.smooth(bsc_cpy[i, :], n_smoothing_rg)
                depol_cpy[i, :] = tr.smooth(depol_cpy[i, :], n_smoothing_rg)
    if n_smoothing_ts > 0:
        for i_smth in range(n_times_smoothing):
            for i in range(len(bsc['rg'])):
                bsc_cpy[:, i] = tr.smooth(bsc_cpy[:, i], n_smoothing_ts)
                depol_cpy[:, i] = tr.smooth(depol_cpy[:, i], n_smoothing_ts)

    if do_despeckle:
        old_mask = bsc_mask_cpy.copy()
        speckle_mask = s2m.despeckle(old_mask * 1, 20)
        bsc_mask_cpy[speckle_mask == 1] = True
        depol_mask_cpy[speckle_mask == 1] = True

    bsc_out[depol_mask_cpy] = np.nan
    depol_out[depol_mask_cpy] = np.nan

    # create directory for plots
    if plot_profiles:
        name = os.path.dirname(__file__) + f'/plots/multiscattering/{begin_dt:%Y%m%d_%H%M%S}/'
        h.change_dir(name)

    for iT, v in enumerate(depol['ts']):

        attbsc_corrected = bsc_cpy[iT, :].copy()
        depol_corrected = depol_cpy[iT, :].copy()

        # normalizing backscatter and depol between provided boundaries
        d_bsc_depol = []
        for ibsc, idepol in zip(attbsc_corrected, depol_corrected):
            norm_bsc = ((ibsc - min_bsc_thresh) / (max_bsc_thresh - min_bsc_thresh))
            norm_depol = ((idepol - min_depol_thresh) / (max_depol_thresh - min_depol_thresh))
            d_bsc_depol.append(norm_bsc - norm_depol)

        # smooth the distance between bsc and depol
        d_bsc_depol = tr.smooth(np.array(d_bsc_depol), n_smoothing_rg)
        # correct the boundaries (necessary due to smoothing)
        N = n_smoothing_rg // 2 + 1
        d_bsc_depol[-N:] = d_bsc_depol[-N]
        d_bsc_depol[:N] = d_bsc_depol[N]

        bsc_diff = bsc_cpy[iT, n_bins_250m:] / bsc_cpy[iT, :-n_bins_250m]
        thresh_fac_fcn = np.full(len(bsc_diff), fill_value=factor_thresh)

        # find the intersection of bsc difference over 250m and the liquid detection threshold factor
        inters_list = np.argwhere(np.diff(np.sign(bsc_diff - thresh_fac_fcn))).flatten() + n_bins_250m
        idx_scl_end_2 = inters_list[::2]  # set the elements with even index to the index of the SCL top

        idx_scl_start = []
        idx_scl_end = []
        idx_bsc_max = 0
        if len(idx_scl_end_2) > 0:
            for idx1 in idx_scl_end_2:
                base = idx1 - n_bins_250m if idx1 - n_bins_250m > 0 else 0
                # idx0 = np.argmax(d_bsc_depol_gradient[base:idx1-5]) + base
                idx0 = np.argmax(d_bsc_depol[base:idx1 - 5]) + base

                if n_bins_250m + 5 >= idx1 - idx0 > 5:
                    idx_scl_start.append(idx0)
                    idx_scl_end.append(idx1)

                    n_rg_scl_depol = idx1 - idx0 if n_rg > idx1 > 0 else n_rg - idx1
                    new_depol = np.minimum(np.linspace(1.e-4, 5.e-4, n_rg_scl_depol), depol_corrected[idx0:idx1])

                    # correct depolarization for multiple scattering
                    depol_corrected[idx0:idx1] = new_depol
                    depol_out[iT, idx0:idx1] = new_depol

                    # if signal is attenuated, set range bins above liquid layer to nan
                    if np.mean(attbsc_corrected[idx0:idx1]) > 1.e-4:
                        idx_bsc_max = idx1
                        depol_corrected[idx_bsc_max:] = np.nan
                        attbsc_corrected[idx_bsc_max:] = np.nan
                        depol_out[iT, idx_bsc_max:] = np.nan
                        bsc_out[iT, idx_bsc_max:] = np.nan
                        continue  # skip the rest of this profile

        if idx_bsc_max == 0:
            # remove signal 262,5m ( =35 range gates) above maximum backscatter value
            # one could try to develop a function depending on max bsc to calculate the number of bins above
            idx_bsc_max = np.argmax(attbsc_corrected) + n_bins_250m
            depol_corrected[idx_bsc_max:] = np.nan
            attbsc_corrected[idx_bsc_max:] = np.nan
            depol_out[iT, idx_bsc_max:] = np.nan
            bsc_out[iT, idx_bsc_max:] = np.nan

        if plot_profiles:
            fltr = {'iT': iT,
                    'depol_filter':   depol_out[iT, :],
                    'idx_scl_end':    idx_scl_end,
                    'idx_scl_start':  idx_scl_start,
                    'd_bsc_depol':    d_bsc_depol,
                    'thresh_fac_fcn': thresh_fac_fcn,
                    'idx_bsc_max':    idx_bsc_max,
                    'min_bsc_thresh': min_bsc_thresh,
                    'bsc_diff':       bsc_diff,
                    'n_bins_250m':    n_bins_250m}

            kwargs = {'font_size': 14, 'font_weight': 'bold'}

            fig, ax = Plot.lidar_profiles(bsc, depol, fltr, **kwargs)
            plt.tight_layout()
            Plot.save_figure(fig, name=f'polly_profile_gradient_diff_{str(iT).zfill(4)}.png', dpi=100)

    bsc['var_lims'] = bsc_var_lims_orig
    depol['var_lims'] = depol_var_lims_orig

    return bsc_out, depol_out


########################################################################################################################
########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
########################################################################################################################
########################################################################################################################

if __name__ == '__main__':

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=False)

    # begin_dt = datetime.datetime(2019, 8, 1, 6, 50)
    ##end_dt = datetime.datetime(2019, 8, 1, 8, 1)
    #
    # begin_dt = datetime.datetime(2019, 8, 1, 0, 0, 5)
    # end_dt = datetime.datetime(2019, 8, 1, 23, 59, 55)
    #
    # begin_dt = datetime.datetime(2019, 6, 15, 9, 0, 5)
    # end_dt = datetime.datetime(2019, 6, 15, 20, 59, 55)
    #
    # begin_dt = datetime.datetime(2019, 8, 30, 3, 0, 5)
    # end_dt = datetime.datetime(2019, 8, 30, 23, 59, 55)

    # begin_dt = datetime.datetime(2019, 8, 13, 21,42, 10)
    # end_dt = datetime.datetime(2019, 8, 13, 23, 0)

    # begin_dt = datetime.datetime(2019, 3, 9, 4, 15, 5)
    # end_dt = datetime.datetime(2019, 3, 9, 4, 45, 55)

    # begin_dt = datetime.datetime(2019, 3, 9, 3, 0, 5)
    # end_dt = datetime.datetime(2019, 3, 9, 9, 29, 55)

    #begin_dt = datetime.datetime(2019, 1, 10, 0, 0, 5)
    #end_dt = datetime.datetime(2019, 1, 10, 23, 59, 55)

    begin_dt = datetime.datetime(2019, 3, 9, 0, 0, 5)
    end_dt = datetime.datetime(2019, 3, 9, 17, 59, 55)

    plot_range = [0, 12000]

    plot_profiles    = True
    plot_timeheight  = True
    plot_trainingset = True

    fig_sz = [9, 3]

    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
    name = VOODOO_PATH + f'/plots/multiscattering/{begin_dt:%Y%m%d_%H%M%S}/'
    h.change_dir(name)

    # exit()
    LIMRAD94_ZE  = larda.read("LIMRAD94", "Ze", [begin_dt, end_dt], plot_range)
    LIMRAD94_VEL = larda.read("LIMRAD94", "VEL", [begin_dt, end_dt], plot_range)
    attbsc_532   = larda.read("POLLY", "attbsc532", [begin_dt, end_dt], plot_range)
    attbsc_1064  = larda.read("POLLY", "attbsc1064", [begin_dt, end_dt], plot_range)
    voldepol_532 = larda.read("POLLY", "depol", [begin_dt, end_dt], plot_range)
    T            = larda.read("CLOUDNET_LIMRAD", "T", [begin_dt, end_dt], plot_range)

    def toC(datalist):
        return datalist[0]['var'] - 273.15, datalist[0]['mask']


    T = pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"})
    contour = {'data': T, 'levels': np.arange(-40, 16, 10)}

    # copy and convert from bool to 0 and 1, remove a pixel if more than 20 neighbours are present (5x5 grid)

    new_mask = s2m.despeckle(LIMRAD94_ZE['mask'].copy() * 1, 20)
    LIMRAD94_ZE['mask'][new_mask == 1] = True

    if plot_profiles or plot_timeheight:
        attbsc_1064['var_lims'] = [1e-7, 1e-3]
        fig, ax = pyLARDA.Transformations.plot_timeheight(attbsc_1064, range_interval=plot_range, rg_converter=True,
                                                          fig_size=fig_sz, z_converter="log", contour=contour)
        fig.savefig(f'polly_bsc1064_original_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png', dpi=300)
        print(f'polly_bsc1064_original_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png')

        fig, _ = pyLARDA.Transformations.plot_timeheight(voldepol_532, range_interval=plot_range, contour=contour,
                                                         rg_converter=True, fig_size=fig_sz)
        fig.savefig(f'polly_depol_original_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}.png', dpi=300)
        print(f'polly_depol_original_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png')

        LIMRAD94_ZE['var_lims'] = [-50, 20]
        LIMRAD94_ZE['colormap'] = 'jet'
        LIMRAD94_ZE['var_unit'] = 'dBZ'
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ZE,
                                                          fig_size=fig_sz, range_interval=plot_range,
                                                          z_converter='lin2z', rg_converter=True, contour=contour,
                                                          # contour=contour
                                                          )

        fig.tight_layout()
        fig_name = f'limrad_ZE_original_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

        LIMRAD94_VEL['var_lims'] = [-4, 2]
        fig, axVel = pyLARDA.Transformations.plot_timeheight(LIMRAD94_VEL, fig_size=fig_sz, range_interval=plot_range,
                                                             title=False, rg_converter=True, contour=contour)
        fig_name = f'limrad_VEL_original_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

    ####
    ####
    ###
    ##
    #
    #
    attbsc_1064['var'], voldepol_532['var'] = apply_filter(attbsc_1064, voldepol_532, plot_profiles=plot_profiles)
    attbsc_1064['var'][np.isnan(attbsc_1064['var'])] = -1.0
    voldepol_532['var'][np.isnan(voldepol_532['var'])] = 0.0
    mask1 = voldepol_532['var'] <= 0.0
    mask2 = voldepol_532['var'] > 0.0
    voldepol_532['mask'][mask1] = True
    voldepol_532['mask'][mask2] = False
    attbsc_1064['mask'][mask1] = True
    attbsc_1064['mask'][mask2] = False

    attbsc_1064['var'] = np.ma.masked_where(attbsc_1064['mask'], attbsc_1064['var'])
    attbsc_1064['var'] = np.ma.masked_invalid(attbsc_1064['var'])

    voldepol_532['var'] = np.ma.masked_where(voldepol_532['mask'], voldepol_532['var'])
    voldepol_532['var'] = np.ma.masked_invalid(voldepol_532['var'])

    """
        ___  _    ____ ___    ____ ____ ____ ____ ____ ____ ___ _ ____ _  _ 
        |__] |    |  |  |     |    |  | |__/ |__/ |___ |     |  | |  | |\ | 
        |    |___ |__|  |     |___ |__| |  \ |  \ |___ |___  |  | |__| | \| 
                                                                    
    """

    if plot_timeheight:
        #
        dt_list = [datetime.datetime.utcfromtimestamp(time) for time in attbsc_1064['ts']]
        # this is the last valid index
        attbsc_1064['colormap'] = 'cloudnet_jet'
        attbsc_1064['var_lims'] = [1e-7, 1e-3]
        fig, _ = pyLARDA.Transformations.plot_timeheight(attbsc_1064, rg_converter=True, contour=contour,
                                                          range_interval=plot_range, fig_size=fig_sz, z_converter="log")
        # ax.plot(dt_list, attbsc_1064['rg'][bsc_max], color='k', linestyle='--', label=r'thresh1')
        name = f'polly_bsc1064_corrected_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.savefig(name, dpi=300)
        print(name)
        #
        voldepol_532['colormap'] = 'cloudnet_jet'
        voldepol_532['var_lims'] = [0, 0.3]
        fig, _ = pyLARDA.Transformations.plot_timeheight(voldepol_532, rg_converter=True, contour=contour,
                                                          range_interval=plot_range, fig_size=fig_sz)
        name = f'polly_depol_corrected_{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.savefig(name, dpi=300)
        print(name)

    """
        ___  _    ____ ___    ___ ____ ____ _ _  _ _ _  _ ____     ____ ____ ___ 
        |__] |    |  |  |      |  |__/ |__| | |\ | | |\ | | __  __ [__  |___  |  
        |    |___ |__|  |      |  |  \ |  | | | \| | | \| |__]     ___] |___  |  
                                                                     
    """

    if plot_trainingset:
        # interpolation

        # attbsc_1064['var'] = np.ma.masked_equal(attbsc_1064['var'], 0.0)
        # attbsc_1064['var'] = np.ma.masked_invalid(attbsc_1064['var'])
        bsc_ip = pyLARDA.Transformations.interpolate2d(attbsc_1064,
                                                       new_time=LIMRAD94_ZE['ts'],
                                                       new_range=LIMRAD94_ZE['rg'])

        depol_ip = pyLARDA.Transformations.interpolate2d(voldepol_532,
                                                         new_time=LIMRAD94_ZE['ts'],
                                                         new_range=LIMRAD94_ZE['rg'])

        ZE_mask = LIMRAD94_ZE['mask'].copy()
        bsc_ip['mask'] = np.logical_or(bsc_ip['mask'], ZE_mask)
        depol_ip['mask'] = np.logical_or(depol_ip['mask'], ZE_mask)

        fig, _ = pyLARDA.Transformations.plot_timeheight(bsc_ip,
                                                         fig_size=fig_sz, range_interval=plot_range,
                                                         z_converter='log', contour=contour, rg_converter=True
                                                         )
        fig.tight_layout()
        fig_name = 'attbsc_trainingset_' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

        fig, _ = pyLARDA.Transformations.plot_timeheight(depol_ip,
                                                         fig_size=fig_sz, range_interval=plot_range,
                                                         rg_converter=True, contour=contour
                                                         )
        fig.tight_layout()
        fig_name = 'depol_trainingset_' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

        # scatter plot
        titlestring = f'scatter bsc-depl -- date: {begin_dt:%Y-%m-%da},\ntime: {begin_dt:%H:%M:%S} -' \
                      f' {end_dt:%H:%M:%S} UTC, range:{plot_range[0]}m - {plot_range[1]}m'

        bsc_ip['var'] = np.ma.log10(bsc_ip['var'])
        bsc_ip['var_unit'] = 'log_10(sr^-1 m^-1)'

        fig, _ = pyLARDA.Transformations.plot_scatter(depol_ip, bsc_ip,
                                                      fig_size=[7, 7], x_lim=[0, 0.5], y_lim=[-7, -2.5],
                                                      colorbar=True, title=titlestring)

        file_name = f'scatter_polly_depol_bsc_{begin_dt:%Y-%m-%da}.png'
        fig.savefig(file_name, dpi=250)
        print(f'save png :: {file_name}')
