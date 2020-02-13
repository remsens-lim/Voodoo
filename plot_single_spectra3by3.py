#!/usr/bin/env python3
"""
Short description:
    This script provides insight into RPG-FMCW94 cloud radar Doppler spectra information.
"""

import os
import sys

# just needed to find pyLARDA from this location
sys.path.append('../larda/')

import matplotlib
matplotlib.use('Agg')

import datetime
import numpy as np
import logging

import pyLARDA
import pyLARDA.helpers as h
import voodoo.libVoodoo.Loader as Loader
import voodoo.libVoodoo.Plot as Plot
from larda.pyLARDA.spec2mom_limrad94 import build_extended_container

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

########################################################################################################################################################
########################################################################################################################################################
#
#
#               _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#               |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#               |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
########################################################################################################################################################
########################################################################################################################################################
if __name__ == '__main__':

    plot_single_spectra = True
    plot_3by3_spectra   = False

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=True)

    #begin_dt = datetime.datetime(2019, 6, 21, 5, 0, 10)
    #end_dt = datetime.datetime(2019, 6, 21, 5, 0, 30)
    #begin_dt = datetime.datetime(2019, 1, 10, 12, 1, 30)
    #end_dt = datetime.datetime(2019, 1, 10, 12, 2, 0)
    #begin_dt = datetime.datetime(2019, 8, 1, 6, 47, 0)
    #end_dt = datetime.datetime(2019, 8, 1, 6, 48, 0)
    begin_dt = datetime.datetime(2019, 1, 10, 11, 0, 0)
    end_dt = datetime.datetime(2019, 1, 10, 11, 0, 1)

    time_span = [begin_dt, end_dt]
    plot_range = [0, 12000]

    # create directory for plots
    PLOTS_PATH = os.path.dirname(__file__) + '/plots/spectra_png/'
    h.change_dir(f'{PLOTS_PATH}/{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_spectra/')

    # loading the spectrum data
    RPG_spectra = build_extended_container(larda, 'VSpec', time_span, rm_precip_ghost=True, do_despeckle3d=False, estimate_noise=True, noise_factor=6.0)

    RPG_spectra_interp = Loader.equalize_rpg_radar_chirps(RPG_spectra)

    #MIRA_Zspec = larda.read("MIRA", "Zspec", time_span, plot_range)
    #MIRA_Zspec['var'] = MIRA_Zspec['var']*6.0   # add 6 dBZ to mira spectrum power
    #MIRA_Zspec['name'] = 'Zspec + 6[dBZ]'

    NFFT = [128, 512, 1024]
    rg_offsets = RPG_spectra[0]['rg_offsets']

    if plot_single_spectra:

        c_all_vnoispows = larda.read("LIMRAD94", f"VNoisePow", [begin_dt, end_dt], plot_range)
        c_all_vnoispows['var'] = np.ma.masked_less_equal(c_all_vnoispows['var'], -999.0)
        chirp_vnoisepows = [c_all_vnoispows['var'][:, rg_offsets[ic]:rg_offsets[ic + 1]] for ic in range(len(rg_offsets) - 1)]

        c_all_hnoispows = larda.read("LIMRAD94", f"HNoisePow", [begin_dt, end_dt], plot_range)
        c_all_hnoispows['var'] = np.ma.masked_less_equal(c_all_hnoispows['var'], -999.0)
        chirp_hnoisepows = [c_all_hnoispows['var'][:, rg_offsets[ic]:rg_offsets[ic + 1]] for ic in range(len(rg_offsets) - 1)]

        spec_wNoise = [None, None, None]
        for ic in range(1, 2):
            RPG_spectra[ic]['var'] = np.ma.masked_less_equal(RPG_spectra[ic]['var'], 0.0)
            RPG_spectra[ic]['var'] = np.ma.masked_invalid(RPG_spectra[ic]['var'])

            data = RPG_spectra[ic]['var'].copy()
            mean = RPG_spectra[ic]['mean'].copy()
            thresh = RPG_spectra[ic]['threshold'].copy()

            for it in range(len(RPG_spectra[ic]['ts'])):
                for ih in range(len(RPG_spectra[ic]['rg'])):

                    data[it, ih, :] = data[it, ih, :]

                    noise_power = chirp_vnoisepows[ic][0, ih]
                    #noise_power = chirp_hnoisepows[ic][0, ih] + chirp_vnoisepows[ic][0, ih])
                    #data[it, ih, :] = RPG_spectra[ic]['var'][it, ih, :] + noise_power / NFFT[ic]

                    mean_noise = noise_power / NFFT[ic]

                    thresh[it, ih] = np.ma.min(data[it, ih, :])

                    RPG_spectra[ic]['mean'][it, ih] = mean_noise
                    RPG_spectra[ic]['threshold'][it, ih] = np.ma.min(RPG_spectra[ic]['var'][it, ih, :])

            spec_wNoise[ic] = h.put_in_container(data, RPG_spectra[ic], name=f'C{ic + 1}VHSpec with noise power')

            plot_kwargs = {'vmin': -60, 'vmax': 20,                 # spectrum power limits (y-axis)
                           'xmin': -7, 'xmax': 7,                   # Doppler velocity limits (x-axis)
                           'mean': RPG_spectra[ic]['mean'],         # mean noise level of the spectrum
                           'thresh': RPG_spectra[ic]['threshold'],  # signal threshold
                           'title': True,                           # print automated title
                           'z_converter': 'lin2z',                  # convert from [mm6 m-3] -> [dBZ]
                           'fig_size': [7, 6],                       # size of the figure in inches
                           'smooth': True,                          # use plot instead of step
                           'alpha': 0.85,                           # alpha of line plots
                           'save': ''                               # save spectra
                           }
            fig, ax = pyLARDA.Transformations.plot_spectra(RPG_spectra[ic], **plot_kwargs)

    if plot_3by3_spectra:
        plot_kwargs = {'ymin': -60,     'ymax': 20,     # spectrum power limits (y-axis)
                       'xmin': -7,      'xmax': 7,      # Doppler velocity limits (x-axis)
                       'title': True,                   # print automated title
                       'z_converter': 'lin2z',          # convert from [mm6 m-3] -> [dBZ]
                       'fig_size': [10, 10],            # size of the figure in inches
                       'alpha': 0.85}                   # alpha of line plots
        intervall = {'time': [RPG_spectra_interp['ts'][0], RPG_spectra_interp['ts'][-1]], 'range': plot_range}
        time_height_slice = pyLARDA.Transformations.slice_container(RPG_spectra_interp, value=intervall)
        fig, ax = Plot.spectra_3by3(time_height_slice, MIRA_Zspec, **plot_kwargs)

        dummy = 5
