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
import matplotlib.pyplot as plt

import pyLARDA
import pyLARDA.helpers as h
import voodoo.libVoodoo.Loader as Loader
import voodoo.libVoodoo.Plot as Plot
from larda.pyLARDA.spec2mom_limrad94 import build_extended_container, spectra2moments

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

    plot_range_spectro_fft = True

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=True)

    begin_dt = datetime.datetime(2019, 8, 1, 5, 0)
    end_dt = datetime.datetime(2019, 8, 1, 9, 0)
    #begin_dt = datetime.datetime(2019, 2, 12, 0, 1)
    #end_dt = datetime.datetime(2019, 2, 12, 0, 2)
    #begin_dt = datetime.datetime(2019, 3, 18, 4, 1)
    #end_dt = datetime.datetime(2019, 3, 18, 17, 1)

    plot_range = [0, 12000]

    # create directory for plots
    PLOTS_PATH = os.path.dirname(__file__) + '/plots/rg-spectrogram-fft/'
    h.change_dir(f'{PLOTS_PATH}/{begin_dt:%Y%m%d-%H%M%S}_spectro-fft/')

    # loading the spectrum data
    time_span = [begin_dt, end_dt]
    RPG_spectra = build_extended_container(larda, 'VSpec', time_span, rm_precip_ghost=True, do_despeckle3d=False, estimate_noise=True, noise_factor=6.0)

    sensitivity_limit = larda.read("LIMRAD94", "SLv", time_span, [0, 12000])
    RPG_spectra_interp = Loader.equalize_rpg_radar_chirps(RPG_spectra)
    RPG_spectra_interp['var'] = Loader.replace_fill_value(RPG_spectra_interp['var'], sensitivity_limit['var'])
    RPG_spectra_interp['mask'][:, :] = False

    #    MIRA_Zspec = larda.read("MIRA", "Zspec", [begin_dt, end_dt], [0, 12000])
#    MIRA_Zspec['var'] = MIRA_Zspec['var'] * 2.0  # add 6 dBZ to mira spectrum power
#    MIRA_Zspec['name'] = 'Zspec + 2[dBZ]'

    NFFT = [128, 512, 1024]
    rg_offsets = RPG_spectra[0]['rg_offsets']
    ts_list    = RPG_spectra_interp['ts']

    FONTW = 'semibold'
    FONTS = 12
    FIGS  = [6, 6]

    if plot_range_spectro_fft:

        for its, ts in enumerate(ts_list):
            intervall = {'time': [ts], 'range': plot_range}

            spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(RPG_spectra_interp, value=intervall)
            spectrogram_LIMRAD['colormap'] = 'cloudnet_jet'
            spectrogram_LIMRAD['var_lims'] = [-60, 20]
            spectrogram_LIMRAD['var_unit'] = 'dBZ'
            spectrogram_LIMRAD['rg_unit'] = 'km'
            spectrogram_LIMRAD['rg'] = spectrogram_LIMRAD['rg'] / 1000.

            fig_name = f'limrad_{str(its).zfill(4)}_{begin_dt:%Y%m%d}'

            fig_sp, (axspec, pcmesh) = pyLARDA.Transformations.plot_spectrogram(spectrogram_LIMRAD,
                                                                                z_converter='lin2z', fig_size=FIGS, v_lims=[-4, 4], grid='both', cbar=True)
            # additional spectrogram settings
            axspec.patch.set_facecolor('#E0E0E0')
            axspec.patch.set_alpha(0.7)
            axspec.set_ylim(np.array(plot_range)/1000.)
            axspec.grid(b=True, which='major', color='white', linestyle='--')
            axspec.grid(b=True, which='minor', color='white', linestyle=':')
            axspec.grid(linestyle=':', color='white')
            axspec.set_ylabel('Height [km]', fontsize=FONTS, fontweight=FONTW)
            axspec.tick_params(axis='y', which='both', right=True, top=True)
            plt.tight_layout()
            Plot.save_figure(fig_sp, name=f'rgspec-{fig_name}.png', dpi=100)

            kwargs = {'vmin': 1.e-3, 'vmax': 1.e1,
                      'z_converter': 'lin2z'}
            fig_fft, axfft = Plot.range_spectro_fft(spectrogram_LIMRAD, **kwargs)
            plt.tight_layout()
            Plot.save_figure(fig_fft, name=f'rgspecfft-{fig_name}.png', dpi=100)
            print(its)
            dummy = 5

#            spectrogram_MIRA = pyLARDA.Transformations.slice_container(MIRA_Zspec, value=intervall)
#            spectrogram_MIRA['colormap'] = 'cloudnet_jet'
#            spectrogram_MIRA['var_lims'] = [-60, 20]
#
#            fig_name = f'mira_{str(its).zfill(4)}_{begin_dt:%Y%m%d}-rg-spectro-fft.png'
#            fig, _ = Plot.range_spectro_fft(spectrogram_MIRA, v_lims=[-4, 4], z_converter='lin2z')
#            Plot.save_figure(fig, name=fig_name, dpi=200)
