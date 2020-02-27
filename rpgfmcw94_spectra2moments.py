#!/usr/bin/env python3
"""
Short description:
    Calculating radar moments from spectra, remove artefact's and speckles
"""

import datetime
import sys

sys.path.append('../larda/')
sys.path.append('.')

import logging
import numpy as np
import time


import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.SpectraProcessing as sp

import voodoo.libVoodoo.Plot   as Plot

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "3.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
if __name__ == '__main__':

    spec_settings = {

        'despeckle2D': True,  # 2D convolution (5x5 window), removes single non-zero values,

        'main_peak': True,  #

        'ghost_echo_1': True,  # reduces the domain (Nyquist velocitdy) by ± 2.5 [m/s], when signal > 0 [dBZ] within 200m above antenna

        'ghost_echo_2': True,  #

    }

    plot_settings = {

        'fig_size': [9, 6],

        'range_interval': [0, 12000],

        'rg_converter': True,

        'contour': {},

        'dpi': 200

    }

    PATH = '/home/sdig/code/larda3/voodoo/plots/spectra2moments_v2/'
    NCPATH = '/home/sdig/code/larda3/voodoo/nc-files/spectra/'

    start_time = time.time()

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu')

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    # gather argument
    if 'date' in kwargs:
        date = str(kwargs['date'])
        begin_dt = datetime.datetime.strptime(date + ' 00:00:10', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 23:59:50', '%Y%m%d %H:%M:%S')
    else:
        # ghost echo 2 test case
        # begin_dt = datetime.datetime(2019, 1, 1, 0, 1)
        # end_dt = datetime.datetime(2019, 1, 1, 0, 59)
        # begin_dt = datetime.datetime(2018, 12, 18, 16, 1)
        # end_dt = datetime.datetime(2018, 12, 18, 18, 59)

        # ghost echo  1 test case
        # begin_dt = datetime.datetime(2019, 3, 15, 10, 1)
        # end_dt = datetime.datetime(2019, 3, 15, 14, 59)

        # ghost echo  3 test case
        # begin_dt = datetime.datetime(2019, 7, 13, 16, 1)
        # end_dt = datetime.datetime(2019, 7, 13, 16, 59)

        # ghost echo  1,2 test case
        # begin_dt = datetime.datetime(2019, 1, 21, 12, 1)
        # end_dt = datetime.datetime(2019, 1, 21, 15, 59)

        # save spectra test case
        begin_dt = datetime.datetime(2019, 4, 10, 20, 59)
        end_dt = datetime.datetime(2019, 4, 10, 21, 31)

    TIME_SPAN_ = [begin_dt, end_dt]

    # check if cloudnet data is available to plot isotherms
    try:
        # loading cloudnet temperature data
        T = larda.read("CLOUDNETpy94", "T", [begin_dt, end_dt], [0, 'max'])

        def toC(datalist):
            return datalist[0]['var'] - 273.15, datalist[0]['mask']

        T = pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"})
        contour = {'data': T, 'levels': np.arange(-35, 1, 10)}
        plot_settings.update({'contour': contour})
    except Exception as e:
        print(e)

    limrad94_Zspec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_, **spec_settings)
    limrad94_mom = sp.spectra2moments(limrad94_Zspec, larda.connectors['LIMRAD94'].system_info['params'], **spec_settings)

    ########################################################################################################FONT=CYBERMEDIUM
    #
    #   ___  _    ____ ___ ___ _ _  _ ____    ____ ____ ___  ____ ____    _  _ ____ _  _ ____ _  _ ___ ____
    #   |__] |    |  |  |   |  | |\ | | __    |__/ |__| |  \ |__| |__/    |\/| |  | |\/| |___ |\ |  |  [__
    #   |    |___ |__|  |   |  | | \| |__]    |  \ |  | |__/ |  | |  \    |  | |__| |  | |___ | \|  |  ___]
    #
    # create folder for subfolders if it doesn't exist already

    make_plots = False

    if make_plots:
        h.change_dir(PATH)

        compare_to_LV1 = False
        LIMRAD94_RPG = {}
        for var in ['Ze', 'VEL', 'sw', 'kurt', 'skew']:
            log.info('loading variable from LV1 :: ' + var)
            LIMRAD94_RPG.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
            LIMRAD94_RPG[var]['var'] = np.ma.masked_less_equal(LIMRAD94_RPG[var]['var'], -999.0)
            log.debug('mean diff of {} = {}'.format(var, np.ma.mean(limrad94_mom[var]['var'][:, :] - LIMRAD94_RPG[var]['var'][:, :])))

        LIMRAD94_ZE = LIMRAD94_RPG['Ze']
        if compare_to_LV1:
            LIMRAD94_ZE['var'] = limrad94_mom['Ze']['var'] - LIMRAD94_RPG['Ze']['var']
            LIMRAD94_ZE['var_lims'] = [-1.e-5, 1.e-5]
        else:
            LIMRAD94_ZE['var_unit'] = 'dBZ'
            LIMRAD94_ZE['var_lims'] = [-60, 20]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ZE, z_converter='lin2z', **plot_settings)
        Plot.save_figure(fig, name=f'limrad_ZeLV1_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=plot_settings['dpi'])

        LIMRAD94_ZE = limrad94_mom['Ze']
        LIMRAD94_ZE['var_unit'] = 'dBZ'
        LIMRAD94_ZE['var_lims'] = [-60, 20]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ZE, z_converter='lin2z', **plot_settings)
        Plot.save_figure(fig, name=f'limrad_ZeLV0_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=plot_settings['dpi'])

        LIMRAD94_VEL = limrad94_mom['VEL']
        if compare_to_LV1:
            LIMRAD94_VEL['var'] = limrad94_mom['VEL']['var'] - LIMRAD94_RPG['VEL']['var']
            LIMRAD94_VEL['var_lims'] = [-1.e-5, 1.e-5]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_VEL, **plot_settings)
        Plot.save_figure(fig, name=f'limrad_VELLV0_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=plot_settings['dpi'])

        LIMRAD94_VEL = LIMRAD94_RPG['VEL']
        if compare_to_LV1:
            LIMRAD94_VEL['var'] = limrad94_mom['VEL']['var'] - LIMRAD94_RPG['VEL']['var']
            LIMRAD94_VEL['var_lims'] = [-1.e-5, 1.e-5]
        fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_VEL, **plot_settings)
        Plot.save_figure(fig, name=f'limrad_VELLV1_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=plot_settings['dpi'])
        #
        #    LIMRAD94_sw = limrad94_mom['sw']
        #    if compare_to_LV1:
        #        LIMRAD94_sw['var_lims'] = [-1.e-4, 1.e-4]
        #        LIMRAD94_sw['var'] = limrad94_mom['sw']['var'] - LIMRAD94_RPG['sw']['var']
        #    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_sw, **plot_settings)
        #    fig_name = f'limrad_sw_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png'
        #    fig.savefig(fig_name, dpi=250)
        #
        #    LIMRAD94_sk = limrad94_mom['skew']
        #    if compare_to_LV1:
        #        LIMRAD94_sk['var_lims'] = [-1.e-4, 1.e-4]
        #        LIMRAD94_sk['var'] = limrad94_mom['skew']['var'] - LIMRAD94_RPG['skew']['var']
        #    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_sk, **plot_settings)
        #    fig_name = f'limrad_skew_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png'
        #    fig.savefig(fig_name, dpi=250)
        #
        #    LIMRAD94_kt = limrad94_mom['kurt']
        #    if compare_to_LV1:
        #        LIMRAD94_kt['var_lims'] = [-1.e-4, 1.e-4]
        #        LIMRAD94_kt['var'] = limrad94_mom['kurt']['var'] - LIMRAD94_RPG['kurt']['var']
        #    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_kt, **plot_settings)
        #    fig_name = f'limrad_kurt_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png'
        #    fig.savefig(fig_name, dpi=250)
        #    ########################################################################################################################

    make_mat_file = True

    if make_mat_file:
        h.change_dir(NCPATH)
        from scipy.io import savemat
        FILE_NAME_1 = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_limrad94_spectra.mat'
        savemat(f'{FILE_NAME_1}', limrad94_Zspec.pop('VHSpec'))  # store spectra separately from other arrays
        print(f'save :: {FILE_NAME_1}')

        FILE_NAME_2 = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}_limrad94_spectra_extra.mat'
        savemat(FILE_NAME_2, limrad94_Zspec)
        print(f'save :: {FILE_NAME_1}')

    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))
