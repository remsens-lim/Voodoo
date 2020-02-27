#!/usr/bin/env python3
"""
Short description:
    Calculating radar polarimetric products from spectra, remove artefact's and speckles
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
__version__ = "0.0.1"
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

        'add_horizontal_channel': True,

    }

    plot_settings = {

        'fig_size': [9, 6],

        'range_interval': [0, 12000],

        'rg_converter': True,

        'contour': {},

        'dpi': 200

    }

    PATH = '/home/sdig/code/larda3/voodoo/plots/spectra2moments_v2/'

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
        # ghost echo  1,2 test case
        begin_dt = datetime.datetime(2019, 1, 21, 12, 1)
        end_dt = datetime.datetime(2019, 1, 21, 12, 59)


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

    plot_settings.update({'contour': {}})
    limrad94_Zspec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_, **spec_settings)
    limrad94_pol = sp.spectra2polarimetry(limrad94_Zspec, larda.connectors['LIMRAD94'].system_info['params'], **spec_settings)

    for iC in range(len(limrad94_Zspec)):
        for varname in ['ldr']:
            limrad94_Zspec[varname] = h.put_in_container(limrad94_pol[varname], limrad94_Zspec['VHSpec'], name=varname)
            limrad94_Zspec[varname]['dimlabel'] = ['time', 'range']
            limrad94_Zspec[varname]['mask'] = np.all(limrad94_pol[f'{varname}_s'] == 0.0, axis=2)
            limrad94_Zspec[varname]['var_lims'] = [-30, 0]
            limrad94_Zspec[varname]['colormap'] = 'LDR'

    ########################################################################################################FONT=CYBERMEDIUM
    #
    #   ___  _    ____ ___ ___ _ _  _ ____    ____ ____ ___  ____ ____    _  _ ____ _  _ ____ _  _ ___ ____
    #   |__] |    |  |  |   |  | |\ | | __    |__/ |__| |  \ |__| |__/    |\/| |  | |\/| |___ |\ |  |  [__
    #   |    |___ |__|  |   |  | | \| |__]    |  \ |  | |__/ |  | |  \    |  | |__| |  | |___ | \|  |  ___]
    #
    # create folder for subfolders if it doesn't exist already
    h.change_dir(PATH)

    compare_to_LV1 = False
    LIMRAD94_RPG = {}
    for var in ['ldr']:
        log.info('loading variable from LV1 :: ' + var)
        LIMRAD94_RPG.update({var: larda.read("LIMRAD94", var, [begin_dt, end_dt], [0, 'max'])})
        LIMRAD94_RPG[var]['var'] = np.ma.masked_less_equal(LIMRAD94_RPG[var]['var'], -999.0)
        log.debug('mean diff of {} = {}'.format(var, np.ma.mean(limrad94_Zspec[var]['var'] - LIMRAD94_RPG[var]['var'])))

    LIMRAD94_ldr = LIMRAD94_RPG['ldr']
    if compare_to_LV1:
        LIMRAD94_ldr['var'] = limrad94_Zspec['ldr']['var'] - LIMRAD94_RPG['SLDR']['var']
        LIMRAD94_ldr['var_lims'] = [-1.e-5, 1.e-5]

    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ldr, **plot_settings)
    Plot.save_figure(fig, name=f'limrad_ldrLV1_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=plot_settings['dpi'])

    LIMRAD94_ldr = limrad94_Zspec['ldr']
    fig, _ = pyLARDA.Transformations.plot_timeheight(LIMRAD94_ldr, z_converter='lin2z', **plot_settings)
    Plot.save_figure(fig, name=f'limrad_ldrLV0_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=plot_settings['dpi'])

    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))
