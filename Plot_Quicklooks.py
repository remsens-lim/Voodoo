#!/usr/bin/env python3
"""
Short description:
    Cloud Radar Doppler Spectra --> Lidar Backscatter and Depolarization

    This is the main program of the voodoo Project. The workflow is as follows:
        1.  reading cloud radar and polarization lidar data and applies pre-processing routines to clean the data.
        2.  plotting the training set and/or the target set, additional plots are provided as well (scatter, history, histograms, ...)
        3.  create an ANN model using the Tensorflow/Keras library or load an existing ANN model.
        4.  training and/or prediction of lidar backscatter and/or lidar depolarization using the ANN model.
        5.  plot predicted target variables


Long description:
    This project aims to provides lidar predictions for variables, such as backscatter and depolarization, beyond regions where the lidar signal was
    completely attenuated by exploiting cloud radar Doppler spectra morphologies.
    An earlier approach was developed by Ed Luke et al. 2010
"""


import glob
import os
import sys
import time
import datetime
import logging

import toml
import pprint
import numpy as np
import itertools
from tensorflow import keras

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

import pyLARDA
import pyLARDA.helpers as h

import libVoodoo.Loader as Loader
import libVoodoo.Plot   as Plot
import libVoodoo.Model  as Model

__author__      = "Willi Schimmel"
__copyright__   = "Copyright 2019, The Voodoo Project"
__credits__     = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__     = "MIT"
__version__     = "0.0.1"
__maintainer__  = "Willi Schimmel"
__email__       = "willi.schimmel@uni-leipzig.de"
__status__      = "Prototype"


########################################################################################################################################################
########################################################################################################################################################
#
#   ______         _____  ______  _______              _____  _______  ______ _______ _______      _______  _____  __   _ _______ _____  ______
#  |  ____ |      |     | |_____] |_____| |           |_____] |_____| |_____/ |_____| |  |  |      |       |     | | \  | |______   |   |  ____
#  |_____| |_____ |_____| |_____] |     | |_____      |       |     | |    \_ |     | |  |  |      |_____  |_____| |  \_| |       __|__ |_____|
#
#
########################################################################################################################################################
########################################################################################################################################################

t0_voodoo = datetime.datetime.today()
time_str = f'{t0_voodoo:%Y%m%d-%H%M%S}'

predict_model = False

plot_training_set           = False
plot_training_set_histogram = False
plot_bsc_dpl_rangespec      = True

add_moments = False
add_spectra = False
add_cwt     = True

fig_size   = [12, 7]
plot_range = [0, 10000]

TRAIN_SHEET = 'training_cases.toml'
TEST_SHEET  = 'training_cases.toml'

# define paths
VOODOO_PATH  = '/home/sdig/code/larda3/voodoo/'
LOGS_PATH    = f'{VOODOO_PATH}logs/'
MODELS_PATH  = f'{VOODOO_PATH}models/'
PLOTS_PATH   = f'{VOODOO_PATH}plots/'


TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-8_8-kernelsize-leakyrelu--20191128-173950.h5'  # even better


radar_list = []
radar_info = {'spec_lims':     [1.e-6, 1.e2],
              'spec_converter': 'lin2z',
              'normalization':  'normalize'}

lidar_list = ['attbsc1064', 'depol']
lidar_info = {'attbsc1064_lims': [1.e-7, 1.e-3],
              'voldepol_lims': [1.e-7, 0.3],
              'bsc_converter': 'log',
              'dpl_converter': 'none',
              'normalization': 'none',
              'bsc_shift': 0,
              'dpl_shift': 0}

# controls the ccontinuous wavelet transformation
CWT_PARAMS   = {'dim': '2d',
                'sfacs': np.linspace(2 ** 1., 2 ** 3.25, 32)}

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

    start_time = time.time()

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=False)

    case_list = toml.load(VOODOO_PATH + TRAIN_SHEET)
    pprint.pprint(case_list)

    for case in case_list['case'].values():
        if case['notes'] == 'ex': continue  # exclude this case and check the next one

        begin_dt = datetime.datetime.strptime(case['begin_dt'], '%Y%m%d-%H%M%S')
        end_dt   = datetime.datetime.strptime(case['end_dt'],   '%Y%m%d-%H%M%S')

        # create directory for plots
        h.change_dir(os.path.join(PLOTS_PATH + f'trainingdata-QL-{begin_dt:%Y%m%d-%H%M%S}/'))

        ########################################################################################################################################################
        #   ______ _______ ______  _______  ______      _____ __   _  _____  _     _ _______
        #  |_____/ |_____| |     \ |_____| |_____/        |   | \  | |_____] |     |    |
        #  |    \_ |     | |_____/ |     | |    \_      __|__ |  \_| |       |_____|    |
        #
        radar_container = Loader.load_radar_data(larda, begin_dt, end_dt,
                                                 rm_precip_ghost=True,  rm_curtain_ghost=True,
                                                 do_despeckle=True,     do_despeckle3d=-1.0,
                                                 estimate_noise=True,   noise_factor=6.0,      main_peak=True
                                                 )

        n_chirp       = len(radar_container['spectra'])
        n_time_LR     = radar_container['moments']['Ze']['ts'].size
        n_range_LR    = radar_container['moments']['Ze']['rg'].size
        n_velocity_LR = radar_container['spectra'][0]['vel'].size
        ts_radar      = radar_container['moments']['Ze']['ts']
        rg_radar      = radar_container['moments']['Ze']['rg']

        ########################################################################################################################################################
        #         _____ ______  _______  ______      _____ __   _  _____  _     _ _______
        #  |        |   |     \ |_____| |_____/        |   | \  | |_____] |     |    |
        #  |_____ __|__ |_____/ |     | |    \_      __|__ |  \_| |       |_____|    |
        #
        lidar_container = Loader.load_lidar_data(larda, lidar_list, begin_dt, end_dt, plot_range, msf=True)
        n_time_Pxt      = lidar_container['attbsc1064']['ts'].size
        n_range_Pxt     = lidar_container['attbsc1064']['rg'].size

        print(f'\nLIMRAD94  (n_ts, n_rg, n_vel) = ({n_time_LR},  {n_range_LR},  {n_velocity_LR})')
        print(f'POLLYxt   (n_ts, n_rg)        = ({n_time_Pxt}, {n_range_Pxt}) \n')

        # interpolate polly xt data onto limrad grid (so that spectra can be used directly)
        lidar_container.update({f'{var}_ip': pyLARDA.Transformations.interpolate2d(
                lidar_container[var], new_time=ts_radar, new_range=rg_radar) for var in lidar_list})

        ########################################################################################################################################################
        #  ___  _    ____ ___ ___ _ _  _ ____    _    _ ___  ____ ____    ____ ____ _  _ ____ ____ ____ ___  ____ ____
        #  |__] |    |  |  |   |  | |\ | | __    |    | |  \ |__| |__/    |__/ |__| |\ | | __ |___ [__  |__] |___ |
        #  |    |___ |__|  |   |  | | \| |__]    |___ | |__/ |  | |  \    |  \ |  | | \| |__] |___ ___] |    |___ |___
        #

        if plot_bsc_dpl_rangespec:
            new_spec = Loader.equalize_radar_chirps(radar_container['spectra'])
            Plot.lidar_profile_range_spectra(lidar_container, new_spec, plot_range=plot_range, colormap='cloudnet_jet')

        ########################################################################################################################################################
        #   _____          _____  _______ _______ _____ __   _  ______       _____
        #  |_____] |      |     |    |       |      |   | \  | |  ____      |   __| |
        #  |       |_____ |_____|    |       |    __|__ |  \_| |_____|      |____\| |_____
        #
        if plot_training_set:
            Plot.Quicklooks(radar_container['moments'], lidar_container, radar_list, lidar_list, begin_dt, end_dt)

        ########################################################################################################################################################
        #  _______ _______ _     _ _______     _____   ______ _______ ______  _____ _______ _______ _____  _____  __   _
        #  |  |  | |_____| |____/  |______    |_____] |_____/ |______ |     \   |   |          |      |   |     | | \  |
        #  |  |  | |     | |    \_ |______    |       |    \_ |______ |_____/ __|__ |_____     |    __|__ |_____| |  \_|
        #
        if predict_model:

            # load weights
            file = os.path.join(MODELS_PATH, TRAINED_MODEL)
            loaded_model = keras.models.load_model(file)
            print(f'Prediction with model :: {file}')
            print('Loading input ...')

            test_set, test_label, list_ts, list_rg = Loader.load_trainingset(radar_container['spectra'],
                                                                             radar_container['moments'],
                                                                             lidar_container,
                                                                             n_time=n_time_LR,
                                                                             n_range=n_range_LR,
                                                                             n_Dbins=n_velocity_LR,
                                                                             task='predict_lidar',
                                                                             add_moments=add_moments,
                                                                             add_spectra=add_spectra,
                                                                             add_cwt=add_cwt,
                                                                             feature_info=radar_info,
                                                                             feature_list=radar_list,
                                                                             label_info=lidar_info,
                                                                             label_list=lidar_list,
                                                                             cwt=CWT_PARAMS
                                                                             )

            dimensions = {'list_ts': list_ts, 'list_rg': list_rg, 'ts_radar': ts_radar, 'rg_radar': rg_radar, 'label_info': lidar_info}

            lidar_pred = Model.predict_lidar(loaded_model, test_set)
            lidar_pred_container = Loader.predict2container(lidar_pred,
                                                            lidar_list,
                                                            dimensions,
                                                            {var: larda.connectors['POLLY'].system_info['params'][var] for var in lidar_list})

            ####################################################################################################################################
            # ___  _    ____ ___    ____ ___ ___ ____ _  _ _  _ ____ ___ ____ ___     ___  ____ ____ _  _ ____ ____ ____ ___ ___ ____ ____
            # |__] |    |  |  |     |__|  |   |  |___ |\ | |  | |__|  |  |___ |  \    |__] |__| |    |_/  [__  |    |__|  |   |  |___ |__/
            # |    |___ |__|  |     |  |  |   |  |___ | \| |__| |  |  |  |___ |__/    |__] |  | |___ | \_ ___] |___ |  |  |   |  |___ |  \
            #
            if 'attbsc1064' in lidar_list:
                fig, _ = pyLARDA.Transformations.plot_timeheight(lidar_pred_container['attbsc1064_pred'],
                                                                 fig_size=fig_size,
                                                                 z_converter='log',
                                                                 range_interval=plot_range,
                                                                 zlim=lidar_pred_container['attbsc1064_pred']['var_lims'],
                                                                 rg_converter=True,
                                                                 title=f'POLLYxt_bsc_pred_{begin_dt:%Y%m%d}')  # , contour=contour)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                Plot.save_figure(fig, name=f'POLLYxt_attbsc_pred_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}__{TRAINED_MODEL[:-3]}.png', dpi=300)

            ####################################################################################################################################
            #  ___  _    ____ ___    _  _ ____ _    _  _ _  _ ____    ___  ____ ___  ____ _    ____ ____ _ ___  ____ ___ _ ____ _  _
            #  |__] |    |  |  |     |  | |  | |    |  | |\/| |___    |  \ |___ |__] |  | |    |__| |__/ |   /  |__|  |  | |  | |\ |
            #  |    |___ |__|  |      \/  |__| |___ |__| |  | |___    |__/ |___ |    |__| |___ |  | |  \ |  /__ |  |  |  | |__| | \|
            #
            if 'voldepol532' in lidar_list:
                fig, _ = pyLARDA.Transformations.plot_timeheight(lidar_pred_container['voldepol532_pred'], fig_size=fig_size,
                                                                 range_interval=plot_range,
                                                                 zlim=lidar_pred_container['voldepol532_pred']['var_lims'],
                                                                 rg_converter=True,
                                                                 title=f'POLLYxt_depol_pred_{begin_dt:%Y%m%d}')
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                Plot.save_figure(fig, name=f'POLLYxt_depol_pred_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}__{TRAINED_MODEL[:-3]}.png', dpi=300)

            ####################################################################################################################################
            #  ___  _    ____ ___    ____ ____ ____ ___ ___ ____ ____    ____ ___ ___ ___  ____ ____    ___  ____ ___  ____ _
            #  |__] |    |  |  |     [__  |    |__|  |   |  |___ |__/    |__|  |   |  |__] [__  |    __ |  \ |___ |__] |  | |
            #  |    |___ |__|  |     ___] |___ |  |  |   |  |___ |  \    |  |  |   |  |__] ___] |___    |__/ |___ |    |__| |___
            #
            if 'attbsc1064_pred' and 'voldepol532_pred' in lidar_list:
                # scatter plot
                titlestring = f'scatter bsc-depol -- date: {begin_dt:%Y-%m-%da},\ntime: {begin_dt:%H:%M:%S} -' \
                              f' {end_dt:%H:%M:%S} UTC, range:{plot_range[0]}-{plot_range[1]}m'

                lidar_pred_container['bsc']['var'][lidar_pred_container['bsc']['var'] <= 0.0] = lidar_pred_container['bsc']['var_lims'][0]
                lidar_pred_container['bsc']['var'] = np.log10(lidar_pred_container['bsc']['var'])
                lidar_pred_container['bsc']['var_unit'] = 'log10(sr^-1 m^-1)'
                lidar_pred_container['bsc']['var_lims'] = [-7, -3]

                fig, ax = pyLARDA.Transformations.plot_scatter(lidar_pred_container['dpl'],
                                                               lidar_pred_container['bsc'],
                                                               x_lim=[0, 0.4],
                                                               y_lim=[-7, -3],
                                                               # z_converter='log',
                                                               title=titlestring)

                Plot.save_figure(fig, name=f'scatter_polly_depol_bsc_{begin_dt:%Y-%m-%d}_{TRAINED_MODEL[:-3]}.png', dpi=200)
