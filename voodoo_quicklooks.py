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

import logging
import os
import pprint
import sys

import time
import toml
from tensorflow import keras

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

import pyLARDA
import pyLARDA.helpers as h

import voodoo.libVoodoo.Loader as Loader
import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model
from voodoo.model_ini import *

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
    plot_bsc_dpl_rangespec = False
    predict_model = False
    lidar_list = [key for key in target_info.keys() if target_info[key]['used']]

    start_time = time.time()

    log = {'larda': logging.getLogger('pyLARDA'), 'voodoo': logging.getLogger('libVoodoo')}
    log['larda'].setLevel(logging.INFO)
    log['voodoo'].setLevel(logging.INFO)
    log['larda'].addHandler(logging.StreamHandler())
    log['voodoo'].addHandler(logging.StreamHandler())

    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=False)

    case_list = toml.load(VOODOO_PATH + TRAIN_SHEET)
    pprint.pprint(case_list)

    for case in case_list['case'].values():
        if case['notes'] == 'ex': continue  # exclude this case and check the next one

        begin_dt = datetime.datetime.strptime(case['begin_dt'], '%Y%m%d-%H%M%S')
        end_dt = datetime.datetime.strptime(case['end_dt'], '%Y%m%d-%H%M%S')

        # create directory for plots
        h.change_dir(os.path.join(PLOTS_PATH + f'trainingdata-QL-{begin_dt:%Y%m%d-%H%M%S}/'))

        ########################################################################################################################################################
        #   ______ _______ ______  _______  ______      _____ __   _  _____  _     _ _______
        #  |_____/ |_____| |     \ |_____| |_____/        |   | \  | |_____] |     |    |
        #  |    \_ |     | |_____/ |     | |    \_      __|__ |  \_| |       |_____|    |
        #
        radar_input_setting = {'rm_precip_ghost': True,  # removes ghost echos (speckles over all chirps) due to precipitation
                               'rm_curtain_ghost': True,  # removes ghost echos (curtain like 1st chirp) due to high signals between 2-5km alt.
                               'do_despeckle': True,  # removes a pixel in 2D arrays, when 80% or more neighbouring pixels are masked (5x5 window)
                               'do_despeckle3d': -1.0,  # save as 2D version but in 3D (5x5x5 window), -1 = no despackle3d
                               'estimate_noise': True,  # calculates the noise level of the Doppler spectra
                               'noise_factor': 6.0,  # number of standard deviations above mean noise
                               'main_peak': True}  # use only main peak for moment calculation

        radar_container = Loader.load_radar_data(larda, begin_dt, end_dt,  **radar_input_setting)

        n_chirp = len(radar_container['spectra'])
        n_time_LR = radar_container['moments']['Ze']['ts'].size
        n_range_LR = radar_container['moments']['Ze']['rg'].size
        n_velocity_LR = radar_container['spectra'][0]['vel'].size
        ts_radar = radar_container['moments']['Ze']['ts']
        rg_radar = radar_container['moments']['Ze']['rg']

        ########################################################################################################################################################
        #         _____ ______  _______  ______      _____ __   _  _____  _     _ _______
        #  |        |   |     \ |_____| |_____/        |   | \  | |_____] |     |    |
        #  |_____ __|__ |_____/ |     | |    \_      __|__ |  \_| |       |_____|    |
        #

        lidar_container = Loader.load_lidar_data(larda, lidar_list, begin_dt, end_dt, msf=True)
        n_time_Pxt = lidar_container['attbsc1064']['ts'].size
        n_range_Pxt = lidar_container['attbsc1064']['rg'].size

        log['voodoo'].info(f'\nLIMRAD94  (n_ts, n_rg, n_vel) = ({n_time_LR},  {n_range_LR},  {n_velocity_LR})')
        log['voodoo'].info(f'POLLYxt   (n_ts, n_rg)        = ({n_time_Pxt}, {n_range_Pxt}) \n')

        # interpolate polly xt data onto limrad grid (so that spectra can be used directly)
        lidar_container.update({f'{var}_ip': pyLARDA.Transformations.interpolate2d(
            lidar_container[var], new_time=ts_radar, new_range=rg_radar, method='nearest') for var in lidar_list})

        ########################################################################################################################################################
        #   _____          _____  _______       _____  _     _ _____ _______ _     _         _____   _____  _     _ _______
        #  |_____] |      |     |    |         |   __| |     |   |   |       |____/  |      |     | |     | |____/  |______
        #  |       |_____ |_____|    |         |____\| |_____| __|__ |_____  |    \_ |_____ |_____| |_____| |    \_ ______|
        #
        if plot_training_set:
            Plot.Quicklooks(radar_container['moments'], lidar_container, begin_dt, end_dt)

        ########################################################################################################################################################
        #   _____          _____  _______       ______  ______     _______  _____  _______ _______                          _____ ______  _______  ______
        #  |_____] |      |     |    |         |_____/ |  ____ ___ |______ |_____] |______ |            ___ ___      |        |   |     \ |_____| |_____/
        #  |       |_____ |_____|    |         |    \_ |_____|     ______| |       |______ |_____                    |_____ __|__ |_____/ |     | |    \_
        #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    _
        new_spec = Loader.equalize_rpg_radar_chirps(radar_container['spectra'])
        if plot_bsc_dpl_rangespec:
            Plot.lidar_profile_range_spectra(lidar_container, new_spec, plot_range=plot_range, colormap='cloudnet_jet')

        if plot_spectra_cwt:
            time_height_mask = Loader.get_mask(new_spec, lidar_container, task='predict')
            cwt_list = Loader.wavlet_transformation(new_spec, time_height_mask, z_converter='lin2z', **feature_info['cwt'])

        ########################################################################################################################################################
        #  _______ _______ _     _ _______     _____   ______ _______ ______  _____ _______ _______ _____  _____  __   _
        #  |  |  | |_____| |____/  |______    |_____] |_____/ |______ |     \   |   |          |      |   |     | | \  |
        #  |  |  | |     | |    \_ |______    |       |    \_ |______ |_____/ __|__ |_____     |    __|__ |_____| |  \_|
        #
        if predict_model:

            # load weights
            file = os.path.join(MODELS_PATH, TRAINED_MODEL)
            loaded_model = keras.models.load_model(file)
            log['voodoo'].info(f'Prediction with model :: {file}')
            log['voodoo'].info('Loading input ...')

            trainingset_settings = {'n_time': n_time_LR,  # number of time steps for LIMRAD94
                                    'n_range': n_range_LR,  # number of range bins for LIMRAD94
                                    'n_Dbins': n_velocity_LR,  # number of Doppler bins steps for LIMRAD94
                                    'task': 'predict',  # masks values for specific task
                                    'output_format': regression_or_binary,  # if True use regression model
                                    'feature_info': feature_info,  # additional information about features
                                    'target_info': target_info,  # additional information about targets
                                    }

            # loadinng the radar data and put into shape for tensorflow
            test_set, test_label, list_ts, list_rg = Loader.load_trainingset(new_spec, radar_container['moments'], lidar_container, **trainingset_settings)

            dimensions = {'list_ts': list_ts, 'list_rg': list_rg, 'ts_radar': ts_radar, 'rg_radar': rg_radar, 'target_info': target_info}

            lidar_pred = Model.predict_lidar(loaded_model, test_set)
            lidar_pred_container = Loader.predict2container(lidar_pred, lidar_list, dimensions,
                                                            {var: larda.connectors['POLLY'].system_info['params'][var] for var in lidar_list})

            ####################################################################################################################################
            # ___  _    ____ ___    ____ ___ ___ ____ _  _ _  _ ____ ___ ____ ___     ___  ____ ____ _  _ ____ ____ ____ ___ ___ ____ ____
            # |__] |    |  |  |     |__|  |   |  |___ |\ | |  | |__|  |  |___ |  \    |__] |__| |    |_/  [__  |    |__|  |   |  |___ |__/
            # |    |___ |__|  |     |  |  |   |  |___ | \| |__| |  |  |  |___ |__/    |__] |  | |___ | \_ ___] |___ |  |  |   |  |___ |  \
            #
            if 'attbsc1064' in lidar_list and use_cnn_regression_model:
                fig, _ = pyLARDA.Transformations.plot_timeheight(lidar_pred_container['attbsc1064_pred'], z_converter='log',
                                                                 zlim=lidar_pred_container['attbsc1064_pred']['var_lims'],
                                                                 title=f'POLLYxt_bsc_pred_{begin_dt:%Y%m%d}', **plot_settings)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                Plot.save_figure(fig, name=f'POLLYxt_attbsc_pred_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}__{TRAINED_MODEL[:-3]}.png', dpi=300)

            ####################################################################################################################################
            #  ___  _    ____ ___    _  _ ____ _    _  _ _  _ ____    ___  ____ ___  ____ _    ____ ____ _ ___  ____ ___ _ ____ _  _
            #  |__] |    |  |  |     |  | |  | |    |  | |\/| |___    |  \ |___ |__] |  | |    |__| |__/ |   /  |__|  |  | |  | |\ |
            #  |    |___ |__|  |      \/  |__| |___ |__| |  | |___    |__/ |___ |    |__| |___ |  | |  \ |  /__ |  |  |  | |__| | \|
            #
            if 'depol' in lidar_list and use_cnn_regression_model:
                fig, _ = pyLARDA.Transformations.plot_timeheight(lidar_pred_container['depol_pred'],
                                                                 zlim=lidar_pred_container['depol_pred']['var_lims'],
                                                                 title=f'POLLYxt_depol_pred_{begin_dt:%Y%m%d}', **plot_settings)
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

                fig, ax = pyLARDA.Transformations.plot_scatter(lidar_pred_container['dpl'], lidar_pred_container['bsc'],
                                                               x_lim=[0, 0.4], y_lim=[-7, -3], title=titlestring)
                Plot.save_figure(fig, name=f'scatter_polly_depol_bsc_{begin_dt:%Y-%m-%d}_{TRAINED_MODEL[:-3]}.png', dpi=200)
