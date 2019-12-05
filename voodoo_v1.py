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
#var = input("Training [t] or Predicting [p]?:  ")
var = 't'

if var == 'p':
    train_model   = False
    predict_model = True
else:
    train_model   = True
    predict_model = False

plot_training_set           = False
plot_training_set_histogram = False
plot_training_history       = True
plot_bsc_dpl_rangespec      = False

add_moments = False
add_spectra = False
add_cwt     = True

fig_size   = [12, 7]
plot_range = [0, 12000]

TRAIN_SHEET = 'training_cases.toml'
TEST_SHEET  = 'training_cases.toml'

# define ANN model hyperparameter space
use_mlp_model = False
use_cnn_model = True

BATCH_SIZE   = 1
EPOCHS       = 150

DENSE_LAYERS = [0]
LAYER_SIZES  = []

CONV_LAYERS  = [3]
KERNEL_SIZE  = (8, 8)
POOL_SIZE    = (2, 2)
NFILTERS     = [(32, 64, 128)]

OPTIMIZERS   = ['sgd']
ACTIVATIONS  = ['leakyrelu']
LOSS_FCNS    = ['mse']

# define paths
VOODOO_PATH  = '/home/sdig/code/larda3/voodoo/'
BOARD_NAME   = f'voodoo-mlp-training-{time_str}__{BATCH_SIZE}-bachsize-{EPOCHS}-epochs'
LOGS_PATH    = f'{VOODOO_PATH}logs/'
MODELS_PATH  = f'{VOODOO_PATH}models/'
PLOTS_PATH   = f'{VOODOO_PATH}plots/'


TRAINED_MODEL = '3-conv-(32, 64, 128)-filter-8_8-kernelsize-leakyrelu--20191128-173950.h5'  # even better
#TRAINED_MODEL = '2-conv-64-filter-8_8-kernelsize-leakyrelu--20191126-215349.h5' # best one jet
#TRAINED_MODEL = '3-conv-64-filter-5_5-kernelsize-leakyrelu--20191122-171526.h5'


# define normalization boundaries and conversion for radar (feature) and lidar (label) space
#radar_info   = {'Ze_lims':       [1.e-6, 1.e2],
#                'sw_lims':       [0, 3],
#                'spec_lims':     [1.e-6, 1.e2],
#                'Ze_converter':  'lin2z',
#                'spec_converter': 'lin2z',
#                'normalization':  'normalize'}
#
#lidar_info   = {'bsc_lims': [1.e-7, 1.e-3],
#                'dpl_lims': [1.e-7, 0.3],
#                'bsc_converter': 'log',
#                'dpl_converter': 'ldr2cdr',
#                'normalization': 'none',
#                'bsc_shift': 0,
#                'dpl_shift': 0}

radar_list = []
radar_info = {'spec_lims':     [1.e-6, 1.e2],
              'spec_converter': 'lin2z',
              'normalization':  'normalize'}

lidar_list = ['attbsc1064']
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
        lidar_container = Loader.load_lidar_data(larda, lidar_list, begin_dt, end_dt, plot_range, msf=False)
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
            Plot.lidar_profile_range_spectra(lidar_container, new_spec, iT=0)
            sys.exit(42)

        ########################################################################################################################################################
        #   _____          _____  _______ _______ _____ __   _  ______       _____
        #  |_____] |      |     |    |       |      |   | \  | |  ____      |   __| |
        #  |       |_____ |_____|    |       |    __|__ |  \_| |_____|      |____\| |_____
        #
        if plot_training_set:
            Plot.Quicklooks(radar_container['moments'], lidar_container, radar_list, lidar_list, begin_dt, end_dt)

        ########################################################################################################################################################
        #  _______  ______ _______ _____ __   _      _______ __   _ __   _
        #     |    |_____/ |_____|   |   | \  |      |_____| | \  | | \  |
        #     |    |    \_ |     | __|__ |  \_|      |     | |  \_| |  \_|
        #
        if train_model:
            train_set, train_label, list_ts, list_rg = Loader.load_trainingset(radar_container['spectra'],
                                                                               radar_container['moments'],
                                                                               lidar_container,
                                                                               n_time=n_time_LR,
                                                                               n_range=n_range_LR,
                                                                               n_Dbins=n_velocity_LR,
                                                                               task='train_radar_lidar',
                                                                               add_moments=add_moments,
                                                                               add_spectra=add_spectra,
                                                                               add_cwt=add_cwt,
                                                                               feature_info=radar_info,
                                                                               feature_list=radar_list,
                                                                               label_info=lidar_info,
                                                                               label_list=lidar_list,
                                                                               cwt=CWT_PARAMS)

            # get dimensionality of the feature and target space
            n_samples, n_input = train_set.shape[0], train_set.shape[1:]
            n_output  = train_label.shape[1:]

            ####################################################################################################################################
            #  ___  _    ____ ___    _  _ _ ____ ___ ____ ____ ____ ____ _  _ ____
            #  |__] |    |  |  |     |__| | [__   |  |  | | __ |__/ |__| |\/| [__
            #  |    |___ |__|  |     |  | | ___]  |  |__| |__] |  \ |  | |  | ___]
            #
            if plot_training_set_histogram:
                fig, _ = Plot.Histogram(train_set, var_info=radar_info, kind='cwt2d')
                Plot.save_figure(fig, name=f'histo_input_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_{radar_info["normalization"]}.png', dpi=200)

                fig, _ = Plot.Histogram(train_label, var_info=lidar_info, kind='traininglabel')
                Plot.save_figure(fig, name=f'histo_output_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_{lidar_info["normalization"]}.png', dpi=200)

            ####################################################################################################################################
            #  ___  ____ ____ _ _  _ ____    _  _ _  _ _    ___ _ _    ____ _   _ ____ ____    ___  ____ ____ ____ ____ ___  ___ ____ ____ _  _ 
            #  |  \ |___ |___ | |\ | |___    |\/| |  | |     |  | |    |__|  \_/  |___ |__/    |__] |___ |__/ |    |___ |__]  |  |__/ |  | |\ | 
            #  |__/ |___ |    | | \| |___    |  | |__| |___  |  | |___ |  |   |   |___ |  \    |    |___ |  \ |___ |___ |     |  |  \ |__| | \| 
            #                                                                                                                                  
            if use_mlp_model:
                # loop through hyperparameter space
                for dl, ls, af, il, op in itertools.product(DENSE_LAYERS, LAYER_SIZES, ACTIVATIONS, LOSS_FCNS, OPTIMIZERS):

                    hyper_params = {'DENSE_LAYERS': dl,
                                    'LAYER_SIZES': ls,
                                    'ACTIVATIONS': af,
                                    'LOSS_FCNS': il,
                                    'OPTIMIZER': op,
                                    'BATCH_SIZE': BATCH_SIZE,
                                    'EPOCHS': EPOCHS}

                    if not TRAINED_MODEL:
                        new_model_name = f'{dl}-dense-{ls}-nodes-{af}--{time_str}.h5'
                        hyper_params.update({'MODEL_PATH': MODELS_PATH + new_model_name,
                                             'LOG_PATH':   LOGS_PATH   + new_model_name})
                    else:
                        hyper_params.update({'MODEL_PATH': MODELS_PATH + TRAINED_MODEL,
                                             'LOG_PATH':   LOGS_PATH   + TRAINED_MODEL})

                    dense_model = Model.define_dense(n_input, n_output, hyper_params)
                    dense_model, history = Model.training(dense_model, train_set, train_label, hyper_params)

                    if plot_training_history:
                        fig, ax = Plot.History(history)
                        Plot.save_figure(fig, name=f'histo_output_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_{radar_info["normalization"]}.png', dpi=200)

                    if TRAINED_MODEL:
                        break  # if a trained model was given, jump out of hyperparameter loop

            ####################################################################################################################################
            #  ___  ____ ____ _ _  _ ____    ____ ____ _  _ _  _ ____ _    _  _ ___ _ ____ _  _ ____ _
            #  |  \ |___ |___ | |\ | |___    |    |  | |\ | |  | |  | |    |  |  |  | |  | |\ | |__| |
            #  |__/ |___ |    | | \| |___    |___ |__| | \|  \/  |__| |___ |__|  |  | |__| | \| |  | |___
            #
            if use_cnn_model:
                # loop through hyperparameter space
                for cl, dl, af, il, op, nf in itertools.product(CONV_LAYERS, DENSE_LAYERS, ACTIVATIONS, LOSS_FCNS, OPTIMIZERS, NFILTERS):

                    hyper_params = {'CONV_LAYERS': cl,
                                    'DENSE_LAYERS': dl,
                                    'NFILTERS': nf,
                                    'KERNEL_SIZE': KERNEL_SIZE,
                                    'POOL_SIZE':  POOL_SIZE,
                                    'ACTIVATIONS': af,
                                    'LOSS_FCNS': il,
                                    'OPTIMIZER': op,
                                    'BATCH_SIZE': BATCH_SIZE,
                                    'EPOCHS': EPOCHS}

                    if not TRAINED_MODEL:
                        new_model_name = f'{cl}-conv-{nf}-filter-{KERNEL_SIZE[0]}_{KERNEL_SIZE[1]}-kernelsize-{af}--{time_str}.h5'
                        hyper_params.update({'MODEL_PATH': MODELS_PATH + new_model_name,
                                             'LOG_PATH':   LOGS_PATH   + new_model_name})
                    else:
                        hyper_params.update({'MODEL_PATH': MODELS_PATH + TRAINED_MODEL,
                                             'LOG_PATH':   LOGS_PATH   + TRAINED_MODEL})

                    dense_model = Model.define_cnn(n_input, n_output, hyper_params)
                    dense_model, history = Model.training(dense_model, train_set, train_label, hyper_params)

                    if plot_training_history:
                        fig, ax = Plot.History(history)
                        Plot.save_figure(fig, name=f'histo_output_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_{radar_info["normalization"]}.png', dpi=300)

                    if TRAINED_MODEL:
                        break  # if a trained model was given, jump out of hyperparameter loop

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

                lidar_pred_container['bsc']['var'][lidar_pred_container['bsc']['var']<=0.0] = lidar_pred_container['bsc']['var_lims'][0]
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

            if TRAINED_MODEL:
                break  # if a trained model was given, jump out of hyperparameter loop

    Plot.print_elapsed_time(start_time, '\nDone, elapsed time = ')


