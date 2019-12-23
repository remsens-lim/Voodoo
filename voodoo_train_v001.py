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

import itertools
import time
import toml

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

import pyLARDA
import pyLARDA.helpers as h

import voodoo.libVoodoo.Loader as Loader
import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model
from   voodoo.model_ini import *

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
#
#               _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#               |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#               |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
########################################################################################################################################################
########################################################################################################################################################
if __name__ == '__main__':

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

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
        n_velocity_LR = radar_container['spectra'][0]['vel'].size
        n_time_LR     = radar_container['moments']['Ze']['ts'].size
        n_range_LR    = radar_container['moments']['Ze']['rg'].size
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

        lidar_list = ['attbsc1064']
        # interpolate polly xt data onto limrad grid (so that spectra can be used directly)
        lidar_container.update({f'{var}_ip': pyLARDA.Transformations.interpolate2d(
                lidar_container[var], new_time=ts_radar, new_range=rg_radar, method='nearest') for var in lidar_list})

        ########################################################################################################################################################
        #  _______  ______ _______ _____ __   _      _______ __   _ __   _
        #     |    |_____/ |_____|   |   | \  |      |_____| | \  | | \  |
        #     |    |    \_ |     | __|__ |  \_|      |     | |  \_| |  \_|
        #
        specta_interp    = Loader.equalize_rpg_radar_chirps(radar_container['spectra'])

        trainingset_settings = {'n_time': n_time_LR,  # number of time steps for LIMRAD94
                                'n_range': n_range_LR,  # number of range bins for LIMRAD94
                                'n_Dbins': n_velocity_LR,  # number of Doppler bins steps for LIMRAD94
                                'task': 'train',  # masks values for specific task
                                'add_moments': add_moments,  # if True adding radar moments to training set
                                'add_spectra': add_spectra,  # if True adding radar Doppler spectra to training set
                                'add_cwt': add_cwt,  # if True adding cont. wavelet transformation to training set
                                'add_lidar_float': add_lidar_float,  # if True use regression model
                                'add_lidar_binary': add_lidar_binary,  # if True use binary classification
                                'feature_info': radar_info,  # additional information about features
                                'feature_list': radar_list,  # list of feature variables names
                                'target_info': lidar_info,  # additional information about targets
                                'target_list': lidar_list,  # list of target variables names
                                'cwt': CWT_PARAMS}  # additional information about wavelet analysis

        train_set, train_label, list_ts, list_rg = Loader.load_trainingset(specta_interp, radar_container['moments'],
                                                                           lidar_container, **trainingset_settings)

        # get dimensionality of the feature and target space
        n_samples, n_input = train_set.shape[0], train_set.shape[1:]
        n_output  = train_label.shape[1:]

        print(f'min/max value in features = {np.min(train_set)},  maximum = {np.max(train_set)}')
        print(f'min/max value in targets  = {np.min(train_label)},  maximum = {np.max(train_label)}')

        ####################################################################################################################################
        #  ___  ____ ____ _ _  _ ____    _  _ _  _ _    ___ _ _    ____ _   _ ____ ____    ___  ____ ____ ____ ____ ___  ___ ____ ____ _  _
        #  |  \ |___ |___ | |\ | |___    |\/| |  | |     |  | |    |__|  \_/  |___ |__/    |__] |___ |__/ |    |___ |__]  |  |__/ |  | |\ |
        #  |__/ |___ |    | | \| |___    |  | |__| |___  |  | |___ |  |   |   |___ |  \    |    |___ |  \ |___ |___ |     |  |  \ |__| | \|
        #
        if use_mlp_regressen_model:
            # loop through hyperparameter space
            for dl, ls, af, il, op in itertools.product(DENSE_LAYERS, LAYER_SIZES, ACTIVATIONS, LOSS_FCNS, OPTIMIZERS):

                hyper_params = {'DENSE_LAYERS': dl,
                                'LAYER_SIZES': ls,
                                'ACTIVATIONS': af,
                                'ACTIVATIONS_OL': 'linear',
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
                dense_model, history = Model.training(dense_model, train_set, train_label, **hyper_params)

                fig, ax = Plot.History(history)
                Plot.save_figure(fig, name=f'histo_output_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_{radar_info["normalization"]}.png', dpi=200)

                if TRAINED_MODEL:
                    break  # if a trained model was given, jump out of hyperparameter loop

        ####################################################################################################################################
        #  ___  ____ ____ _ _  _ ____    ____ ____ _  _ _  _ ____ _    _  _ ___ _ ____ _  _ ____ _
        #  |  \ |___ |___ | |\ | |___    |    |  | |\ | |  | |  | |    |  |  |  | |  | |\ | |__| |
        #  |__/ |___ |    | | \| |___    |___ |__| | \|  \/  |__| |___ |__|  |  | |__| | \| |  | |___
        #
        if use_cnn_regression_model:
            # loop through hyperparameter space
            for cl, dl, dn, af, il, op, nf in itertools.product(CONV_LAYERS, DENSE_LAYERS, LAYER_SIZES, ACTIVATIONS, LOSS_FCNS, OPTIMIZERS, NFILTERS):

                hyper_params = {'CONV_LAYERS': cl,
                                'DENSE_LAYERS': dl,
                                'DENSE_NODES': dn,
                                'NFILTERS': nf,
                                'KERNEL_SIZE': KERNEL_SIZE,
                                'POOL_SIZE':  POOL_SIZE,
                                'ACTIVATIONS': af,
                                'ACTIVATIONS_OL': 'linear',
                                'LOSS_FCNS': il,
                                'OPTIMIZER': op,
                                'BATCH_SIZE': BATCH_SIZE,
                                'EPOCHS': EPOCHS,
                                'DEVICE': 0}

                if not TRAINED_MODEL:
                    new_model_name = f'{cl}-conv-{nf}-filter-{KERNEL_SIZE[0]}_{KERNEL_SIZE[1]}-kernelsize-{af}--{time_str}.h5'
                    hyper_params.update({'MODEL_PATH': MODELS_PATH + new_model_name,
                                         'LOG_PATH':   LOGS_PATH   + new_model_name})
                else:
                    hyper_params.update({'MODEL_PATH': MODELS_PATH + TRAINED_MODEL,
                                         'LOG_PATH':   LOGS_PATH   + TRAINED_MODEL})

                dense_model = Model.define_cnn(n_input, n_output, hyper_params)
                dense_model, history = Model.training(dense_model, train_set, train_label, hyper_params)

                fig, ax = Plot.History(history)
                Plot.save_figure(fig, name=f'histo_output_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_{radar_info["normalization"]}.png', dpi=300)

                if TRAINED_MODEL:
                    break  # if a trained model was given, jump out of hyperparameter loop



        ####################################################################################################################################
        #  ___  ____ ____ _ _  _ ____    ____ ____ _  _ _  _ ____ _    _  _ ___ _ ____ _  _ ____ _
        #  |  \ |___ |___ | |\ | |___    |    |  | |\ | |  | |  | |    |  |  |  | |  | |\ | |__| |
        #  |__/ |___ |    | | \| |___    |___ |__| | \|  \/  |__| |___ |__|  |  | |__| | \| |  | |___ classifier
        #
        if use_cnn_classfier_model:
            # loop through hyperparameter space
            for cl, dl, dn, af, il, op, nf in itertools.product(CONV_LAYERS, DENSE_LAYERS, LAYER_SIZES, ACTIVATIONS, LOSS_FCNS, OPTIMIZERS, NFILTERS):

                hyper_params = {'CONV_LAYERS': cl,
                                'DENSE_LAYERS': dl,
                                'DENSE_NODES': dn,
                                'NFILTERS': nf,
                                'KERNEL_SIZE': KERNEL_SIZE,
                                'POOL_SIZE':  POOL_SIZE,
                                'ACTIVATIONS': af,
                                'ACTIVATIONS_OL': 'softmax',
                                'LOSS_FCNS': il,
                                'OPTIMIZER': op,
                                'BATCH_SIZE': BATCH_SIZE,
                                'EPOCHS': EPOCHS,
                                'DEVICE': 0}

                if not TRAINED_MODEL:
                    new_model_name = f'{cl}-conv-{nf}-filter-{KERNEL_SIZE[0]}_{KERNEL_SIZE[1]}-kernelsize-{af}--{time_str}.h5'
                    hyper_params.update({'MODEL_PATH': MODELS_PATH + new_model_name,
                                         'LOG_PATH':   LOGS_PATH   + new_model_name})
                else:
                    hyper_params.update({'MODEL_PATH': MODELS_PATH + TRAINED_MODEL,
                                         'LOG_PATH':   LOGS_PATH   + TRAINED_MODEL})

                dense_model = Model.define_cnn(n_input, n_output, hyper_params)
                dense_model, history = Model.training(dense_model, train_set, train_label, hyper_params)

                fig, ax = Plot.History(history)
                Plot.save_figure(fig, name=f'histo_output_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}_{radar_info["normalization"]}.png', dpi=300)

                if TRAINED_MODEL:
                    break  # if a trained model was given, jump out of hyperparameter loop

    Plot.print_elapsed_time(start_time, '\nDone, elapsed time = ')


