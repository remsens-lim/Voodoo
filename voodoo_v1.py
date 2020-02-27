#!/home/sdig/anaconda3/bin/python
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

import datetime
import itertools
import logging
import os
import sys
import time

import numpy as np
import toml
from scipy.io import loadmat

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'
sys.path.append('../larda/')

import pyLARDA.helpers as h

import pyLARDA.Transformations as tr

import voodoo.libVoodoo.Loader_v2 as Loader
import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def get_logger(logger_list, status='info'):
    log = []
    for ilog in logger_list:
        log_ = logging.getLogger(ilog)
        if status == 'info':
            log_.setLevel(logging.INFO)
        elif status == 'debug':
            log_.setLevel(logging.DEBUG)
        elif status == 'critical':
            log_.setLevel(logging.CRITICAL)
        log_.addHandler(logging.StreamHandler())
        log.append(log_)

    return log


def log_dimensions(spec_dim, cn_dim, *args):
    loggers[0].info(f'Radar-Input  :: LIMRAD94\n'
                    f'      (n_ts, n_rg, n_vel) = ({spec_dim["n_ts"]}, {spec_dim["n_rg"]}, {spec_dim["n_vel"]})')
    loggers[0].info(f'Target-Input :: Cloudnetpy94\n'
                    f'      (n_ts, n_rg)        = ({cn_dim["n_ts"]}, {cn_dim["n_rg"]})')
    if len(args) > 0:
        loggers[0].info(f'Target-Input :: PollyNET\n'
                        f'      (n_ts, n_rg)        = ({args[0]["n_ts"]}, {args[0]["n_rg"]})')


def create_filename(modelname, **kwargs):
    if not TRAINED_MODEL:
        name = f'{kwargs["CONV_LAYERS"]}-cl--' \
               f'{kwargs["KERNEL_SIZE"][0]}_{kwargs["KERNEL_SIZE"][1]}-ks--' \
               f'{kwargs["ACTIVATIONS"]}-af--' \
               f'{kwargs["OPTIMIZER"]}-opt--' \
               f'{kwargs["LOSS_FCNS"]}-loss--' \
               f'{kwargs["BATCH_SIZE"]}-bs--' \
               f'{kwargs["EPOCHS"]}-ep--' \
               f'{kwargs["LEARNING_RATE"]}-lr--' \
               f'{kwargs["DECAY_RATE"]}-dr--' \
               f'{kwargs["DENSE_LAYERS"]}-dl--' \
               f'{kwargs["DENSE_NODES"]}-dn--' \
               f'{time_str}.h5'
        return {'MODEL_PATH': MODELS_PATH + name, 'LOG_PATH': LOGS_PATH + name}
    else:
        return {'MODEL_PATH': MODELS_PATH + modelname, 'LOG_PATH': LOGS_PATH + modelname}


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

    CASE_LIST = '/home/sdig/code/larda3/case2html/dacapo_case_studies.toml'
    NCPATH = '/home/sdig/code/larda3/voodoo/nc-files/spectra/'
    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'

    # gather todays date
    t0_voodoo = datetime.datetime.today()
    time_str = f'{t0_voodoo:%Y%m%d-%H%M%S}'

    # load ann model parameter and other global values
    config_global_model = toml.load(VOODOO_PATH + 'ann_model_setting.toml')

    # BOARD_NAME = f'voodoo-board-{time_str}_{BATCH_SIZE}-bachsize-{EPOCHS}-epochs'
    LOGS_PATH = f'{VOODOO_PATH}logs/'
    MODELS_PATH = f'{VOODOO_PATH}models/'
    PLOTS_PATH = f'{VOODOO_PATH}plots/'

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    if 'model' in kwargs:
        TRAINED_MODEL = kwargs['model']
    else:
        # TRAINED_MODEL = '3-conv-4_4-kernelsize-relu--20200227-030455.h5'
        # TRAINED_MODEL = '3-conv-3_3-kernelsize-relu--20200227-004401.h5'
        # TRAINED_MODEL = '3-conv-3_3-kernelsize-relu--20200227-013838.h5'
        TRAINED_MODEL = ''

    if 'case' in kwargs:
        case_string = kwargs['case']
    else:
        # load the case
        # case_string = '20190801-01'
        case_string = '20190410-02'
        # case_string = '20190208-01'

        # case_string = '20190304-02' # good trainingset
        # case_string = '20190904-03'  # best trainingset !!!
        # case_string = '20190904-01'  # best trainingset !!!
        # case_string = '20190801-03'

    if 'task' in kwargs:
        TASK = kwargs['task']
    else:
        TASK = 'train'

    # get all loggers
    loggers = get_logger(['libVoodoo'])

    # gather time interval, etc.
    case = Loader.load_case_list(CASE_LIST, case_string)
    dt_interval = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
    begin_dt, end_dt = dt_interval
    dt_string = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}'

    ########################################################################################################################################################
    #   _    ____ ____ ___     ____ ____ ___  ____ ____    ___  ____ ___ ____
    #   |    |  | |__| |  \    |__/ |__| |  \ |__| |__/    |  \ |__|  |  |__|
    #   |___ |__| |  | |__/    |  \ |  | |__/ |  | |  \    |__/ |  |  |  |  |
    #
    #
    radar_input_setting = config_global_model['feature']['info']['VSpec']
    tfp = config_global_model['tensorflow']


    h.change_dir(NCPATH)

    radar_container = {'VSpec': loadmat(f'{dt_string}_limrad94_spectra.mat')}
    radar_container.update({'SLv': loadmat(f'{dt_string}_limrad94_spectra_SLv.mat')})
    radar_container.update(loadmat(f'{dt_string}_limrad94_spectra_extra.mat'))

    print(f'loading spectra done :: {dt_string}_limrad94_spectra.mat')

    dim_input = {

        'spectra':
            {'n_ts': int(radar_container['n_ts']),
             'n_rg': int(radar_container['n_rg']),
             'n_ch': int(radar_container['n_ch']),
             'n_vel': int(radar_container['n_vel'])
             }
    }
    ts_radar = radar_container['VSpec']['ts'].reshape(dim_input['spectra']['n_ts'])
    rg_radar = radar_container['VSpec']['rg'].reshape(dim_input['spectra']['n_rg'])

    ########################################################################################################################################################
    #   _    ____ ____ ___     ____ _    ____ _  _ ___  _  _ ____ ___    ___  ____ ___ ____
    #   |    |  | |__| |  \    |    |    |  | |  | |  \ |\ | |___  |     |  \ |__|  |  |__|
    #   |___ |__| |  | |__/    |___ |___ |__| |__| |__/ | \| |___  |     |__/ |  |  |  |  |
    #
    #                                                                                           
    target_container = {

        'cn_class': loadmat(f'{dt_string}_cloudnetpy94_class.mat'),

        'cn_status': loadmat(f'{dt_string}_cloudnetpy94_status.mat'),

    }

    dim_target = {

        'cn_class':
            {'n_ts': int(target_container['cn_class']['n_ts']),
             'n_rg': int(target_container['cn_class']['n_rg'])},

        # 'pn_class':
        #    {'n_ts': int(target_container['pn_class']['n_ts']),
        #     'n_rg': int(target_container['pn_class']['n_rg'])}
    }

    # log the input dimensions (quick check for errors)
    log_dimensions(dim_input["spectra"], dim_target["cn_class"])

    dt_radar = h.ts_to_dt(radar_container['VSpec']['ts'][0, 0]), h.ts_to_dt(radar_container['VSpec']['ts'][0, -1])
    dt_cn = h.ts_to_dt(target_container['cn_class']['ts'][0, 0]), h.ts_to_dt(target_container['cn_class']['ts'][0, -1])
    # dt_pn = h.ts_to_dt(target_container['pn_class']['ts'][0, 0]), h.ts_to_dt(target_container['pn_class']['ts'][0, -1])

    print('radar    - first/last dt :: ', dt_radar, [radar_container['VSpec']['rg'][0, 0], radar_container['VSpec']['rg'][0, -1]])
    print('cloudnet - first/last dt :: ', dt_cn, [target_container['cn_class']['rg'][0, 0], target_container['cn_class']['rg'][0, -1]])
    # print('pollynet - first/last dt :: ', dt_pn, [target_container['pn_class']['rg'][0, 0], target_container['pn_class']['rg'][0, -1]])

    ############################################################################################################################################################
    #   _    ____ ____ ___     ___ ____ ____ _ _  _ _ _  _ ____ ____ ____ ___
    #   |    |  | |__| |  \     |  |__/ |__| | |\ | | |\ | | __ [__  |___  |
    #   |___ |__| |  | |__/     |  |  \ |  | | | \| | | \| |__] ___] |___  |
    #

    feature_set, target_labels, masked = Loader.load_data(

        radar_container['VSpec'],

        target_container['cn_class'],

        target_container['cn_status'],

        task = TASK,

        **config_global_model['feature']['info']

    )

    ########################################################################################################################################################
    #   ___ ____ ____ _ _  _ _ _  _ ____
    #    |  |__/ |__| | |\ | | |\ | | __
    #    |  |  \ |  | | | \| | | \| |__]
    #
    #
    if TASK == 'train':
        # loop through hyperparameter space
        for cl, dl, af, il, op in itertools.product(tfp['CONV_LAYERS'], tfp['DENSE_LAYERS'], tfp['ACTIVATIONS'], tfp['LOSS_FCNS'], tfp['OPTIMIZERS']):

            hyper_params = {

                # Convolutional part of the model
                'CONV_LAYERS': cl,
                'NFILTERS': tfp['NFILTERS'],
                'KERNEL_SIZE': tfp['KERNEL_SIZE'],
                'POOL_SIZE': tfp['POOL_SIZE'],
                'ACTIVATIONS': af,

                # fully connected layers
                'DENSE_LAYERS': dl,
                'DENSE_NODES': tfp['DENSE_NODES'],
                'ACTIVATIONS_OL': tfp['ACTIVATIONS_OL'],
                'LOSS_FCNS': il,
                'OPTIMIZER': op,

                # traning settings
                'BATCH_SIZE': tfp['BATCH_SIZE'],
                'EPOCHS': tfp['EPOCHS'],
                'LEARNING_RATE': tfp['LEARNING_RATE'],
                'DECAY_RATE': tfp['DECAY_RATE'],
                'MOMENTUM': tfp['MOMENTUM'],

                # GPU
                'DEVICE': 0
            }

            # create file name and add MODEL_PATH and LOGS_PATH to hyper_parameter dict
            hyper_params.update(create_filename(TRAINED_MODEL, **hyper_params))

            # define a new model or load an existing one
            cnn_model = Model.define_cnn(feature_set.shape[1:], target_labels.shape[1:], **hyper_params)

            # parse the training set to the optimizer
            history = Model.training(cnn_model, feature_set, target_labels, **hyper_params)

            # create directory for plots
            h.change_dir(f'{PLOTS_PATH}/training/{begin_dt:%Y%m%d-%H%M}-{end_dt:%H%M}/')

            fig, _ = Plot.History(history)
            Plot.save_figure(fig, name=f'histo_loss-acc_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png', dpi=300)

            if TRAINED_MODEL:
                break  # if a trained model was given, jump out of hyperparameter loop

    ############################################################################################################################################################
    #   ___  ____ ____ ___  _ ____ ___ _ ____ _  _
    #   |__] |__/ |___ |  \ | |     |  | |  | |\ |
    #   |    |  \ |___ |__/ | |___  |  | |__| | \|
    #
    #
    if TASK == 'predict':

        hyper_params = {

            # Objective measures
            'LOSS_FCNS': tfp['LOSS_FCNS'][0],
            'OPTIMIZERS': tfp['OPTIMIZERS'],

            # traning settings
            'BATCH_SIZE': tfp['BATCH_SIZE'],
            'EPOCHS': tfp['EPOCHS'],

            'DEVICE': 1

        }

        # define a new model or load an existing one
        cnn_model = Model.define_cnn(feature_set.shape[1:], target_labels.shape[1:], MODEL_PATH=MODELS_PATH + TRAINED_MODEL, **hyper_params)

        # make predictions
        cnn_pred = Model.predict_liquid(cnn_model, feature_set)

        ts_rg_dim = (dim_target['cn_class']['n_ts'], dim_target['cn_class']['n_rg'])
        prediction2D = np.zeros(ts_rg_dim, dtype=np.float32)
        prediction2D_classes = np.full(ts_rg_dim, False)

        cnt = 0
        for iT in range(dim_target["cn_class"]["n_ts"]):
            for iR in range(dim_target["cn_class"]["n_rg"]):
                if masked[iT, iR]: continue
                # [np.where(r == 1)[0][0] for r in a]
                if cnn_pred[cnt, 1] > 0.5: prediction2D_classes[iT, iR] = True
                prediction2D[iT, iR] = cnn_pred[cnt, 1]
                cnt += 1

        prediction_container = h.put_in_container(prediction2D, target_container['cn_class'])  # , **kwargs)
        prediction_container['dimlabel'] = ['time', 'range']
        prediction_container['name'] = 'prediction'
        prediction_container['joints'] = ''
        prediction_container['rg_unit'] = 'm'
        prediction_container['colormap'] = 'coolwarm'
        # ZE['paraminfo'] = dict(ZE['paraminfo'][0])
        prediction_container['system'] = 'ANN'
        prediction_container['ts'] = np.squeeze(prediction_container['ts'])
        prediction_container['rg'] = np.squeeze(prediction_container['rg'])
        prediction_container['var_lims'] = [prediction_container['var'].min(), prediction_container['var'].max()]
        prediction_container['var_unit'] = 'likelihood'
        prediction_container['mask'] = masked

        # create directory for plots
        h.change_dir(f'{PLOTS_PATH}/training/{dt_string}/')
        fig, _ = tr.plot_timeheight(prediction_container, title=f'preliminary results (ANN prediction) {dt_string}')  # , **plot_settings)
        Plot.save_figure(fig, name=f'prediction_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=200)

    ####################################################################################################################################
    Plot.print_elapsed_time(start_time, '\nDone, elapsed time = ')
