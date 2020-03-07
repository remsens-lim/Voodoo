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

import logging
import os
import pprint
import numpy as np
import sys
import datetime

import itertools
import time
import toml

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
__version__ = "0.1.1"
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
        TRAINED_MODEL = '3-conv-4_4-kernelsize-relu--20200227-030455.h5'
        # TRAINED_MODEL = '3-conv-3_3-kernelsize-relu--20200227-004401.h5'
        # TRAINED_MODEL = '3-conv-3_3-kernelsize-relu--20200227-013838.h5'

    if 'case' in kwargs or 'date' in kwargs:
        case_string = kwargs['case']
    else:
        # load the case
        case_string = '20190313-01'
        #case_string = '20190801-01'
        #case_string = '20190410-03'
        #case_string = '20190304-02'

    case = Loader.load_case_list(CASE_LIST, case_string)
    # get all loggers
    loggers = get_logger(['libVoodoo'])
    dt_interval = [datetime.datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
    begin_dt, end_dt = dt_interval
    dt_string = f'{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}'


    ########################################################################################################################################################
    #   ______ _______ ______  _______  ______      _____ __   _  _____  _     _ _______
    #  |_____/ |_____| |     \ |_____| |_____/        |   | \  | |_____] |     |    |
    #  |    \_ |     | |_____/ |     | |    \_      __|__ |  \_| |       |_____|    |
    #

    radar_input_setting = config_global_model['feature']['info']['VSpec']
    tfp = config_global_model['tensorflow']

    from scipy.io import loadmat

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
    #  _______ _______  ______  ______ _______ _______      _____ __   _  _____  _     _ _______
    #     |    |_____| |_____/ |  ____ |______    |           |   | \  | |_____] |     |    |   
    #     |    |     | |    \_ |_____| |______    |         __|__ |  \_| |       |_____|    |   
    #                                                                                           

    target_container = {

        'cn_class': loadmat(f'{dt_string}_cloudnetpy94_class.mat'),

        'cn_status': loadmat(f'{dt_string}_cloudnetpy94_status.mat'),

    }

    dim_target = {

        'cn_class':
            {'n_ts': int(target_container['cn_class']['n_ts']),
             'n_rg': int(target_container['cn_class']['n_rg'])},

        #'pn_class':
        #    {'n_ts': int(target_container['pn_class']['n_ts']),
        #     'n_rg': int(target_container['pn_class']['n_rg'])}
    }

    loggers[0].info(f'Radar-Input  :: LIMRAD94\n'
                    f'      (n_ts, n_rg, n_vel) = ({dim_input["spectra"]["n_ts"]}, {dim_input["spectra"]["n_rg"]}, {dim_input["spectra"]["n_vel"]})')
    loggers[0].info(f'Target-Input :: Cloudnetpy94\n'
                    f'      (n_ts, n_rg)        = ({dim_target["cn_class"]["n_ts"]}, {dim_target["cn_class"]["n_rg"]})')
    #loggers[0].info(f'Target-Input :: PollyNET\n
    #                f'      (n_ts, n_rg)        = ({dim_target["pn_class"]["n_ts"]}, {dim_target["pn_class"]["n_rg"]})')

    dt_radar = h.ts_to_dt(radar_container['VSpec']['ts'][0, 0]), h.ts_to_dt(radar_container['VSpec']['ts'][0, -1])
    dt_cn = h.ts_to_dt(target_container['cn_class']['ts'][0, 0]), h.ts_to_dt(target_container['cn_class']['ts'][0, -1])
    #dt_pn = h.ts_to_dt(target_container['pn_class']['ts'][0, 0]), h.ts_to_dt(target_container['pn_class']['ts'][0, -1])

    print('radar    - first/last dt :: ', dt_radar, [radar_container['VSpec']['rg'][0, 0], radar_container['VSpec']['rg'][0, -1]])
    print('cloudnet - first/last dt :: ', dt_cn, [target_container['cn_class']['rg'][0, 0], target_container['cn_class']['rg'][0, -1]])
    #print('pollynet - first/last dt :: ', dt_pn, [target_container['pn_class']['rg'][0, 0], target_container['pn_class']['rg'][0, -1]])

    ########################################################################################################################################################
    #  _______  ______ _______ _____ __   _      _______ __   _ __   _
    #     |    |_____/ |_____|   |   | \  |      |_____| | \  | | \  |
    #     |    |    \_ |     | __|__ |  \_|      |     | |  \_| |  \_|
    #
    """
    This is just for orientation
    
    " status
    \nValue 0: Clear sky.
    \nValue 1: Good radar and lidar echos.
    \nValue 2: Good radar echo only.
    \nValue 3: Radar echo, corrected for liquid attenuation.
    \nValue 4: Lidar echo only.
    \nValue 5: Radar echo, uncorrected for liquid attenuation.
    \nValue 6: Radar ground clutter.
    \nValue 7: Lidar clear-air molecular scattering.";

    " classes
    \nValue 0: Clear sky.
    \nValue 1: Cloud liquid droplets only.
    \nValue 2: Drizzle or rain.
    \nValue 3: Drizzle or rain coexisting with cloud liquid droplets.
    \nValue 4: Ice particles.
    \nValue 5: Ice coexisting with supercooled liquid droplets.
    \nValue 6: Melting ice particles.
    \nValue 7: Melting ice particles coexisting with cloud liquid droplets.
    \nValue 8: Aerosol particles, no cloud or precipitation.
    \nValue 9: Insects, no cloud or precipitation.
    \nValue 10: Aerosol coexisting with insects, no cloud or precipitation.";
    """
    # use only good radar & lidar echos
    #masked_cloudnet = np.squeeze(target_container['cn_status']['var']) != 2
    masked_radar_ip = np.squeeze(np.all(radar_container['VSpec']['mask'], axis=2))
    #masked_scl_class = np.squeeze(target_container['cn_class']['var']) != 5
    #masked_cdrop_class = np.squeeze(target_container['cn_class']['var']) != 1
    masked = masked_radar_ip # + masked_cloudnet

    #masked[~masked_scl_class] = False   # add scl from the non-masked values
    #masked[~masked_cdrop_class] = False   # add scl from the non-masked values

    quick_check = False
    if quick_check:
        # create directory for plots
        h.change_dir(f'{PLOTS_PATH}/training/{dt_string}/')
        ZE = np.sum(radar_container['VSpec']['var'], axis=2)
        ZE = h.put_in_container(ZE, radar_container['SLv'])#, **kwargs)
        ZE['dimlabel'] = ['time', 'range']
        ZE['name'] = ZE['name'][0]
        ZE['joints'] = ZE['joints'][0]
        ZE['rg_unit'] = ZE['rg_unit'][0]
        ZE['colormap'] = ZE['colormap'][0]
        #ZE['paraminfo'] = dict(ZE['paraminfo'][0])
        ZE['system'] = ZE['system'][0]
        ZE['ts'] = np.squeeze(ZE['ts'])
        ZE['rg'] = np.squeeze(ZE['rg'])
        ZE['var_lims'] = [-60, 20]
        ZE['var_unit'] = 'dBZ'
        ZE['mask'] = masked

        fig, _ = tr.plot_timeheight(ZE, z_converter='lin2z', title='bla')#, **plot_settings)
        Plot.save_figure(fig, name=f'mitscllimrad_ZeLV0_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=200)


    valtarget = np.zeros(target_container['cn_class']['var'].shape, dtype=np.float32)
    valtarget[~masked] = 1.0
    feature_set, target_labels = Loader.load_trainingset(radar_container['VSpec'],
                                                         valtarget,
                                                         masked,
                                                         SLv=radar_container['SLv'],
                                                         **config_global_model['feature']['info'])


    # get dimensionality of the feature and target space
    n_samples, n_input = feature_set.shape[0], feature_set.shape[1:]
    n_output = target_labels.shape[1:]

    print(f'min/max value in features = {np.min(feature_set)},  maximum = {np.max(feature_set)}')
    print(f'min/max value in targets  = {np.min(target_labels)},  maximum = {np.max(target_labels)}')

    ####################################################################################################################################
    #  ___  ____ ____ _ _  _ ____    ____ ____ _  _ _  _ ____ _    _  _ ___ _ ____ _  _ ____ _
    #  |  \ |___ |___ | |\ | |___    |    |  | |\ | |  | |  | |    |  |  |  | |  | |\ | |__| |
    #  |__/ |___ |    | | \| |___    |___ |__| | \|  \/  |__| |___ |__|  |  | |__| | \| |  | |___ classifier
    #
    use_cnn_classfier_model = True

    if use_cnn_classfier_model:
        # loop through hyperparameter space

        hyper_params = {'ACTIVATIONS': tfp['ACTIVATIONS'],
                        'ACTIVATIONS_OL': 'softmax',
                        'LOSS_FCNS': tfp['LOSS_FCNS'][0],
                        'OPTIMIZERS': tfp['OPTIMIZERS'],
                        'BATCH_SIZE': tfp['BATCH_SIZE'],
                        'EPOCHS': tfp['EPOCHS'],
                        'DEVICE': 1}

        cnn_model = Model.define_cnn(n_input, n_output, MODEL_PATH=MODELS_PATH + TRAINED_MODEL, **hyper_params)
        cnn_pred = Model.predict_liquid(cnn_model, feature_set)

        prediction2D = np.zeros(masked.shape, dtype=np.float32)

        cnt = 0
        for iT in range(dim_target["cn_class"]["n_ts"]):
            for iR in range(dim_target["cn_class"]["n_rg"]):
                if masked[iT, iR]: continue
                print([np.where(r == 1)[cnt][0] for r in cnn_pred])
                #prediction2D[iT, iR] = [np.where(r == 1)[cnt][0] for r in cnn_pred]
                cnt += 1

        prediction_container = h.put_in_container(prediction2D, target_container['cn_class'])  # , **kwargs)
        prediction_container['dimlabel'] = ['time', 'range']
        prediction_container['name'] = 'prediction'
        prediction_container['joints'] = ''
        prediction_container['rg_unit'] = 'm'
        prediction_container['colormap'] = 'ann_target'
        #ZE['paraminfo'] = dict(ZE['paraminfo'][0])
        prediction_container['system'] = 'ANN'
        prediction_container['ts'] = np.squeeze(prediction_container['ts'])
        prediction_container['rg'] = np.squeeze(prediction_container['rg'])
        prediction_container['var_lims'] = [prediction_container['var'].min(), prediction_container['var'].max()]
        prediction_container['var_unit'] = 'likelihood'
        prediction_container['mask'] = masked

        # create directory for plots
        h.change_dir(f'{PLOTS_PATH}/training/{dt_string}/')
        fig, _ = tr.plot_timeheight(prediction_container, title=f'preliminary results (ANN prediction) {dt_string}')#, **plot_settings)
        Plot.save_figure(fig, name=f'prediction_{begin_dt:%Y%m%d_%H%M}-{end_dt:%H%M}.png', dpi=200)

        dummy=5



    ####################################################################################################################################
    Plot.print_elapsed_time(start_time, '\nDone, elapsed time = ')
