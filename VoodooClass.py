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
import logging
import numpy as np
import os
import time
import toml
from json2html import *

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'

sys.path.append('../larda/')
import matplotlib
import matplotlib.pyplot as plt

import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

import voodoo.libVoodoo.Plot   as Plot
import voodoo.libVoodoo.Model  as Model
import voodoo.libVoodoo.Utils  as Utils

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2020, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.1.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

# list of cloudnet data sets used for training
CLOUDNETs = ['CLOUDNETpy94']



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


# get all loggers
loggers = get_logger(['libVoodoo'], status='info')


def _log_number_of_classes(labels, text=''):
    # numer of samples per class afer removing ice
    class_n_distribution = {
        'Clear sky': 0,
        'Cloud liquid droplets only': 0,
        'Drizzle or rain.': 0,
        'Drizzle/rain & cloud droplet': 0,
        'Ice particles.': 0,
        'Ice coexisting with supercooled liquid droplets.': 0,
        'Melting ice particles': 0,
        'Melting ice & cloud droplets': 0,
        'Aerosol': 0,
        'Insects': 0,
        'Aerosol and Insects': 0,
    }
    loggers[0].info(text)
    loggers[0].info(f'{labels.shape[0]:12d}   total')
    for i, key in enumerate(class_n_distribution.keys()):
        n = np.sum(labels == i)
        loggers[0].info(f'{n:12d}   {key}')
        class_n_distribution[key] = n
    return class_n_distribution


def _init_cnn_setup(tf_settings, feature_setting, feature_set, target_labels, models_path='', model_path='', logs_path=''):
    cnn_parameters = {

        # Convolutional part of the model
        'KIND': 'HSI',
        'CONV_DIMENSION': tf_settings['USE_MODEL'],

        # I/O dimensions
        'INPUT_DIMENSION': feature_set.shape,
        'OUTPUT_DIMENSION': target_labels.shape,

        # time of creation
        'time_str': f'{datetime.datetime.today():%Y%m%d-%H%M%S}',

    }

    # create file name and add MODEL_PATH and LOGS_PATH to hyper_parameter dict
    model_name = f"{cnn_parameters['time_str']}_ann-model-weights_{cnn_parameters['CONV_DIMENSION']}.h5" if len(model_path) == 0 else model_path
    cnn_parameters.update({
        'MODEL_NAME': model_name,
        'MODEL_PATH': f'{models_path}/{model_name}',
        'LOG_PATH': logs_path,
        #'cloudnet': CLOUDNETs,
        **tf_settings,
        **feature_setting
    })

    Utils.write_ann_config_file(
        name=model_name.replace('.h5', '.json'),
        path=models_path,
        **cnn_parameters
    )
    return cnn_parameters


class Voodoo():

    def __init__(self, voodoo_path='', radar='limrad94', ann_model_toml='', **kwargs2):
        # gather command line arguments
        method_name, args, kwargs = Utils.read_cmd_line_args(sys.argv)

        self.RADAR = radar
        self.TASK = kwargs['task'] if 'task' in kwargs else 'train'
        self.CLOUDNET = kwargs['cloudnet'] if 'cloudnet' in kwargs else 'CLOUDNETpy94'
        self.LOGS_PATH = f'{voodoo_path}/logs/'
        self.DATA_PATH = f'{voodoo_path}/data/'
        self.MODELS_PATH = f'{voodoo_path}/models/'
        self.MODEL_NAME = kwargs['model'] + ' ' + args[0][:] if len(args) > 0 else kwargs['model'] if 'model' in kwargs else ''
        self.MODEL_PATH = f'{self.MODELS_PATH}/{self.MODEL_NAME}'
        self.PLOTS_PATH = f'{voodoo_path}/plots/'
        self.ANN_MODEL_TOML = ann_model_toml
        self.CASE = kwargs['case'] if 'case' in kwargs else ''
        self.n_classes = {}

        if self.TASK == 'predict' and not os.path.isfile(f'{self.MODEL_PATH}'):
            raise FileNotFoundError(f'Trained model not found! {self.MODEL_PATH}')

        if len(self.CASE) == 17:
            self.data_chunk_toml = f'{voodoo_path}/tomls/auto-trainingset-{self.CASE}.toml'
            self.data_chunk_heads = [chunk for chunk in Utils.load_case_file(self.data_chunk_toml).keys()]
        else:
            raise ValueError('Check keyword argument "case" ! (format(string): YYYYMMDD-YYYYMMDD')

        # load ann model parameter and other global values
        config_global_model = toml.load(voodoo_path + self.ANN_MODEL_TOML)
        self.feature_setting = config_global_model['feature']
        self.tf_settings = config_global_model['tensorflow']

        loggers[0].info(f'\nLoading {self.tf_settings["USE_MODEL"]} neural network input......')

        # self.cloudnet_data_kwargs = {
        self.feature_selector_settings = {
            'VOODOO_PATH': VOODOO_PATH,  # NONSENSE PATH
            # 'SAVE': True,
            'remove_ice': kwargs2['remove_ice'] if 'remove_ice' in kwargs2 else 0.,
            'remove_drizzle':  kwargs2['remove_drizzle'] if 'remove_drizzle' in kwargs2 else 0.,
            'n_validation': kwargs2['n_val'] if 'n_val' in kwargs2 else 0.,
        }

    def import_dataset(self, data_root='', **kwargs):

        def remove_randomely(rm_class, _classnr, _feature_set, _target_labels):
            if 100.0 > self.feature_selector_settings[rm_class] > 0:
                idx = np.where(_target_labels == _classnr)[0]
                rand_choice = np.random.choice(idx, int(idx.size * self.feature_selector_settings[rm_class] / 100.))
                _feature_set = np.delete(_feature_set, rand_choice, axis=0)
                _target_labels = np.delete(_target_labels, rand_choice, axis=0)
                self.n_classes.update({
                    f'n_samples_{rm_class}': _log_number_of_classes(
                    _target_labels,
                    text=f'\nsamples per class after removing {self.feature_selector_settings[rm_class]:.2f}% of {rm_class}')
                })
            return _feature_set, _target_labels

        cloudnet_data = [
            Utils.load_dataset_from_zarr(
                self.data_chunk_heads, self.data_chunk_toml,
                DATA_PATH=f'{data_root}/{self.CLOUDNET}',
                CLOUDNET=self.CLOUDNET, **kwargs,
                RADAR=self.RADAR,
                add_flipped=self.feature_setting['VSpec']['add_flipped'],
                CDIM=self.tf_settings['USE_MODEL'],
                TASK=self.TASK,
            )
        ]

        feature_set = np.concatenate([i for icn in cloudnet_data for i in icn[0]], axis=0)
        target_labels = np.concatenate([i for icn in cloudnet_data for i in icn[1]], axis=0)

        # concatenate classes and mask for plotting
        if self.TASK == 'predict':
            _cn = cloudnet_data[0]
            cloudnet_class = np.concatenate(_cn[2], axis=0)
            cloudnet_status = np.concatenate(_cn[3], axis=0)
            masked_total = np.concatenate(_cn[4], axis=0)
            model_temp = np.concatenate(_cn[5], axis=0)
            cloudnet_ts = np.concatenate(_cn[6], axis=0)
            cloudnet_rg = _cn[7]
        else:
            cloudnet_class = None
            cloudnet_status = None
            masked_total = None
            model_temp = None
            cloudnet_ts = None
            cloudnet_rg = None

        _log_number_of_classes(target_labels, text=f'\nsamples per class')

        validation_set = ()
        if self.TASK == 'train':
            feature_set, target_labels = remove_randomely('remove_ice', 4, feature_set, target_labels)
            feature_set, target_labels = remove_randomely('remove_drizzle', 2, feature_set, target_labels)

            # splitting into training and validation set, use every n-th element from the training set for validation
            N_VAL = self.feature_selector_settings['n_validation']
            validation_set = (feature_set[::N_VAL], target_labels[::N_VAL])
            feature_set = np.array([item for index, item in enumerate(feature_set) if (index + 1) % N_VAL != 0])
            target_labels = np.array([item for index, item in enumerate(target_labels) if (index + 1) % N_VAL != 0])
            _log_number_of_classes(target_labels, text=f'\nsamples per class after removing 1 in {N_VAL} values for the validation split')

        self.dataset = {
            'feature_set': feature_set,
            'target_labels': np.squeeze(target_labels),
            'validation_set': validation_set,
            'cloudnet_class': cloudnet_class,
            'cloudnet_status': cloudnet_status,
            'mask': masked_total,
            'model_temp': model_temp,
            'cloudnet_ts': cloudnet_ts,
            'cloudnet_rg': cloudnet_rg,
        }

    def Aftermath(self, pred_class, pred_probs):
        self.case_plot_path = f'{self.PLOTS_PATH}/training/{self.CASE}/'
        h.change_dir(self.case_plot_path)

        cloudnet_data_available = self.dataset['cloudnet_class'].size > 0

        if cloudnet_data_available:
            self.contour_T = Utils.get_isotherms(
                self.dataset['model_temp'],
                self.dataset['cloudnet_ts'],
                self.dataset['cloudnet_rg'],
                self.dataset['mask'],
                name='Temperature'
            )
            cloudnet_status_container = Utils.variable_to_container(
                self.dataset['cloudnet_status'],
                self.dataset['cloudnet_ts'],
                self.dataset['cloudnet_rg'],
                self.dataset['mask'],
                name='detection_status'
            )
            cloudnet_class_container = Utils.variable_to_container(
                self.dataset['cloudnet_class'],
                self.dataset['cloudnet_ts'],
                self.dataset['cloudnet_rg'],
                self.dataset['mask'],
                name='CLASS'
            )

        else:
            self.contour_T = None
        # ---------------------------
        # POST PROCESSOR OFF, class probabilities
        self.predprobab_plot_name_PPoff = f'{self.CASE}-{self.MODEL_NAME}-class-probabilities--{"-".join(x for x in CLOUDNETs)}-postprocessor-off.png'
        fig_P, _ = tr.plot_timeheight(
            pred_probs,
            title='',
            range_interval=_PLOT_RANGE,
            contour=self.contour_T,
            fig_size=_FIG_SIZE,
            rg_converter=_RG_CONVERTER,
            font_size=_FONT_SIZE,
            font_weight=_FONT_WEIGHT,
        )
        fig_P.savefig(f'{self.case_plot_path}/{self.predprobab_plot_name_PPoff}', dpi=_DPI)
        matplotlib.pyplot.close(fig=fig_P)
        loggers[0].info(f'plot saved -->  {self.predprobab_plot_name_PPoff}')

        # ---------------------------
        # POST PROCESSOR OFF
        fig_size_plus_extra = np.copy(_FIG_SIZE)
        if cloudnet_data_available:
            fig_size_plus_extra[1] = fig_size_plus_extra[1] + 3

        self.prediction_plot_name_PPoff = f'{self.CASE}-{self.MODEL_NAME}-classification--{"-".join(x for x in CLOUDNETs)}-postprocessor-off.png'
        fig_raw_pred, ax_raw_pred = tr.plot_timeheight(
            pred_class,
            title='',
            range_interval=_PLOT_RANGE,
            contour=self.contour_T,
            fig_size=fig_size_plus_extra,
            rg_converter=_RG_CONVERTER,
            font_size=_FONT_SIZE,
            font_weight=_FONT_WEIGHT,
        )

        if cloudnet_data_available:
            fig_raw_pred.tight_layout(rect=[0., 0., 1.0, .65])
            fig_raw_pred, ax_raw_pred = Plot.add_lwp_to_classification(
                pred_class,
                cloudnet_class_container,
                fig_raw_pred,
                ax_raw_pred,
                cloudnet=self.CLOUDNET
            )
        fig_raw_pred.savefig(f'{self.case_plot_path}/{self.prediction_plot_name_PPoff}', dpi=_DPI)
        matplotlib.pyplot.close(fig=fig_raw_pred)
        loggers[0].info(f'plot saved -->  {self.prediction_plot_name_PPoff}')

        # POST PROCESSOR ON
        prediction_container = Utils.post_processor_temperature(
            pred_class,
            self.contour_T
        )

        prediction_container = Utils.post_processor_cloudnet_quality_flag(
            prediction_container,
            cloudnet_status_container['var'],
            cloudnet_class_container['var'],
            cloudnet_type=self.CLOUDNET
        )

        prediction_container = Utils.post_processor_cloudnet_classes(
            prediction_container,
            cloudnet_class_container['var']
        )

        if self.CLOUDNET == 'CLOUDNET_LIMRAD':
            # the matlab/polly version missclassifies a lot drizzle as aerosol and insects
            prediction_container['var'][cloudnet_class_container['var'] == 10] = 2

        prediction_container = Utils.post_processor_homogenize(
            prediction_container,
            NCLOUDNET_LABELS
        )

        self.prediction_plot_name_PPon = f'{self.CASE}-{self.MODEL_NAME}-classification--{"-".join(x for x in CLOUDNETs)}-postprocessor-on.png'

        # create directory for plots
        fig, ax = tr.plot_timeheight(
            prediction_container,
            title='',
            range_interval=_PLOT_RANGE,
            contour=self.contour_T,
            fig_size=fig_size_plus_extra,
            rg_converter=True
        )

        fig.tight_layout(rect=[0., 0., 1.0, .65])
        fig, ax = Plot.add_lwp_to_classification(prediction_container, cloudnet_class_container, fig, ax, cloudnet=self.CLOUDNET)

        fig.savefig(f'{self.case_plot_path}/{self.prediction_plot_name_PPon}', dpi=_DPI)
        matplotlib.pyplot.close(fig=fig)
        loggers[0].info(f'plot saved --> {self.case_plot_path}/{self.prediction_plot_name_PPon}')

    def prediction_to_larda_container(self, prediction, mask):
        # transform to 2D (time, range) map
        prediction2D_classes, prediction2D_probs = Utils.one_hot_to_classes(prediction, mask)

        # convert 2D arrays to larda container
        _class = Utils.container_from_prediction(
            np.copy(self.dataset['cloudnet_ts']),
            np.copy(self.dataset['cloudnet_rg']),
            np.copy(prediction2D_classes),
            np.copy(self.dataset['mask'])
        )
        _probs = Utils.container_from_prediction(
            np.copy(self.dataset['cloudnet_ts']),
            np.copy(self.dataset['cloudnet_rg']),
            prediction2D_probs,
            np.copy(self.dataset['mask']),
            name='probability',
            colormap='viridis',
            var_lims=[0.5, 1.0]
        )
        return _class, _probs


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

    N_VAL = 5  # controls size of validation data set
    NCLOUDNET_LABELS = 9    # number of ann output nodes

    _FIG_SIZE = [14, 7]
    _DPI = 450
    _FONT_SIZE = 14
    _FONT_WEIGHT = 'semibold'
    _RG_CONVERTER = True
    _PLOT_RAW_DATA = False
    _PLOT_RANGE = [0, 12000]

    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
    ANN_MODEL_TOML = 'ann_model_setting.toml'


    start_time = time.time()

    x = Voodoo(
        voodoo_path=VOODOO_PATH,
        radar='limrad94',
        ann_model_toml=ANN_MODEL_TOML,
        remove_ice=0.0,
        remove_drizzle=0.0,
        n_val=N_VAL
    )

    x.import_dataset(data_root=f'{VOODOO_PATH}/data/')

    ########################################################################################################################################################
    #   ___ ____ ____ _ _  _ _ _  _ ____
    #    |  |__/ |__| | |\ | | |\ | | __
    #    |  |  \ |  | | | \| | | \| |__]
    #
    if x.TASK == 'train':
        cnn_parameters = _init_cnn_setup(
            x.tf_settings,
            x.feature_setting,
            x.dataset['feature_set'],
            x.dataset['target_labels'],
            models_path=x.MODELS_PATH,
            model_path=x.MODEL_NAME,
            logs_path=x.LOGS_PATH
        )
        # define a new model or load an existing one
        cnn_model = Model.define_convnet_new(
            x.dataset['feature_set'].shape[1:],
            (NCLOUDNET_LABELS,),
            **cnn_parameters
        )

        #        cnn_model = densenet_model(growth_rate=32, nb_filter=64, nb_layers = [6,12,24,16], reduction=0.0,
        #                   dropout_rate=0.0, classes=9, shape=feature_set.shape[1:], batch_size=32,
        #                   with_output_block=True, with_se_layers=True)

        # parse the training set to the optimizer
        history = Model.training(
            cnn_model,
            x.dataset['feature_set'],
            x.dataset['target_labels'],
            validation=x.dataset['validation_set'],
            **cnn_parameters
        )

        # create directory for plots
        fig, _ = Plot.History(history)
        Plot.save_figure(
            fig,
            path=f'{x.PLOTS_PATH}/training/',
            name=f'histo_loss-acc_{cnn_parameters["time_str"]}__{cnn_parameters["MODEL_NAME"].replace(".h5", ".png")}',
            dpi=_DPI
        )

    ############################################################################################################################################################
    #   ___  ____ ____ ___  _ ____ ___ _ ____ _  _
    #   |__] |__/ |___ |  \ | |     |  | |  | |\ |
    #   |    |  \ |___ |__/ | |___  |  | |__| | \|
    #
    if x.TASK == 'predict':

        cnn_parameters = {}

        # define a new model or load an existing one
        cnn_model = Model.define_convnet_new(
            x.dataset['feature_set'].shape[1:],
            (NCLOUDNET_LABELS,),
            MODEL_PATH=x.MODEL_PATH,
            **cnn_parameters
        )

        # make predictions, output dimension: (n_samples, n_DBins, n_channels, 1)
        cnn_pred = Model.predict_classes(cnn_model, x.dataset['feature_set'], batch_size=x.tf_settings['BATCH_SIZE'])
        pred_class, pred_probs = x.prediction_to_larda_container(cnn_pred, x.dataset['mask'])

        x.Aftermath(pred_class, pred_probs)

    if _PLOT_RAW_DATA:
        dt_interval = [h.ts_to_dt(x.dataset['cloudnet_ts'][0]), h.ts_to_dt(x.dataset['cloudnet_ts'][-1])]

        # plot cloudnet data
        analyser_vars = {
            'campaign': 'lacros_dacapo_gpu',
            'system': ['CLOUDNETpy94', 'CLOUDNET_LIMRAD'],
            'var_name': ['Z', 'VEL', 'width', 'LDR', 'beta', 'CLASS', 'detection_status'],
            'var_converter': ['none', 'none', 'none', 'lin2z', 'log', 'none', 'none'],
            'time_interval': dt_interval,
            'range_interval': _PLOT_RANGE,
            'contour': x.contour_T,
            'plot_dir': x.case_plot_path,
            'case_name': x.CASE,
        }
        png_namesCLOUDNET = Plot.quicklooks(analyser_vars)

        # plot polly data
        analyser_vars_polly = {
            'campaign': 'lacros_dacapo_gpu',
            'system': ['POLLYNET'],
            'var_name': ['attbsc1064', 'attbsc532', 'attbsc355', 'voldepol532'],
            'var_converter': ['log', 'log', 'log', 'none'],
            'time_interval': dt_interval,
            'range_interval': _PLOT_RANGE,
            'contour': x.contour_T,
            'plot_dir': x.case_plot_path,
            'case_name': x.CASE,
        }
        png_names_polly = Plot.quicklooks(analyser_vars_polly)

        # add prediction names to png_names list
        png_names = {'prediction_PPoff': x.prediction_plot_name_PPoff,
                     'prediction_PPon': x.prediction_plot_name_PPon,
                     **png_namesCLOUDNET, **png_names_polly}

        # make predictions using the following model
        ann_params_info = Utils.read_ann_config_file(name=x.MODEL_PATH.replace(".h5", ".json"), path=x.MODELS_PATH, **cnn_parameters)
        case_study_info = {
            'html_params': json2html.convert(json=ann_params_info),
            'link': Utils.get_explorer_link(
                'lacros_dacapo', dt_interval, _PLOT_RANGE,
                ["CLOUDNET|CLASS", "CLOUDNET|Z", "POLLY|attbsc1064", "POLLY|depol"]
            ),
            'location': 'Punta-Arenas, Chile',
            'coordinates': [-53.1354, -70.8845],
            'plot_dir': analyser_vars['plot_dir'],
            'case_name': analyser_vars['case_name'],
            'time_interval': analyser_vars['time_interval'],
            'range_interval': analyser_vars['range_interval'],
            'feature_settings': x.feature_selector_settings,
        }
        Utils.make_html_overview(VOODOO_PATH, case_study_info, png_names)

    ####################################################################################################################################
    loggers[0].info(f'\n        *****Done*****, elapsed time = {datetime.timedelta(seconds=int(time.time() - start_time))} [min:sec]')
