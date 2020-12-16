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
import sys
import time
import toml
from json2html import *
import itertools

# disable the OpenMP warnings
os.environ['KMP_WARNINGS'] = 'off'

sys.path.append('../larda/')
import matplotlib

import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

import libVoodoo.Plot   as Plot
import libVoodoo.Model  as Model
import libVoodoo.Utils  as Utils
import libVoodoo.meteoSI  as meteoSI
from generate_trainingset import VoodooXR

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2020, The VOODOO Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "1.3.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

# list of cloudnet data sets used for training
CLOUDNETs = ['CLOUDNETpy94']

# get all logger
logger = Utils.logger
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.StreamHandler())


class Voodoo():

    def __init__(self, voodoo_path='', radar='limrad94'):
        # gather command line arguments
        method_name, args, kwargs = Utils.read_cmd_line_args(sys.argv)

        self.RADAR = radar
        self.TASK = kwargs['task'] if 'task' in kwargs else 'train'
        self.CLOUDNET = kwargs['cloudnet'] if 'cloudnet' in kwargs else 'CLOUDNETpy94'
        self.VOODOO_PATH = voodoo_path
        self.LOGS_PATH = f'{voodoo_path}/logs/'
        self.DATA_PATH = f'{voodoo_path}/data/'
        self.MODELS_PATH = f'{voodoo_path}/models/'
        self.MODEL_NAME = kwargs['model'] + ' ' + args[0][:] if len(args) > 0 else kwargs['model'] if 'model' in kwargs else ''
        self.MODEL_PATH = f'{self.MODELS_PATH}/{self.MODEL_NAME}'
        self.PLOTS_PATH = f'{voodoo_path}/plots/'
        self.CASE = kwargs['case'] if 'case' in kwargs else ''
        self.n_classes = {}
        self.plotnames = {}

        if self.TASK == 'predict' and not os.path.isfile(f'{self.MODEL_PATH}'):
            raise FileNotFoundError(f'Trained model not found! {self.MODEL_PATH}')

        if len(self.CASE) == 17:
            self.data_chunk_toml = f'{voodoo_path}/tomls/auto-trainingset-{self.CASE}.toml'
            self.data_chunk_heads = [chunk for chunk in Utils.load_case_file(self.data_chunk_toml).keys()]
        elif '-X' in self.CASE:
            self.data_chunk_toml = f'{voodoo_path}/tomls/auto-trainingset-{self.CASE[:-2]}.toml'
            self.data_chunk_heads = [chunk for chunk in Utils.load_case_file(self.data_chunk_toml).keys()]
        else:
            raise ValueError('Check keyword argument "case" ! (format(string): YYYYMMDD-YYYYMMDD')

    def import_dataset(self, data_root='', save=False, **kwargs):

        (classes, status, catbits, qualbits, iprob, masked, temp, press, q, ts, rg, lwp, uw, vw, multiy,
         Z, VEL, VEL_sigma, width, beta, attbsc532, depol) = (None,) * 22

        if '-X' in self.CASE:
            start_time = time.time()
            import xarray as xr
            data_path = f'{self.DATA_PATH}/xarray_zarr/{self.CASE}.zarr'
            with xr.open_zarr(data_path) as zarr_data:
                X = zarr_data['features'].values
                X = X[:, :, :, np.newaxis]
                y = zarr_data['targets'].values

            logger.critical(f'\nReading zarr {data_path}, elapsed time = {datetime.timedelta(seconds=int(time.time() - start_time))} [min:sec]')
        else:
            X, y, multiy, classes, status, catbits, qualbits, iprob, masked, temp, press, q, ts, rg, lwp, uw, vw, Z, VEL, VEL_sigma, width, beta, attbsc532, \
            depol = Utils.load_dataset_from_zarr(
                self.data_chunk_heads, self.data_chunk_toml,
                DATA_PATH=f'{data_root}/{self.CLOUDNET}',
                CLOUDNET=self.CLOUDNET, **kwargs,
                RADAR=self.RADAR,
                add_flipped=self.feature_settings['VSpec']['add_flipped'],
                CDIM=self.tf_settings['USE_MODEL'],
                TASK=self.TASK,
            )

        Utils.log_number_of_classes(y, text=f'\nsamples per class')

        if save and '-X' not in self.CASE:
            xr_ds = VoodooXR(None, None)

            # add coordinates
            xr_ds._add_coordinate({'nsamples': 'Number of training samples'}, '-', np.arange(X.shape[0]))
            xr_ds._add_coordinate({'nvelocity': 'Number of velocity bins'}, '-', np.arange(X.shape[1]))
            xr_ds._add_coordinate({'nchannels': 'Number of stacked spectra'}, '-', np.arange(X.shape[2]))
            xr_ds.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels'), np.squeeze(X), **{})
            xr_ds.add_nD_variable('targets', ('nsamples'), y, **{})

            h.change_dir(f'{self.DATA_PATH}/xarray_zarr/')
            FILE_NAME = f'{self.CASE}-X.zarr'
            xr_ds.to_zarr(store=FILE_NAME, mode='w')
            logger.critical(f'save :: {FILE_NAME}')

        # specify which kind of output we want to use image classification or multitarget classification
        if self.tf_settings['OUTPUT_TYPE'] == 'multitarget':
            y = np.fliplr(multiy)
            self.tf_settings['NCLOUDNET_LABELS'] = (6,)
        else:
            # aerosol, aerosol&insects, insects == class 8 from here on
            y[y > 8] = 8.0
            self.tf_settings['NCLOUDNET_LABELS'] = (9,)

        if 'trim_spectra' in kwargs and len(kwargs['trim_spectra']) == 2:
            X = X[:, kwargs['trim_spectra'][0]:kwargs['trim_spectra'][1], :, :]

        self.dataset = {}
        validation_set = ()

        # test from sklearn.model_selection import train_test_split
        if self.TASK == 'train':
            X, y = self.remove_randomely('remove_ice', X, y)
            X, y = self.remove_randomely('remove_drizzle', X, y)

            # splitting into training and validation set, use every n-th element from the training set for validation
            N_VAL = self.feature_settings['n_validation']
            validation_set = (X[::N_VAL], y[::N_VAL])
            X = np.array([item for index, item in enumerate(X) if (index + 1) % N_VAL != 0])
            y = np.array([item for index, item in enumerate(y) if (index + 1) % N_VAL != 0])
            Utils.log_number_of_classes(y, text=f'\nsamples per class after removing 1 in {N_VAL} values for the validation split')

        self.dataset.update({
            'feature_set': X,
            'target_labels': y,
            'validation_set': validation_set
        })

        if self.TASK == 'predict':
            self.dataset.update({
                'cloudnet_class': classes,
                'cloudnet_status': status,
                'cloudnet_category_bits': catbits,
                'cloudnet_quality_bits': qualbits,
                'insect_prob': iprob,
                'mask': masked,
                'model_temp': temp,
                'model_press': press,
                'model_q': q,
                'cloudnet_ts': ts,
                'cloudnet_rg': rg,
                'cloudnet_lwp': lwp,
                'uwind': uw,
                'vwind': vw,
                'Z': Z,
                'VEL': VEL ,
                'VEL_sigma': VEL_sigma,
                'width': width,
                'beta': beta,
                'attbsc532': attbsc532,
                'depol': depol,
            })

    def init_cnn_setup(self, ann_config_toml):

        self.ANN_MODEL_TOML = ann_config_toml
        # load ann model parameter and other global values
        config_global_model = toml.load(f'{self.VOODOO_PATH}/{self.ANN_MODEL_TOML}')
        self.feature_settings = config_global_model['feature']
        self.tf_settings = config_global_model['tensorflow']
        self.stat_settings = config_global_model['statistics']

        logger.critical(f'\nLoading {self.tf_settings["USE_MODEL"]} neural network input......')

        cnn_parameters = {

            # Convolutional part of the model
            'KIND': 'HSI',
            'CONV_DIMENSION': self.tf_settings['USE_MODEL'],

            # time of creation
            'time_str': f'{datetime.datetime.today():%Y%m%d-%H%M%S}',

        }
        assert self.tf_settings['OUTPUT_TYPE'] in ['singletarget', 'multitarget'], f'Specify the correct output layer in {self.ANN_MODEL_TOML}!'

        # create file name and add MODEL_PATH and LOGS_PATH to hyper_parameter dict
        self.MODEL_NAME = f"{cnn_parameters['time_str']}_weights.h5" if len(self.MODEL_NAME) == 0 else self.MODEL_NAME
        cnn_parameters.update({
            'MODEL_NAME': self.MODEL_NAME,
            'MODEL_PATH': f'{self.MODELS_PATH}/{self.MODEL_NAME}',
            'LOG_PATH': self.LOGS_PATH,
            # 'cloudnet': CLOUDNETs,
            **self.tf_settings,
            **self.feature_settings
        })

        if self.TASK == 'train':
            Utils.write_ann_config_file(name=self.MODEL_NAME.replace('.h5', '.json'), path=self.MODELS_PATH, **cnn_parameters)
        return cnn_parameters

    def remove_randomely(self, remove_from_class, _feature_set, _target_labels):

        if remove_from_class == 'remove_ice':
            class_nr = 4
        elif remove_from_class == 'remove_drizzle':
            class_nr = 2
        else:
            raise ValueError(f'Somethin went wrong... cannot remove randomly from class: {remove_from_class}.')

        if 100.0 > self.feature_settings[remove_from_class] > 0.0:
            idx = np.where(_target_labels == class_nr)[0]
            rand_choice = np.random.choice(idx, int(idx.size * self.feature_settings[remove_from_class] / 100.))
            _feature_set = np.delete(_feature_set, rand_choice, axis=0)
            _target_labels = np.delete(_target_labels, rand_choice, axis=0)
            self.n_classes.update({
                f'n_samples_{remove_from_class}': Utils.log_number_of_classes(
                    _target_labels,
                    text=f'\nsamples per class after removing {self.feature_settings[remove_from_class]:.2f}% of {remove_from_class}')
            })
        return _feature_set, _target_labels

    def remove_wet_radome(self, rain_flag, kind=''):
        if self.stat_settings['exclude_drizzle'] in ['disdro', 'anydrizzle', 'rg0drizzle'] and \
                self.stat_settings['exclude_wet_radome_mwr'] > 0:
            for iT in range(self.dataset['cloudnet_lwp'].size - self.stat_settings['exclude_wet_radome_mwr'], -1, -1):
                if rain_flag[self.stat_settings['exclude_drizzle']][iT] == 1:
                    self.dataset['cloudnet_lwp'][iT:iT + self.stat_settings['exclude_wet_radome_mwr']] = np.nan
                    self.dataset[f'LLT_V{kind}'][iT:iT + self.stat_settings['exclude_wet_radome_mwr']] = np.nan
                    self.dataset[f'LLT_C{kind}'][iT:iT + self.stat_settings['exclude_wet_radome_mwr']] = np.nan

            self.dataset['cloudnet_lwp'][self.dataset['cloudnet_lwp'] <= 0.0] = np.nan
            self.dataset[f'LLT_V{kind}'][self.dataset[f'LLT_V{kind}'] <= 0.0] = np.nan
            self.dataset[f'LLT_C{kind}'][self.dataset[f'LLT_C{kind}'] <= 0.0] = np.nan

            mask = ~np.isfinite(self.dataset['cloudnet_lwp']) + ~np.isfinite(self.dataset[f'LLT_V{kind}']) + ~np.isfinite(self.dataset[f'LLT_C{kind}'])
            self.dataset['cloudnet_lwp'] = np.ma.masked_where(mask, self.dataset['cloudnet_lwp'])
            self.dataset[f'LLT_V{kind}'] = np.ma.masked_where(mask, self.dataset[f'LLT_V{kind}'])
            self.dataset[f'LLT_C{kind}'] = np.ma.masked_where(mask, self.dataset[f'LLT_C{kind}'])

    def calc_adLWP(self, liquid_mask):
        bt_lists, bt_mask = Utils.find_bases_tops(liquid_mask, self.dataset['cloudnet_rg'])
        adLWP = np.zeros(self.dataset['cloudnet_ts'].size)

        for iT in range(self.dataset['cloudnet_ts'].size):
            # print(f'time step {iT}')
            n_cloud_layers = len(bt_lists[iT]['idx_cb'])
            if n_cloud_layers < 1: continue
            Tclouds, Pclouds, RGclouds = [], [], []
            for iL in range(n_cloud_layers):
                tmp_idx = range(bt_lists[iT]['idx_cb'][iL], bt_lists[iT]['idx_ct'][iL])
                if tmp_idx.stop - tmp_idx.start > 1:  # exclude single range gate clouds
                    Tclouds.append(self.dataset['model_temp'][iT, tmp_idx])
                    Pclouds.append(self.dataset['model_press'][iT, tmp_idx])
                    RGclouds.append(self.dataset['cloudnet_rg'][tmp_idx])

            try:
                Tclouds = np.concatenate(Tclouds)
                Pclouds = np.concatenate(Pclouds)
                RGclouds = np.concatenate(RGclouds)

                adLWP[iT] = np.sum(meteoSI.mod_ad(Tclouds, Pclouds, RGclouds))
            except:
                continue

        return adLWP

    def plot_2d(self, container, mask, contour=None, varname='', info='', **kwargs):


        container['mask'] = mask

        fig, ax = tr.plot_timeheight(
            container,
            title='',
            range_interval=_PLOT_RANGE,
            fig_size=_FIG_SIZE,
            rg_converter=_RG_CONVERTER,
            font_size=_FONT_SIZE,
            font_weight=_FONT_WEIGHT,
            fig=kwargs['fig'] if 'fig' in kwargs else None,
            ax=kwargs['ax'] if 'ax' in kwargs else None,
            zlim=kwargs['var_lims']
        )

        #        levels = np.arange(-40, 16, 5)
        #        cont = ax.contour(dt_list, container['rg'],
        #                          contour['var'].T,
        #                          levels,
        #                          #linestyles='dashed', colors='black', linewidths=0.75
        #                          )
        #
        #        ax.clabel(cont, fontsize=_FONT_SIZE, inline=1, fmt='%1.1f°C', )
        return fig, ax

    def save_to_png(self, fig, varname, info):
        self.case_plot_path = f'{self.LOGS_PATH}/{self.MODEL_NAME}/'
        h.change_dir(self.case_plot_path)
        self.plotnames[varname] = f'{self.CASE}-{self.MODEL_NAME}-{varname}--{"-".join(x for x in CLOUDNETs)}-{info}.png'
        fig.savefig(f'{self.plotnames[varname]}', dpi=_DPI)
        matplotlib.pyplot.close(fig=fig)
        logger.critical(f'plot saved -->  {self.plotnames[varname]}')

    def data_to_xarray(self, cnn_pred):


        xr_ds = VoodooXR(self.dataset['cloudnet_ts'], self.dataset['cloudnet_rg'])

        # add coordinates
        xr_ds._add_coordinate({'nsamples': 'Number of samples'}, '-', np.arange(self.dataset['feature_set'].shape[0]))
        xr_ds._add_coordinate({'nvelocity': 'Number of velocity bins'}, '-', np.arange(self.dataset['feature_set'].shape[1]))
        xr_ds._add_coordinate({'nchannels': 'Number of stacked spectra'}, '-', np.arange(self.dataset['feature_set'].shape[2]))

        xr_ds._add_coordinate({'cl': 'Number of Cloudnet Classes'}, '-', np.arange(9))

        # add features and labels
        xr_ds.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels'), np.squeeze(self.dataset['feature_set']), **{})
        xr_ds.add_nD_variable('targets', ('nsamples'), self.dataset['target_labels'], **{})

        # add 2D variabels from cloudnet
        xr_ds.add_nD_variable('mask_nc', ('ts', 'rg'), self.dataset['cloudnet_class'] < 1, **{})
        xr_ds.add_nD_variable('mask', ('ts', 'rg'), self.dataset['mask'], **{})
        xr_ds.add_nD_variable('temperature', ('ts', 'rg'), self.dataset['model_temp'], **{})
        xr_ds.add_nD_variable('pressure', ('ts', 'rg'), self.dataset['model_press'], **{})
        xr_ds.add_nD_variable('q', ('ts', 'rg'), self.dataset['model_q'], **{})
        xr_ds.add_nD_variable('uwind', ('ts', 'rg'), self.dataset['uwind'], **{})
        xr_ds.add_nD_variable('vwind', ('ts', 'rg'), self.dataset['vwind'], **{})
        xr_ds.add_nD_variable('insect_prob', ('ts', 'rg'), self.dataset['insect_prob'], **{})
        xr_ds.add_nD_variable('detection_status', ('ts', 'rg'), self.dataset['cloudnet_status'], **{'colormap': 'cloudnet_target_new'})
        xr_ds.add_nD_variable('target_classification', ('ts', 'rg'), self.dataset['cloudnet_class'],
                              **{'colormap': 'cloudnet_target_new', 'rg_unit': 'km', 'var_unit': '', 'system': 'Cloudnetpy'})

        # add voodoo classificationd
        pred_class, pred_probs, pred_probs3D = self.prediction_to_larda_container(cnn_pred, self.dataset['mask'])

        voodoo_specs = {key: val for key, val in pred_class.items() if isinstance(val, str) and key != 'name'}

        for var in ['Z', 'VEL', 'VEL_sigma', 'width', 'beta', 'attbsc532', 'depol']:
            self.dataset[var][np.isnan(self.dataset[var])] = -999.0
            xr_ds.add_nD_variable(var, ('ts', 'rg'), self.dataset[var], **{**voodoo_specs, **{'colormap': 'jet'}})

        xr_ds.add_nD_variable('voodoo_classification', ('ts', 'rg'), pred_class['var'], **{**voodoo_specs, **{'colormap': 'cloudnet_target_new'}})
        xr_ds.add_nD_variable('voodoo_classification_post', ('ts', 'rg'), pred_class['var'], **{**voodoo_specs, **{'colormap': 'cloudnet_target_new'}})
        xr_ds.add_nD_variable('voodoo_classification_probabilities', ('ts', 'rg', 'cl'), pred_probs3D, **{**voodoo_specs, **{'colormap': 'viridis'}})

        # plot smoothed data
        from scipy.ndimage import gaussian_filter
        smoothed_probs = np.zeros(xr_ds.voodoo_classification_probabilities.shape)
        for i in range(9):
            smoothed_probs[:, :, i] = gaussian_filter(xr_ds.voodoo_classification_probabilities[:, :, i].values, sigma=1)

        xr_ds['voodoo_classification_smoothed'] = (('ts', 'rg'), np.argmax(smoothed_probs, axis=2))
        xr_ds['voodoo_classification_smoothed'].attrs = {'colormap': 'cloudnet_target_new', 'rg_unit': 'km', 'var_unit': '', 'system': 'Cloudnetpy'}
        xr_ds['voodoo_classification_probabilities_smoothed'] = (('ts', 'rg', 'cl'), smoothed_probs)
        xr_ds['voodoo_classification_probabilities_smoothed'].attrs = {'colormap': 'cloudnet_target_new', 'rg_unit': 'km', 'var_unit': '', 'system': 'Cloudnetpy'}


        # load lwp
        self.dataset['cloudnet_lwp'] = np.ma.masked_invalid(self.dataset['cloudnet_lwp'])
        self.dataset['cloudnet_lwp'] = np.ma.masked_greater_equal(self.dataset['cloudnet_lwp'], 1.0e6)
        xr_ds.add_nD_variable('lwp', ('ts',), self.dataset['cloudnet_lwp'], **{})

        return xr_ds


    def Aftermath(self, pred_):

        # create folder for pngs
        self.case_plot_path = f'{self.LOGS_PATH}/{self.MODEL_NAME}/'
        h.change_dir(self.case_plot_path)

        xr_ds = self.data_to_xarray(pred_)

        # extract a binary mask, where True contains liquid cloud droplets
        mean_delta_rg = np.mean(np.diff(xr_ds['rg']))
        xr_ds.add_nD_variable('cloudnet_droplet_mask', ('ts', 'rg'), Utils.get_combined_mask(xr_ds['target_classification'].values, [1, 3, 5, 7]), **{})
        xr_ds.add_nD_variable('voodoo_droplet_mask', ('ts', 'rg'), Utils.get_combined_mask(xr_ds['voodoo_classification'].values, [1, 3, 5, 7]), **{})

        # calculate the liquid layer thicknes (number of pixels x delta_range for each profile)
        xr_ds.add_nD_variable('llt_cloudnet_raw', ('ts',), Utils.sum_liquid_layer_thickness(xr_ds['cloudnet_droplet_mask'], rg_res=mean_delta_rg), **{})
        xr_ds.add_nD_variable('llt_voodoo_raw', ('ts',), Utils.sum_liquid_layer_thickness(xr_ds['voodoo_droplet_mask'], rg_res=mean_delta_rg), **{})

        # exctract different rain flags, ignore invalid value warnings
        with np.errstate(invalid='ignore'):
            rain_flag = {'disdro': None,  # categorization['rainrate_ts'] > 0.0,
                         'anydrizzle': (self.dataset['cloudnet_class'] == 2).any(axis=1),
                         'rg0drizzle': (self.dataset['cloudnet_class'] == 2)[:, 0].copy()}

        # self.remove_wet_radome(rain_flag, kind='_raw')
        # self.remove_wet_radome(rain_flag)

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # POST PROCESSOR ON
        prediction_container = Utils.postprocessor(xr_ds)
        xr_ds['voodoo_classification_post'] = prediction_container.copy()
        xr_ds['mask_proc'] = (('ts', 'rg'), xr_ds['voodoo_classification_post'].values < 1)

        contour_Temp = {'data': xr_ds.temperature.values - 273.15, 'var_unit': "C", 'levels': np.arange(-40, 16, 5)}
        xr_ds.add_nD_variable('voodoo_droplet_mask_proc', ('ts', 'rg'), Utils.get_combined_mask(prediction_container.values, [1, 3, 5, 7]), **{})
        xr_ds.add_nD_variable('llt_voodoo_proc', ('ts',), Utils.sum_liquid_layer_thickness(xr_ds['voodoo_droplet_mask_proc'], rg_res=mean_delta_rg), **{})

        # correlation lwp with liquid layer thickness
        joint_availability = (xr_ds.lwp.values >= 0.0) * (xr_ds.llt_voodoo_proc.values >= 0.0)

        if self.stat_settings['exclude_drizzle'] in ['disdro', 'anydrizzle', 'rg0drizzle']:
            if self.stat_settings['exclude_wet_radome_mwr'] > 0:
                joint_availability[rain_flag[self.stat_settings['exclude_drizzle']]] = False


        predictions_nc = f'{self.CASE}-{self.MODEL_NAME}.nc'
        xr_ds.to_netcdf(predictions_nc)
        logger.critical(f'VOODOO predictions saved : {predictions_nc}')

        # ---------------------------------------------------------------------------------------------------------------------------------------
        import matplotlib.pyplot as plt
        # plot raw VOODOO classification
        fig_raw_pred, ax_raw_pred = plt.subplots(nrows=1, ncols=1, figsize=_FIG_SIZE)
        fig_raw_pred, ax_raw_pred = self.plot_2d(
            xr_ds['voodoo_classification'], xr_ds['mask'],
            fig=fig_raw_pred, ax=ax_raw_pred,
            contour=contour_Temp,
            varname='VOODOO_class', info='postproc-off', var_lims=[0, 10]
        )
        self.save_to_png(fig_raw_pred, 'VOODOO_class', info='postproc-off')


        fig_raw_pred, ax_raw_pred = plt.subplots(nrows=1, ncols=1, figsize=_FIG_SIZE)
        fig_raw_pred, ax_raw_pred = self.plot_2d(
            xr_ds['voodoo_classification_smoothed'], xr_ds['mask'],
            fig=fig_raw_pred, ax=ax_raw_pred,
            contour=contour_Temp,
            varname='VOODOO_class', info='postproc-off', var_lims=[0, 10]
        )
        self.save_to_png(fig_raw_pred, 'VOODOO_class_smoothed', info='postproc-off')

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # plot cloudnet target classification
        fig_cn_class, ax_cn_class = plt.subplots(nrows=1, ncols=1, figsize=_FIG_SIZE)
        fig_cn_class, ax_cn_class = self.plot_2d(
            xr_ds['target_classification'], xr_ds['mask'],
            fig=fig_cn_class, ax=ax_cn_class,
            contour=contour_Temp,
            varname='CLOUDNET_class', info='postproc-off', var_lims=[0, 10]
        )
        self.save_to_png(fig_cn_class, 'CLOUDNET_class', info='postproc-off')

        # ---------------------------------------------------------------------------------------------------------------------------------------
        # plot cloudnet target classification merged with voodoo
        fig_proc_pred, ax_proc_pred = plt.subplots(nrows=1, ncols=1, figsize=_FIG_SIZE)
        fig_proc_pred, ax_proc_pred = self.plot_2d(
            prediction_container, xr_ds['mask'],
            fig=fig_proc_pred, ax=ax_proc_pred,
            contour=contour_Temp,
            varname='VOODOO_class2', info='postproc-on', var_lims=[0, 10])
        self.save_to_png(fig_proc_pred, 'VOODOO_class2', info='postproc-on')

        prediction_container = Utils.postprocessor(xr_ds, smooth=True)
        fig_proc_pred, ax_proc_pred = plt.subplots(nrows=1, ncols=1, figsize=_FIG_SIZE)
        fig_proc_pred, ax_proc_pred = self.plot_2d(
            prediction_container, xr_ds['mask'],
            fig=fig_proc_pred, ax=ax_proc_pred,
            contour=contour_Temp,
            varname='VOODOO_class2', info='postproc-on', var_lims=[0, 10])
        self.save_to_png(fig_proc_pred, 'VOODOO_class2_smoothed', info='postproc-on')


    def prediction_to_larda_container(self, prediction, mask):

        # transform to 2D (time, range) map
        if self.tf_settings['OUTPUT_TYPE'] == 'multitarget':
            tmp = self.get_target_classification(prediction, threshold=0.3)

            prediction2D_classes = np.zeros(mask.shape, dtype=np.float32)
            _probs3D = np.zeros(mask.shape, dtype=np.float32)
            cnt = 0
            for iT, iR in itertools.product(range(mask.shape[0]), range(mask.shape[1])):
                if mask[iT, iR]: continue
                prediction2D_classes[iT, iR] = tmp[cnt]
                _probs3D[iT, iR] = 0.0
                cnt += 1
        else:
            prediction2D_classes, _probs3D = Utils.one_hot_to_classes(prediction, mask)
            prediction2D_classes[prediction2D_classes > 7] = 10

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
            np.max(_probs3D, axis=2),
            np.copy(self.dataset['mask']),
            name='probability',
            colormap='viridis',
            var_lims=[0.5, 1.0]
        )

        return _class, _probs, _probs3D

    def get_target_classification(self, bits, threshold=0.1):
        """
            0'Small liquid droplets are present.',
            1'Falling hydrometeors are present',
            2'Wet-bulb temperature is less than 0 degrees C, implying the phase of Bit-1 particles.',
            3'Melting ice particles are present.',
            4'Aerosol particles are present and visible to the lidar.',
            5'Insects are present and visible to the radar.'

        Args:
            bits:

        Returns:

        """
        classification = np.zeros(bits.shape[0], dtype=int)

        droplets = bits[:, 0] > threshold
        falling = bits[:, 1] > threshold
        cold = bits[:, 2] > 0.4
        melting = bits[:, 3] > threshold
        aerosol = bits[:, 4] > threshold
        insect = bits[:, 5] > threshold
        classification[droplets & ~falling] = 1
        classification[~droplets & falling] = 2
        classification[droplets & falling] = 3
        classification[~droplets & falling & cold] = 4
        classification[droplets & falling & cold] = 5
        classification[melting] = 6
        classification[melting & droplets] = 7
        # classification[aerosol] = 8
        # classification[insect] = 9
        classification[aerosol & insect] = 10
        # classification[~aerosol] = 0
        return classification


# clounet bitmask
#    :comment = "This variable contains information on the nature of the targets at each pixel,
#    thereby facilitating the application of algorithms that work with only one type of target.
#    The information is in the form of an array of bits, each of which states either whether a
#    certain type of particle is present (e.g. aerosols), or the whether some of the target
#    particles have a particular property. The definitions of each bit are given in the definition
#    attribute. Bit 0 is the least significant.";
#   :definition = "
#   Bit 0: Small liquid droplets are present.
#   Bit 1: Falling hydrometeors are present; if Bit 2 is set then these are most likely ice particles, otherwise they are drizzle or rain drops.
#   Bit 2: Wet-bulb temperature is less than 0 degrees C, implying the phase of Bit-1 particles.
#   Bit 3: Melting ice particles are present.
#   Bit 4: Aerosol particles are present and visible to the lidar.
#   Bit 5: Insects are present and visible to the radar.";

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

    _FIG_SIZE = [17, 7]
    _DPI = 450
    _FONT_SIZE = 14
    _FONT_WEIGHT = 'normal'
    _RG_CONVERTER = True
    _PLOT_RAW_DATA = False
    _PLOT_RANGE = [0, 12000]

    VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
    ANN_MODEL_TOML = 'ann_model_setting.toml'
    DATA_LOC = f'{VOODOO_PATH}/data/'

    start_time = time.time()
    logger.critical(f'\nData location : {DATA_LOC}')
    logger.critical(f'Model settings : {ANN_MODEL_TOML}')

    ########################################################################################################################################################
    #   _ _  _ _ ___ _ ____ _    _ ___  _ _  _ ____
    #   | |\ | |  |  | |__| |    |   /  | |\ | | __
    #   | | \| |  |  | |  | |___ |  /__ | | \| |__]
    #
    x = Voodoo(voodoo_path=VOODOO_PATH, radar='limrad94')

    cnn_parameters = x.init_cnn_setup(ann_config_toml=ANN_MODEL_TOML)

    x.import_dataset(data_root=DATA_LOC, save=False)
    # x.import_dataset(data_root=DATA_LOC, trim_spectra=[256, 512])
    #sys.exit()

    # I/O dimensions
    cnn_parameters.update({
        'INPUT_DIMENSION': x.dataset['feature_set'].shape,
        'OUTPUT_DIMENSION': x.dataset['target_labels'].shape
    })

    ########################################################################################################################################################
    #   ___ ____ ____ _ _  _ _ _  _ ____
    #    |  |__/ |__| | |\ | | |\ | | __
    #    |  |  \ |  | | | \| | | \| |__]
    #
    if x.TASK == 'train':
        # define a new model or load an existing one
        cnn_model = Model.define_convnet_new(
            x.dataset['feature_set'].shape[1:],
            x.tf_settings['NCLOUDNET_LABELS'],
            **cnn_parameters
        )

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
        # define a new model or load an existing one
        cnn_model = Model.define_convnet_new(
            x.dataset['feature_set'].shape[1:],
            x.tf_settings['NCLOUDNET_LABELS'],
            MODEL_PATH=x.MODEL_PATH,
        )

        # make predictions, output dimension: (n_samples, n_DBins, n_channels, 1)
        cnn_pred = Model.predict_classes(cnn_model, x.dataset['feature_set'])

        x.Aftermath(cnn_pred)

    ############################################################################################################################################################
    #   ___  _    ____ ___    ____ ____ _ _ _    ___  ____ ___ ____
    #   |__] |    |  |  |     |__/ |__| | | |    |  \ |__|  |  |__|
    #   |    |___ |__|  |     |  \ |  | |_|_|    |__/ |  |  |  |  |
    #
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
            'fig_size': _FIG_SIZE,
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
            'fig_size': _FIG_SIZE,
        }

        png_names_polly = Plot.quicklooks(analyser_vars_polly)

        # add prediction names to png_names list
        png_names = {'prediction_PPoff': x.plotnames['VOODOO_class'],
                     'prediction_PPon': x.plotnames['VOODOO_class2'],
                     **png_namesCLOUDNET, **png_names_polly}

        # make predictions using the following model
        ann_params_info = Utils.read_ann_config_file(name=x.MODEL_PATH.replace(".h5", ".json"), path=x.MODELS_PATH)
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
            'feature_settingss': x.feature_settings,
            'fig_size': _FIG_SIZE,
        }
        Utils.make_html_overview(VOODOO_PATH, case_study_info, png_names)

    ####################################################################################################################################
    logger.critical(f'\n        *****Done*****, elapsed time = {datetime.timedelta(seconds=int(time.time() - start_time))} [min:sec]')
