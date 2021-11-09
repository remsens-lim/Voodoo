#!/usr/bin/env python
# coding: utf-8
import sys
import glob
import sys

sys.path.append('/home/sdig/code/larda3/larda/')
sys.path.append('/home/sdig/code/Voodoo/')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import numpy as np
import xarray as xr
import seaborn as sns
import latextable
import pandas as pd
import scipy.interpolate

from datetime import timezone

sys.path.append('/home/sdig/code/larda3/larda/')
sys.path.append('/home/sdig/code/Voodoo/')

import scipy.interpolate

import datetime
import csv

# optionally configure the logging
# StreamHandler will print to console
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import libVoodoo.Utils as UT

MODEL = 'Vnet0x60de1687-fnX-gpu0-VN'

PRED_PATH = '/home/sdig/code/Voodoo/torch_models/'
MEDIA_PATH = '/media/sdig/'

CODE_PATH = '/Users/willi/Documents/LaTeX/VOODOO/code/'

QUICKLOOK_PATH = f'{PRED_PATH}/{MODEL[:14]}/plots/analyzer/'

n_smoothing = 20

p = 0.4


def decimalhour2unix(dt, time):
    return np.array([x * 3600. + h.dt_to_ts(datetime.datetime(int(dt[:4]), int(dt[4:6]), int(dt[6:]), 0, 0, 0)) for x in time])


def open_xarray_datasets(path):
    ds = xr.open_mfdataset(path, parallel=True, decode_times=False, )
    x = re.findall("\d{8}", path)[0]

    # convert time to unix
    ds = ds.assign_coords(time=("time", decimalhour2unix(str(x), ds['time'].values)))
    ds['time'].attrs['units'] = 'Unix Time: Seconds since January 1st, 1970'

    return ds

# VOODOO cloud droplet likelyhood colorbar (viridis + grey below minimum value)
viridis = cm.get_cmap('viridis', 6)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[:1, :] = np.array([220 / 256, 220 / 256, 220 / 256, 1])
newcolors = ListedColormap(newcolors)

_, agrs, kwargs = UT.read_cmd_line_args()

if 'YYMM' in kwargs:
    YYMM = kwargs['YYMM']
else:
    YYMM = '201908'

if int(YYMM) > 202001:
    site = 'LIM'
else:
    site = 'punta-arenas'

new_cloudnet_cat_files = glob.glob(f'{PRED_PATH}/{MODEL[:14]}/nc/*{YYMM}*-{site}-categorize-*{MODEL}*.nc')
new_cloudnet_class_files = glob.glob(f'{PRED_PATH}/{MODEL[:14]}/nc/*{YYMM}*-{site}-classification-*{MODEL}*.nc')
new_cloudnet_cat_files.sort()
new_cloudnet_class_files.sort()

var_list = {'Z': [-50, 20], 'v': [-4, 2], 'width': [0, 1], 'v_sigma': [0, 1], 'ldr': [-30, 0], 'beta': [1.0e-7, 1.0e-4], 'lwp': [-25, 500]}

for cat_file, class_file in zip(new_cloudnet_cat_files, new_cloudnet_class_files):

    rg_lims = [0, 12000]
    date = cat_file[cat_file.rfind('/') + 1:cat_file.rfind('/') + 9]
    print(f'\n {date} \n')

    try:
        csv_row_list = []
        with open(f'/home/sdig/code/larda3/scripts_Willi/larda_cloud_sniffer/sniffer_code/{date}-cloud-props.csv', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                csv_row_list.append(row)
            #print(csv_row_list)
    except FileNotFoundError:
        csv_row_list = [['begin_dt', 'end_dt'], [date+'000005', date+'-235955']]
        print('use default case', csv_row_list)

    int_names = ['TP', 'TN', 'FP', 'FN']
    flt_names = ['precision', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score', 'Fhalf-score']
    new_header = csv_row_list[0] + int_names + flt_names
    new_rows = []

    if site == 'punta-arenas':
        MEDIA_PATH = '/media/sdig/LACROS/cloudnet/data/punta-arenas/'
        cn_cat_file = f'{MEDIA_PATH}/processed/limrad94/categorize-py/{date[:4]}/{date}-{site}-categorize-limrad94.nc'
        cn_class_file = f'{MEDIA_PATH}/products/limrad94/classification-cloudnetpy/{date[:4]}/{date}-{site}-classification-limrad94.nc'
        ds_ceilo = xr.open_mfdataset(MEDIA_PATH + f'/calibrated/chm15x/{date[:4]}/{date}_{site}_chm15x.nc')
    else:
        site = 'LIM'
        MEDIA_PATH = '/media/sdig/leipzig/'
        cn_cat_file = f'{MEDIA_PATH}/cloudnet/processed/limrad94/categorize-py/{date[:4]}/{date}-{site}-categorize-limrad94.nc'
        cn_class_file = f'{MEDIA_PATH}/cloudnet/products/limrad94/classification-cloudnetpy/{date[:4]}/{date}-{site}-classification-limrad94.nc'
        ds_ceilo = xr.open_mfdataset(f'{MEDIA_PATH}/instruments/ceilim/data/Y{date[:4]}/M{date[4:6]}/{date}_Leipzig_CHM200114_000.nc')

    # # Load CloudnetPy
    ds_ED = open_xarray_datasets(cn_cat_file)
    ds_cnclass_ED = open_xarray_datasets(cn_class_file)
    ds_voodoo_cat_ED = open_xarray_datasets(cat_file)
    ds_vclass_ED = open_xarray_datasets(class_file)

    for ind_row, row in enumerate(csv_row_list[1:]):

        # life time > 15 minutes and cloud fraction > 0.7
        if float(row[7]) < 15*60 or float(row[8]) < 0.7:
            print('skipped', ind_row)
            continue

        date = row[0][:8]
        if len(row) > 2:
            rg_lims = [float(row[10]), float(row[10])+float(row[12])]

        begin_dt = datetime.datetime.strptime(f'{row[0]}', '%Y%m%d-%H%M%S')
        end_dt = datetime.datetime.strptime(f'{row[1]}', '%Y%m%d-%H%M%S')
        ts_lims = [begin_dt.replace(tzinfo=timezone.utc).timestamp(), end_dt.replace(tzinfo=timezone.utc).timestamp()]

        # slice daily dataset
        ds = ds_ED.sel(time=slice(*ts_lims), height=slice(*rg_lims))
        ds_cnclass = ds_cnclass_ED.sel(time=slice(*ts_lims), height=slice(*rg_lims))
        ds_voodoo_cat = ds_voodoo_cat_ED.sel(time=slice(*ts_lims), height=slice(*rg_lims))
        ds_vclass = ds_vclass_ED.sel(time=slice(*ts_lims), height=slice(*rg_lims))

        _ = [h.ts_to_dt(ts) for ts in ds['Z']['time'].values]

        clear_sky_mask = ds['Z'].values > 1.0e5

        # VOODOO target classification
        ds['voodoo_class'] = ds_vclass['target_classification'].copy()
        ds['voodoo_class'].values = np.ma.masked_where(clear_sky_mask, ds['voodoo_class'].values)
        ds['voodoo_class'].attrs['units'] = ''
        ds['voodoo_class'].attrs['rg_unit'] = 'km'
        ds['voodoo_class'].attrs['system'] = 'CloudnetPy+Voodoo'
        ds['voodoo_class'].attrs['var_lims'] = [0, 11]
        ds['voodoo_class'].attrs['colormap'] = 'cloudnet_target_new'
        ds['voodoo_class'].attrs['dimlabel'] = ['time', 'range']
        ds['voodoo_class'].height.attrs['units'] = 'km'

        # CLOUDNET target classification
        ds['cloudnet_class'] = ds_cnclass['target_classification'].copy()
        ds['cloudnet_class'].values = np.ma.masked_where(clear_sky_mask, ds['cloudnet_class'].values)
        ds['cloudnet_class'].attrs['units'] = ''
        ds['cloudnet_class'].attrs['system'] = 'CloudnetPy'
        ds['cloudnet_class'].attrs['var_lims'] = [0, 11]
        ds['cloudnet_class'].attrs['dimlabel'] = ['time', 'range']
        ds['cloudnet_class'].height.attrs['units'] = 'km'
        ds['cloudnet_class'].attrs['rg_unit'] = 'km'
        ds['cloudnet_class'].attrs['var_unit'] = ''
        ds['cloudnet_class'].attrs['colormap'] = 'cloudnet_target_new_tex'

        # CLOUDNET detection statsus
        ds['cloudnet_status'] = ds_cnclass['detection_status'].copy()
        ds['cloudnet_status'].values = np.ma.masked_where(clear_sky_mask, ds['cloudnet_status'].values)
        ds['cloudnet_status'].attrs['units'] = ''
        ds['cloudnet_status'].attrs['rg_unit'] = 'km'
        ds['cloudnet_status'].attrs['system'] = 'CloudnetPy'
        ds['cloudnet_status'].attrs['var_lims'] = [0, 11]
        ds['cloudnet_status'].attrs['var_unit'] = ''
        ds['cloudnet_status'].attrs['colormap'] = 'cloudnetpy_detection_status'
        ds['cloudnet_status'].attrs['dimlabel'] = ['time', 'range']
        ds['cloudnet_status'].height.attrs['units'] = 'km'

        ds['voodoo_cd_prob'] = ds_voodoo_cat['pred_liquid_prob_1'].copy()
        ds['voodoo_cd_prob'].values = np.ma.masked_where(clear_sky_mask, ds['voodoo_cd_prob'].values)
        ds['voodoo_cd_prob'].attrs['units'] = '1'
        ds['voodoo_cd_prob'].attrs['rg_unit'] = 'km'
        ds['voodoo_cd_prob'].attrs['system'] = 'Voodoo'
        ds['voodoo_cd_prob'].attrs['var_lims'] = [p, 1.]
        ds['voodoo_cd_prob'].attrs['var_unit'] = '1'
        ds['voodoo_cd_prob'].attrs['colormap'] = newcolors
        ds['voodoo_cd_prob'].attrs['dimlabel'] = ['time', 'range']
        ds['voodoo_cd_prob'].height.attrs['units'] = 'km'

        def _interp_model_variable(var_name):
            ts, rg = ds['time'].values, ds['height'].values
            f = scipy.interpolate.interp2d(
                ds['model_time'].values,
                ds['model_height'].values,
                ds[var_name].values.T,
                kind='linear',
                copy=True,
                bounds_error=False,
                fill_value=None
            )
            return f(ts, rg)[:, :]

        _T = _interp_model_variable('temperature')
        _p = _interp_model_variable('pressure')

        # copy datetime list and range list for plotting
        dt_list = [h.ts_to_dt(ts) for ts in ds['Z']['time'].values]
        rg_list = ds['Z']['height'].values

        try:
            ds = ds.rename({'time': 'ts'})
            ds = ds.rename({'height': 'rg'})
        except:
            pass

        _ccl = ds_cnclass['target_classification'].values
        _cst = ds_cnclass['detection_status'].values
        _vcl = ds_vclass['target_classification'].values
        _mask_voodoo = (ds_voodoo_cat['pred_liquid_prob_1'].values > p)  # * (ds_voodoo_cat['pred_liquid_prob_1'].values < 0.90)
        _mask_cloudnet = (_ccl == 1) + (_ccl == 3) + (_ccl == 5) + (_ccl == 7)
        _mask_combi = (_vcl == 1) + (_vcl == 3) + (_vcl == 5) + (_vcl == 7)
        _is_clutter = _ccl > 7
        _is_rain = np.array([ds['is_rain'].values] * ds['rg'].size, dtype=bool).T
        _cloud_mask = _mask_cloudnet + (_ccl == 2) + (_ccl == 4) + (_ccl == 6)
        _is_falling = (_ccl == 2) * (ds['v'].values < 3)

        # # create dictionary with liquid pixel masks
        liquid_masks2D = {
            'Voodoo': _mask_voodoo * ~_is_clutter * ~_is_falling,
            'CloudnetPy': _mask_cloudnet * ~_is_clutter * ~_is_falling ,
        }

        h.change_dir(QUICKLOOK_PATH)


        def _error_matrix_masks(_mask_CN, _mask_V, _status):
            _TP_mask = _mask_CN * _mask_V * ~_is_rain
            _TN_mask = (~_mask_CN * ~_mask_V) * (_mask_CN + (_status == 1))
            _FP_mask = (~_mask_CN * _mask_V) * ~_TP_mask * ~_is_rain
            _FN_mask = (_mask_CN * ~_mask_V) * (_mask_CN + (_status == 1))
            _FN_mask[(_status == 4)] = 0

            return [_TP_mask, _FN_mask,  _FP_mask, _TN_mask]


        def _histo_errormatrix(var1='v', var2='width'):
            title_list = ['TP', 'FN', 'FP', 'TN']
            mask_list = _error_matrix_masks(_mask_cloudnet, liquid_masks2D['Voodoo'], ds['cloudnet_status'].values)
            Nbins = 40
            hist = []
            for i, (i_mask, i_title) in enumerate(zip(mask_list, title_list)):
                _mask = (ds[var1].values < 90) * i_mask

                d1 = ds[var1].values[_mask].ravel()
                d2 = ds[var2].values[_mask].ravel()

                hist.append(tr._create_histogram(
                    d1, d2,
                    Nbins=Nbins,
                    x_lim=var_list[var1],
                    y_lim=var_list[var2],
                ))

            return hist

        errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='v')
    #    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='v_sigma')
    #    errormatrix_histo_list = _histo_errormatrix(var1='v', var2='v_sigma')
    #    errormatrix_histo_list = _histo_errormatrix(var1='width', var2='v_sigma')
    #    errormatrix_histo_list = _histo_errormatrix(var1='v', var2='width')
    #    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='width')

        # PERFORMANCE METRICS
        def _performance_metrics(_mask_CN, _mask_V, _status):

            mask_list = _error_matrix_masks(_mask_CN, _mask_V, _status)
            TP = np.count_nonzero(mask_list[0])
            TN = np.count_nonzero(mask_list[1])
            FN = np.count_nonzero(mask_list[2])
            FP = np.count_nonzero(mask_list[3])

            arr = np.zeros((11))
            beta = 0.5
            _eps = 1.0e-7

            arr[:4] = TP, TN, FP, FN
            arr[4] = TP / max(TP + FP, _eps)  # precision
            arr[5] = TN / max(TN + FN, _eps)  # npv
            arr[6] = TP / max(TP + FN, _eps)  # recall
            arr[7] = TN / max(TN + FP, _eps)  # specificity
            arr[8] = (TP + TN) / max(TP + TN + FP + FN, _eps)  # accuracy
            arr[9] = 2 * TP / max(2 * TP + FP + FN, _eps)  # F1-score
            arr[10] = (1 + beta * beta) * arr[4] * arr[6] / (arr[6] + beta * beta * arr[4])

            return arr

        arr0 = np.zeros((1, 11), dtype=float)
        arr0[0, :] = _performance_metrics(_mask_cloudnet, liquid_masks2D['Voodoo'], ds['cloudnet_status'].values)
        new_rows.append(row + list(arr0[0, :]))

        # create pandas dataframe and save to csv
        print(f'\n')
        thr_names = ['', 'all', ]
        stats_list = [[thr_names, ] +
                      [[key + '',] + list(val.astype(int)) for key, val in zip(int_names, arr0[:, :4].T)] +
                      [[key + '', ] + list(val) for key, val in zip(flt_names, arr0[:, 4:].T)]
           ]

        table1 = latextable.Texttable()
        table1.set_deco(latextable.Texttable.HEADER)
        table1.set_cols_align(["r", 'c', ])
        table1.add_rows(stats_list[0])
        # print(table1.draw() + "\t\n")
        print(latextable.draw_latex(table1, caption=f"a"))
#
#        df = pd.DataFrame(
#            np.array(stats_list[0][1:])[:, 1:],
#            columns=stats_list[0][0][1:],
#            index=[n + '' for n in int_names] +
#                  [n + '' for n in flt_names]
#        )
#        df = df.T

        h.change_dir(f'{PRED_PATH}/{MODEL[:14]}/statistics/')
#        df.to_csv(f'{begin_dt:%Y%m%d%H%M%S}-{end_dt:%H%M%S}-csv3.csv')

    with open(f'{date}-cloud-props-csv3.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        for i, cloud in enumerate(new_rows):
            writer.writerow(cloud)
