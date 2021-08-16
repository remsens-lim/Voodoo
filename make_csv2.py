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

from datetime import timezone

sys.path.append('/home/sdig/code/larda3/larda/')
sys.path.append('/home/sdig/code/Voodoo/')

import scipy.interpolate

import datetime

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
    YYMM = '202012'

if int(YYMM) > 202001:
    SITE = 'LIM'
else:
    SITE = 'punta-arenas'

new_cloudnet_cat_files = glob.glob(f'{PRED_PATH}/{MODEL[:14]}/nc/*{YYMM}*-{SITE}-categorize-*{MODEL}*.nc')
new_cloudnet_class_files = glob.glob(f'{PRED_PATH}/{MODEL[:14]}/nc/*{YYMM}*-{SITE}-classification-*{MODEL}*.nc')
new_cloudnet_cat_files.sort()
new_cloudnet_class_files.sort()

for cat_file, class_file in zip(new_cloudnet_cat_files, new_cloudnet_class_files):

    date = cat_file[cat_file.rfind('/') + 1:cat_file.rfind('/') + 9]
    print(f'\n {date} \n')
    # if not (20190223 <= int(date) < 20190224):
    # if not (20201220 <= int(date) < 20201221):
    #        continue

    begin_dt = datetime.datetime.strptime(f'{date}0001', '%Y%m%d%H%M')
    end_dt = datetime.datetime.strptime(f'{date}2359', '%Y%m%d%H%M')

    # begin_dt = datetime.datetime.strptime('201908010000', '%Y%m%d%H%M')
    # end_dt = datetime.datetime.strptime('201908010905', '%Y%m%d%H%M')

    d0_str = f'{begin_dt.year}{str(begin_dt.month).zfill(2)}{str(begin_dt.day).zfill(2)}'
    d1_str = f'{end_dt.year}{str(end_dt.month).zfill(2)}{str(end_dt.day).zfill(2)}'

    ts_lims = [begin_dt.replace(tzinfo=timezone.utc).timestamp(), end_dt.replace(tzinfo=timezone.utc).timestamp()]
    rg_lims = [0, 12000]
    var_list = {'Z': [-50, 20], 'v': [-4, 2], 'width': [0, 1], 'ldr': [-30, 0], 'beta': [1.0e-7, 1.0e-4], 'lwp': [-25, 500]}

    if begin_dt < datetime.datetime.strptime('2020', '%Y'):
        site = 'punta-arenas'
    else:
        site = 'LIM'

    if begin_dt < datetime.datetime.strptime('2020', '%Y'):
        MEDIA_PATH = '/media/sdig/LACROS/cloudnet/data/punta-arenas/'
        site = 'punta-arenas'
        cn_cat_file = f'{MEDIA_PATH}/processed/limrad94/categorize-py/{date[:4]}/{date}-{site}-categorize-limrad94.nc'
        cn_class_file = f'{MEDIA_PATH}/products/limrad94/classification-cloudnetpy/{date[:4]}/{date}-{site}-classification-limrad94.nc'
        ds_ceilo = xr.open_mfdataset(MEDIA_PATH + f'/calibrated/chm15x/{date[:4]}/{date}_{site}_chm15x.nc')
    else:
        MEDIA_PATH = '/media/sdig/leipzig/'
        site = 'LIM'
        cn_cat_file = f'{MEDIA_PATH}/cloudnet/processed/limrad94/categorize-py/{date[:4]}/{date}-{site}-categorize-limrad94.nc'
        cn_class_file = f'{MEDIA_PATH}/cloudnet/products/limrad94/classification-cloudnetpy/{date[:4]}/{date}-{site}-classification-limrad94.nc'
        ds_ceilo = xr.open_mfdataset(f'{MEDIA_PATH}/instruments/ceilim/data/Y{date[:4]}/M{date[4:6]}/{date}_Leipzig_CHM200114_000.nc')

    # # Load CloudnetPy
    ds = open_xarray_datasets(cn_cat_file)
    ds_cnclass = open_xarray_datasets(cn_class_file)
    ds_voodoo_cat = open_xarray_datasets(cat_file)
    ds_vclass = open_xarray_datasets(class_file)

    _ = [h.ts_to_dt(ts) for ts in ds['Z']['time'].values]
    ceilo_cbh_interp = ds_ceilo['cbh'].values

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

    ds['kappa'] = ds_voodoo_cat['pred_liquid_prob_1'].copy()
    ds['kappa'].attrs['units'] = '-'
    ds['kappa'].attrs['rg_unit'] = 'km'
    ds['kappa'].attrs['system'] = 'Voodoo'
    ds['kappa'].attrs['var_lims'] = [0, 0.4]
    ds['kappa'].attrs['var_unit'] = ''
    ds['kappa'].attrs['colormap'] = 'jet'
    ds['kappa'].attrs['dimlabel'] = ['time', 'range']
    ds['kappa'].height.attrs['units'] = 'km'

    import scipy.interpolate

    i0 = (begin_dt - datetime.datetime(1970, 1, 1)).total_seconds()
    ts, rg = ds['time'].values, ds['height'].values
    # ts = (ts-i0)/3600
    _ts_from_ceilo = np.array([(dt64 - ds_ceilo['time'][0].values) / np.timedelta64(1, 's') for dt64 in ds_ceilo['time'].values]) / 3600

    f = scipy.interpolate.interp2d(
        ds['model_time'].values,
        ds['model_height'].values,
        ds['temperature'].values.T,
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )
    _T = f(ts, rg)[:, :].T

    f = scipy.interpolate.interp2d(
        ds['model_time'].values,
        ds['model_height'].values,
        ds['pressure'].values.T,
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )
    _p = f(ts, rg)[:, :].T

    __cbh = ds_ceilo['cbh'][:, 0].values

    f = scipy.interpolate.interp1d(
        _ts_from_ceilo,
        __cbh,
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )
    _cbh = f((ts - i0) / 3600)[:]

    # slice
    #    ds = ds.sel(time=slice(*ts_lims), height=slice(*rg_lims))
    #    ds_cnclass = ds_cnclass.sel(time=slice(*ts_lims), height=slice(*rg_lims))
    #    ds_vclass = ds_vclass.sel(time=slice(*ts_lims), height=slice(*rg_lims))
    #    ds_voodoo_cat = ds_voodoo_cat.sel(time=slice(*ts_lims), height=slice(*rg_lims))

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


    _mask_below_cloudbase = np.full(_ccl.shape, False)
    for its, icb in enumerate(_cbh):
        if icb > 250:
            idx_cb = h.argnearest(rg_list, icb)
            _mask_below_cloudbase[its, :idx_cb] = True

    # # Calculate tubulence indext $\kappa=\frac{\vert \mathrm{MDV} - \overline{\mathrm{MDV}} \vert}{\overline{\mathrm{MDV}}}$
    #
    # compute convective index
    kappa_min, kappa_max = -4, 4
    mdv = ds['v'].copy()
    kappa = UT.convection_index_fast(mdv.values, dts=20, drg=1)
    ds['kappa'].values = kappa

    _ = np.ma.masked_less(kappa, kappa_min)
    _ = np.ma.masked_greater(_, kappa_max)
    _ = np.ma.masked_where(_ == -1, _)
    kappa_mask = np.ma.getmaskarray(_)

    # # create dictionary with liquid pixel masks
    liquid_masks2D = {
        'Voodoo': _mask_voodoo * ~_is_clutter * ~_is_falling,
        'Voodoo-post': _mask_voodoo * ~_is_clutter * ~_is_falling * ~kappa_mask,
        'CloudnetPy': _mask_cloudnet * ~_is_clutter * ~_is_falling ,
    }


    rg_res = np.mean(np.diff(ds['rg'].values)) * 0.001

    # # calculate cloud bases, cloud tops, liquid water content, liqudi water lap, liquid layer thickness
    # cloud tops and cloud bases
    bases_tops_dict = {
        alg: UT.find_bases_tops(liquid_masks2D[alg], ds['rg'].values)
        for alg in liquid_masks2D.keys()
    }
    bases = {
        alg: UT.get_bases_or_tops([h.ts_to_dt(ts) for ts in ds['Z']['ts'].values], bases_tops_dict[alg], key='cb')
        for alg in liquid_masks2D.keys()
    }

    # liquid water content
    lwc_dict = {}
    for key, mask in liquid_masks2D.items():
        _lwc = UT.adiabatic_liquid_water_content(_T, _p, mask, delta_h=float(np.mean(np.diff(ds['rg'].values))))
        _lwc[_lwc > 1000] = np.nan
        _lwc[_lwc < 1] = np.nan
        lwc_dict[key] = np.ma.masked_invalid(_lwc)  # * (1 + np.ma.masked_invalid(ds['voodoo_cd_prob'].values))

    _lwp = ds['lwp'].values
    _lwp[_lwp > 4000] = np.nan
    _lwp[_lwp < 1] = np.nan
    a = pd.Series(_lwp)

    # liquid water path and liquid layer thickness
    llt_dict, lwp_dict = {}, {}
    lwp_dict['mwr'] = a.interpolate(method='nearest').values
    lwp_dict['mwr_s'] = h.smooth(lwp_dict['mwr'], n_smoothing)
    for key in liquid_masks2D.keys():
        lwp_dict[key] = np.ma.sum(lwc_dict[key], axis=1)
        lwp_dict[key + '_s'] = h.smooth(lwp_dict[key], n_smoothing)

        llt_dict[key] = np.count_nonzero(liquid_masks2D[key], axis=1) * rg_res
        llt_dict[key + '_s'] = h.smooth(llt_dict[key], n_smoothing)

    # compute the first liquid cloud base height
    CBH_dict = {'CEILO': _cbh}
    for key in liquid_masks2D.keys():
        _tmp = np.argmax(liquid_masks2D[key] == 1, axis=1)
        CBH_dict[key] = np.ma.masked_less_equal([rg_list[ind_rg] for ind_rg in _tmp], 300.0)

    h.change_dir(QUICKLOOK_PATH)




    ##########################################################################################################################################################
    ##########################################################################################################################################################

    def _error_matrix_masks(_mask_CN, _mask_V, _status):
        _TP_mask = _mask_CN * _mask_V * ~_is_rain
        _TN_mask = (~_mask_CN * ~_mask_V) * (_mask_CN + (_status == 1))
        _FP_mask = (~_mask_CN * _mask_V) * ~_TP_mask * ~_is_rain
        _FN_mask = (_mask_CN * ~_mask_V) * (_mask_CN + (_status == 1))
        _FN_mask[(_status == 4)] = 0

        for its, icb in enumerate(_cbh):
            if icb > 250:
                idx_cb = h.argnearest(rg_list, icb)
                _TN_mask[its, idx_cb + 1:] = False
                _FP_mask[its, idx_cb + 1:] = False
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
                x_lim=ds[var1].attrs['var_lims'],
                y_lim=ds[var2].attrs['var_lims'],
            ))

        return hist

    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='v')
#    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='v_sigma')
#    errormatrix_histo_list = _histo_errormatrix(var1='v', var2='v_sigma')
#    errormatrix_histo_list = _histo_errormatrix(var1='width', var2='v_sigma')
#    errormatrix_histo_list = _histo_errormatrix(var1='v', var2='width')
#    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='width')


    # CORRELATION COEFFICIENTS
    bin_edges = [
        [0, 20], [20, 30], [30, 40], [40, 50],
        [50, 65], [65, 80], [80, 100], [100, 150],
        [150, 200], [200, 300], [300, 400], [400, 2000]
    ]
    lwp_masks = [(edge[0] < lwp_dict['mwr']) * (lwp_dict['mwr'] < edge[1]) for edge in bin_edges]

    correlation_LLT, correlaion_LLT_s = {}, {}
    correlation_LWP, correlation_LWP_s = {}, {}
    correlation_LCBH = {}

    for alg in liquid_masks2D.keys():
        correlation_LLT[alg + 'corr(LLT)'] = [UT.correlation_coefficient(lwp_dict['mwr'], llt_dict[alg]), ] + [
            UT.correlation_coefficient(lwp_dict['mwr'][_msk], llt_dict[alg][_msk]) for _msk in lwp_masks]

        correlaion_LLT_s[alg + 'corr(LLT)-s'] = [UT.correlation_coefficient(lwp_dict['mwr_s'], llt_dict[alg + '_s']), ] + [
            UT.correlation_coefficient(lwp_dict['mwr_s'][_msk], llt_dict[alg + '_s'][_msk]) for _msk in lwp_masks]

        correlation_LWP[alg + 'corr(LWP)'] = [UT.correlation_coefficient(lwp_dict['mwr'], lwp_dict[alg]), ] + [
            UT.correlation_coefficient(lwp_dict['mwr'][_msk], lwp_dict[alg][_msk]) for _msk in lwp_masks]

        correlation_LWP_s[alg + 'corr(LWP)-s'] = [UT.correlation_coefficient(lwp_dict['mwr_s'], lwp_dict[alg + '_s']), ] + [
            UT.correlation_coefficient(lwp_dict['mwr_s'][_msk], lwp_dict[alg + '_s'][_msk]) for _msk in lwp_masks]

        correlation_LCBH[alg + 'corr((L)CBH)'] = [UT.correlation_coefficient(CBH_dict[alg], CBH_dict['CEILO']), ] + [
            UT.correlation_coefficient(CBH_dict[alg][_msk], CBH_dict['CEILO'][_msk]) for _msk in lwp_masks]


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


    arr0 = np.zeros((len(lwp_masks)+1, 11))
    arr1 = np.zeros((len(lwp_masks)+1, 11))
    arr0[0, :] = _performance_metrics(_mask_cloudnet, liquid_masks2D['Voodoo'], ds['cloudnet_status'].values)
    arr1[0, :] = _performance_metrics(_mask_cloudnet, liquid_masks2D['Voodoo-post'], ds['cloudnet_status'].values)
    for i in range(1, len(lwp_masks)+1):
        _lwp_bin_mask = np.array([lwp_masks[i - 1]] * _mask_cloudnet.shape[1]).T
        tmp0 = _mask_cloudnet.copy()
        tmp1 = liquid_masks2D['Voodoo'].copy()
        tmp11 = liquid_masks2D['Voodoo-post'].copy()
        tmp2 = ds['cloudnet_status'].values.copy()
        tmp0[~_lwp_bin_mask] = 0
        tmp1[~_lwp_bin_mask] = 0
        tmp11[~_lwp_bin_mask] = 0
        tmp2[~_lwp_bin_mask] = 0
        arr0[i, :] = _performance_metrics(tmp0, tmp1, tmp2)
        arr1[i, :] = _performance_metrics(tmp0, tmp11, tmp2)

    # create pandas dataframe and save to csv
    print(f'\n')
    thr_names = ['', 'all', ] + [f'{edge[0]}<LWP<{edge[1]}' for edge in bin_edges]

    int_names = ['TP', 'TN', 'FP', 'FN']
    flt_names = ['precision', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score', 'Fhalf-score']
    corr_names = ['Voodoo-rho(LLT)', 'Cloudnet-rho(LLT)', 'Combination-rho(LLT)',
                  'Voodoo-rho(LWP)', 'Cloudnet-rho(LWP)', 'Combination-rho(LWP)']
    stats_list = [[thr_names, ] +
                  [[key + '',] + list(val.astype(int)) for key, val in zip(int_names, arr0[:, :4].T)] +
                  [[key + '', ] + list(val) for key, val in zip(flt_names, arr0[:, 4:].T)] +
                  [[key + '-post', ] + list(val.astype(int)) for key, val in zip(int_names, arr1[:, :4].T)] +
                  [[key + '-post', ] + list(val) for key, val in zip(flt_names, arr1[:, 4:].T)] +
                  [[alg, ] + val for alg, val in correlation_LLT.items()] +
                  [[alg, ] + val for alg, val in correlaion_LLT_s.items()] +
                  [[alg, ] + val for alg, val in correlation_LWP.items()] +
                  [[alg, ] + val for alg, val in correlation_LWP_s.items()] +
                  [[alg, ] + val for alg, val in correlation_LCBH.items()]]

    table1 = latextable.Texttable()
    table1.set_deco(latextable.Texttable.HEADER)
    table1.set_cols_align(["r", 'c', ] + ['c' for _ in bin_edges])
    table1.add_rows(stats_list[0])
    # print(table1.draw() + "\t\n")
    print(latextable.draw_latex(table1, caption=f"a"))

    df = pd.DataFrame(
        np.array(stats_list[0][1:])[:, 1:],
        columns=stats_list[0][0][1:],
        index=[n + '' for n in int_names] +
              [n + '' for n in flt_names] +
              [n + '-post' for n in int_names] +
              [n + '-post' for n in flt_names] +
              list(correlation_LLT.keys()) +
              list(correlaion_LLT_s.keys()) +
              list(correlation_LWP.keys()) +
              list(correlation_LWP_s.keys()) +
              list(correlation_LCBH.keys())
    )

    h.change_dir(f'{PRED_PATH}/{MODEL[:14]}/statistics/')
    df.to_csv(f'{date}.csv')
