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
import csv

# optionally configure the logging
# StreamHandler will print to console
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates

_colornames = ["clear\nsky", "non-CD\next.", "CD\next.", "TN", "TP", "FP", "FN"]
_colors = np.array([
    [255, 255, 255, 255],
    [0, 0, 0, 75],
    [70, 74, 185, 255],
    [0, 0, 0, 15],
    [108, 255, 236, 255],
    [180, 55, 87, 255],
    [255, 165, 0, 155],
    #[111, 234,  92, 255],
#    [206, 188, 137, 255],
]) / 255

import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import libVoodoo.Utils as UT

#MODEL = 'Vnet0x60de1687-fnX-gpu0-VN'
MODEL = 'Vnet0x615580bf-fn1-gpu0-VN'

PRED_PATH = '/home/sdig/code/Voodoo/torch_models/'
MEDIA_PATH = '/media/sdig/'

CODE_PATH = '/Users/willi/Documents/LaTeX/VOODOO/code/'

QUICKLOOK_PATH = f'{PRED_PATH}/{MODEL[:14]}/plots/analyzer/'

n_smoothing = 20
n_cloud_edge = 3


def decimalhour2unix(dt, time):
    return np.array([x * 3600. + h.dt_to_ts(datetime.datetime(int(dt[:4]), int(dt[4:6]), int(dt[6:]), 0, 0, 0)) for x in time])


def open_xarray_datasets(path):
    ds = xr.open_mfdataset(path, parallel=True, decode_times=False, )
    x = re.findall("\d{8}", path)[0]

    # convert time to unix
    ds = ds.assign_coords(time=("time", decimalhour2unix(str(x), ds['time'].values)))
    ds['time'].attrs['units'] = 'Unix Time: Seconds since January 1st, 1970'

    return ds


_, agrs, kwargs = UT.read_cmd_line_args()

if 'YYMM' in kwargs:
    YYMM = kwargs['YYMM']
else:
    YYMM = '201908'



if int(YYMM) > 202001:
    site = 'LIM'
    p = 0.4 if 'p' not in kwargs else kwargs['p']
else:
    site = 'punta-arenas'
    p = 0.5 if 'p' not in kwargs else kwargs['p']

new_cloudnet_cat_files = glob.glob(f'{PRED_PATH}/{MODEL[:14]}/nc/*{YYMM}*-{site}-categorize-*{MODEL}*.nc')
new_cloudnet_class_files = glob.glob(f'{PRED_PATH}/{MODEL[:14]}/nc/*{YYMM}*-{site}-classification-*{MODEL}*.nc')
new_cloudnet_cat_files.sort()
new_cloudnet_class_files.sort()

h0 = 117 if site == 'LIM' else 7

for cat_file, class_file in zip(new_cloudnet_cat_files, new_cloudnet_class_files):

    date = cat_file[cat_file.rfind('/') + 1:cat_file.rfind('/') + 9]
    print(f'\n {date} \n')


    csv_row_list = [['begin_dt', 'end_dt'], [date+'-000005', date+'-235955']]
    rg_lims = [0, 12000]
    print('use default case', csv_row_list)

    for row in csv_row_list[1:]:
        date = row[0][:8]

        begin_dt = datetime.datetime.strptime(f'{row[0]}', '%Y%m%d-%H%M%S')
        end_dt = datetime.datetime.strptime(f'{row[1]}', '%Y%m%d-%H%M%S')


        d0_str = f'{begin_dt.year}{str(begin_dt.month).zfill(2)}{str(begin_dt.day).zfill(2)}'
        d1_str = f'{end_dt.year}{str(end_dt.month).zfill(2)}{str(end_dt.day).zfill(2)}'

        ts_lims = [begin_dt.replace(tzinfo=timezone.utc).timestamp(), end_dt.replace(tzinfo=timezone.utc).timestamp()]
        var_list = {'Z': [-50, 20], 'v': [-4, 2], 'width': [0, 1], 'ldr': [-30, 0], 'beta': [1.0e-7, 1.0e-4], 'lwp': [-25, 500]}

        if site == 'LIM':
            MEDIA_PATH = '/media/sdig/leipzig/'
            cn_cat_file = f'{MEDIA_PATH}/cloudnet/processed/limrad94/categorize-py/{date[:4]}/{date}-{site}-categorize-limrad94.nc'
            cn_class_file = f'{MEDIA_PATH}/cloudnet/products/limrad94/classification-cloudnetpy/{date[:4]}/{date}-{site}-classification-limrad94.nc'
            ds_ceilo = xr.open_mfdataset(f'{MEDIA_PATH}/instruments/ceilim/data/Y{date[:4]}/M{date[4:6]}/{date}_Leipzig_CHM200114_000.nc')
        else:
            MEDIA_PATH = '/media/sdig/LACROS/cloudnet/data/punta-arenas/'
            cn_cat_file = f'{MEDIA_PATH}/processed/limrad94/categorize-py/{date[:4]}/{date}-{site}-categorize-limrad94.nc'
            cn_class_file = f'{MEDIA_PATH}/products/limrad94/classification-cloudnetpy/{date[:4]}/{date}-{site}-classification-limrad94.nc'
            ds_ceilo = xr.open_mfdataset(MEDIA_PATH + f'/calibrated/chm15x/{date[:4]}/{date}_{site}_chm15x.nc')

        # # Load CloudnetPy
        ds = open_xarray_datasets(cn_cat_file)
        ds_cnclass = open_xarray_datasets(cn_class_file)
        ds_voodoo_cat = open_xarray_datasets(cat_file)
        ds_vclass = open_xarray_datasets(class_file)

        dt_list = [h.ts_to_dt(ts) for ts in ds['Z']['time'].values]
        rg_list = ds['Z']['height'].values

        _ = [h.ts_to_dt(ts) for ts in ds['Z']['time'].values]
        ceilo_cbh_interp = ds_ceilo['cbh'].values

        clear_sky_mask = ds['Z'].values > 1.0e5

        i0 = (begin_dt.replace(hour=0, minute=0, second=0) - datetime.datetime(1970, 1, 1)).total_seconds()
        ts, rg = ds['time'].values, ds['height'].values
        _ts_from_ceilo = np.array([
            (dt64 - ds_ceilo['time'][0].values) / np.timedelta64(1, 's') for dt64 in ds_ceilo['time'].values
        ]) / 3600

        _interp_kwargs = {'kind': 'linear', 'copy': True, 'bounds_error': False, 'fill_value': None}

        f = scipy.interpolate.interp2d(
            ds['model_time'].values, ds['model_height'].values, ds['temperature'].values.T, **_interp_kwargs
        )
        _T = f(ts, rg)[:, :].T

        f = scipy.interpolate.interp2d(
            ds['model_time'].values, ds['model_height'].values, ds['pressure'].values.T, **_interp_kwargs
        )
        _p = f(ts, rg)[:, :].T

        _cbh_list = []
        for i in range(ds_ceilo['cbh'].shape[1]):
            f = scipy.interpolate.interp1d(_ts_from_ceilo, ds_ceilo['cbh'][:, i].values, **_interp_kwargs)
            _cbh = f((ts - i0) / 3600)[:] + h0
            idx_start, idx_stop = h.argnearest(dt_list, begin_dt), h.argnearest(dt_list, end_dt)
            if len(dt_list) == idx_stop - idx_start + 1:
                idx_ts = slice(h.argnearest(dt_list, begin_dt), idx_stop + 1)
            else:
                idx_ts = slice(h.argnearest(dt_list, begin_dt), idx_stop)

            _cbh_list.append(_cbh[idx_ts])

        _cbhN = np.nanmax(_cbh_list)
        _cbh = _cbh_list[0]


        # copy datetime list and range list for plotting

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
        _is_lidar_only = _cst == 4
        _is_good_radar_lidar = _cst == 1
        _is_clutter = _ccl > 7
        _is_rain = np.array([ds['is_rain'].values] * ds['rg'].size, dtype=bool).T
        _is_falling = (_ccl == 2) * (ds['v'].values < -3)
        _beta_mask = (1.0e-8 < ds["beta"].values) * (ds["beta"].values < 99)

        # reclassify all fast-falling hydrometeors and insects/clutter to non-CD
        _mask_voodoo[_is_falling] = False
        _mask_voodoo[_is_clutter] = False

        _cloud_mask = _mask_cloudnet + (_ccl == 2) + (_ccl == 4) + (_ccl == 6)
        _cloud_mask = UT.remove_cloud_edges(_cloud_mask, n=1)
        _cloud_mask[_is_lidar_only] = False
        _cloud_mask[_is_clutter] = False
        _cloud_mask[_is_rain] = False

        _TP_mask = _cloud_mask * (_mask_cloudnet * _mask_voodoo)
        _FP_mask = _cloud_mask * (~_mask_cloudnet * _mask_voodoo)
        _FN_mask = _cloud_mask * (_mask_cloudnet * ~_mask_voodoo)
        _TN_mask = _cloud_mask * (~_mask_cloudnet * ~_mask_voodoo)

        _mask_below_cloudbase = np.full(_ccl.shape, False)
        for its, icb in enumerate(_cbh):
            if icb > 150:
                idx_cb = h.argnearest(rg_list, icb)
                _mask_below_cloudbase[its, :idx_cb] = True

                _TN_mask[its, idx_cb + 1:] = False
                _FP_mask[its, idx_cb + 1:] = False

        # # create dictionary with liquid pixel masks
        liquid_masks2D = {
            'Voodoo': _mask_voodoo * _cloud_mask,
            'CloudnetPy': _mask_cloudnet * _cloud_mask,
        }

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
        rg_res = np.mean(np.diff(ds['rg'].values)) * 0.001

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
            CBH_dict[key] = np.ma.masked_less_equal([rg_list[ind_rg] for ind_rg in _tmp], 200.0)
            CBH_dict[key] = np.ma.masked_invalid(CBH_dict[key])

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

            #    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='v')
    #    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='v_sigma')
    #    errormatrix_histo_list = _histo_errormatrix(var1='v', var2='v_sigma')
    #    errormatrix_histo_list = _histo_errormatrix(var1='width', var2='v_sigma')
    #    errormatrix_histo_list = _histo_errormatrix(var1='v', var2='width')
    #    errormatrix_histo_list = _histo_errormatrix(var1='Z', var2='width')

        ##########################################################################################################################################################
        ##########################################################################################################################################################


        # CORRELATION COEFFICIENTS
        bin_edges = [
            [0, 2000],
            [0, 25], [25, 50], [50, 100], [100, 150],
            [150, 200], [200, 300], [300, 400], [400, 2000]
        ]
        lwp_masks = [
            (edge[0] < lwp_dict['mwr']) * (lwp_dict['mwr'] < edge[1]) for edge in bin_edges
        ]
        lwp_masks_idx = [
            np.argwhere((edge[0] < lwp_dict['mwr']) * (lwp_dict['mwr'] < edge[1])) for edge in bin_edges
        ]

        correlation_LLT, correlation_LLT_s = {}, {}
        correlation_LWP, correlation_LWP_s = {}, {}
        correlation_LCBH = {}
        correlations = {}

        corr_matrix = np.array((len(lwp_masks), 5))
        for alg in liquid_masks2D.keys():
            correlation_LLT_s[alg + 'corr(LLT)-s'] = [
                UT.ma_correlation_coefficient(lwp_dict['mwr_s'][_msk], llt_dict[alg + '_s'][_msk]) for _msk in lwp_masks
            ]
            correlation_LWP_s[alg + 'corr(LWP)-s'] = [
                UT.ma_correlation_coefficient(lwp_dict['mwr_s'][_msk], lwp_dict[alg + '_s'][_msk]) for _msk in lwp_masks
            ]
            correlation_LCBH[alg + 'corr((L)CBH)'] = [
                UT.ma_correlation_coefficient(CBH_dict[alg][_msk], CBH_dict['CEILO'][_msk]) for _msk in lwp_masks
            ]

            correlations[alg] = np.array([
                correlation_LLT_s[alg + 'corr(LLT)-s'],
                correlation_LWP_s[alg + 'corr(LWP)-s'],
                correlation_LCBH[alg + 'corr((L)CBH)']
            ])


        int_columns = ['TP', 'TN', 'FP', 'FN']
        flt_columns = ['precision', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score']
        corr_columns = ['Vr2(LLT)', 'Vr2(LWP)', 'Vr2(LCBH)', 'Cr2(LLT)', 'Cr2(LWP)','Cr2(LCBH)']
        extra_columns = ['n_time_steps', 'ETS']

        num_columns = len(int_columns) + len(flt_columns) + len(corr_columns) + len(extra_columns)
        ##### ds

        title_list = ['TP', 'TN', 'FP', 'FN']
        mask_list = [_TP_mask, _TN_mask, _FP_mask, _FN_mask]
        n_time_steps = ds['rg'].size

        arr0 = np.zeros((len(lwp_masks), num_columns), dtype=float)
        for i in range(len(lwp_masks)):
            _lwp_bin_mask = np.array([lwp_masks[i]] * n_time_steps).T
            _lwp_bin_mask = _lwp_bin_mask *_cloud_mask

            n_masks = []
            for i_mask in mask_list:
                n = np.count_nonzero( i_mask * _lwp_bin_mask )
                n_masks.append(n)

            sum_stats = UT.performance_metrics(*n_masks)
            sum_stats_list = [val for val in sum_stats.values()]
            arr0[i, :] = np.array(
                n_masks +
                sum_stats_list +
                list(correlations['Voodoo'][:, i]) +
                list(correlations['CloudnetPy'][:, i]) +
                [np.count_nonzero(np.any(_lwp_bin_mask, axis=1)), UT.equitable_thread_score(*n_masks)]
            )

        # create pandas dataframe and save to csv
        print(f'\n')

        stats_list = [
            [['', ] + int_columns + flt_columns + corr_columns + extra_columns] +
            [[f'{site}-lwp-bin{i}', ] + list(val) for i, val in enumerate(arr0[:, :])]
        ]

        h.change_dir(f'{PRED_PATH}/{MODEL[:14]}/statistics/')

        with open(f'{date}-make_csv2.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(stats_list[0][0])
            for i, cloud in enumerate(stats_list[0][1:]):
                writer.writerow(cloud)



        def _quicklook():
            fig, ax = plt.subplots(figsize=(12,6))
            # H
            combi_liq_mask = np.zeros(_cloud_mask.shape)

            _non_CD_ext_mask = _cloud_mask * ~_TP_mask * ~_TN_mask * ~_FP_mask * ~_FN_mask
            _CD_ext_mask = liquid_masks2D['Voodoo'] * ~_TP_mask * ~_FP_mask
            combi_liq_mask[_non_CD_ext_mask] = 1
            combi_liq_mask[_CD_ext_mask] = 2
            combi_liq_mask[_TN_mask] = 3
            combi_liq_mask[_TP_mask] = 4
            combi_liq_mask[_FP_mask] = 5
            combi_liq_mask[_FN_mask] = 6
            combi_liq_mask[~_cloud_mask] = 0

            pcmesh = ax.pcolormesh(
                dt_list, rg_list * 0.001, combi_liq_mask.T,
                cmap=matplotlib.colors.ListedColormap(tuple(_colors), "colors5"),
                vmin=0, vmax=7
            )
            ax.set_title(arr0[0, :])


            cbaxes = inset_axes(ax, width="50%", height="5%", loc='upper left')

            cbar = fig.colorbar(pcmesh, cax=cbaxes, pad=0.05, orientation="horizontal")
            cbar.set_ticks(np.arange(0.5, len(_colornames) + 0.5))
            cbar.ax.set_xticklabels(_colornames, fontsize=8)
            ax.set_ylim([0, 12])
            ax.set_ylabel('Height [km]')
            ax.set_xlabel('Time [UTC]')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.subplots_adjust(top=0.9)
            fig.savefig(f'QL_{begin_dt:%Y%m%d}.png', dpi=250, facecolor='white',)

        #_quicklook()
        #print('quicklook True')


        dbg_dummy=6