#!/usr/bin/env python

import glob
import os
import sys

sys.path.append('/home/sdig/code/larda3/larda/')
sys.path.append('/home/sdig/code/Voodoo/')

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import pyLARDA.VIS_Colormaps as colors
import numpy as np
import xarray as xr
import seaborn as sns
import pandas as pd
import latextable
import scipy.interpolate

from cloudnetpy.categorize.atmos import find_cloud_bases

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

VOODOO_PATH = os.getcwd()

# QUICKLOOK_PATH = '/Users/willi/Documents/LaTeX/VOODOO/plots/'
PRED_PATH = '/home/sdig/code/Voodoo/torch_models/'
MEDIA_PATH = '/media/sdig/'

# MODEL = 'Vnet0x60cc7cd9-fnX-gpu0-VN'  # 3x3 conv
MODEL = 'Vnet0x60de1687-fnX-gpu0-VN'  # 3x3 conv

n_smoothing = 20
p = 0.40


def decimalhour2unix(dt, time):
    return np.array([x * 3600. + h.dt_to_ts(datetime.datetime.strptime(dt + '0000', '%Y%m%d%H%M')) for x in time])


def open_xarray_datasets(path):
    ds = xr.open_mfdataset(path, parallel=True, decode_times=False, )
    x = re.findall("\d{8}", path)[0]
    # convert time to unix
    ds.assign_coords(time=decimalhour2unix(str(x), ds['time'].values))
    ds['time'].attrs['units'] = 'Unix Time: Seconds since January 1st, 1970'
    return ds


# VOODOO cloud droplet likelyhood colorbar (viridis + grey below minimum value)
viridis = cm.get_cmap('viridis', 6)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[:1, :] = np.array([220 / 256, 220 / 256, 220 / 256, 1])
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
    if not (20201220 <= int(date) < 20201221):
            continue

    begin_dt = datetime.datetime.strptime(f'{date}0001', '%Y%m%d%H%M')
    end_dt = datetime.datetime.strptime(f'{date}2359', '%Y%m%d%H%M')


    # ts_lims = [begin_dt.replace(tzinfo=timezone.utc).timestamp(), end_dt.replace(tzinfo=timezone.utc).timestamp()]
    # rg_lims = [0, 12000]
    # plot_range = [0, 8]

    var_list = {
        'Z': [-50, 20],
        'v': [-4, 2],
        'width': [0, 1],
        'ldr': [-30, 0],
        'beta': [1.0e-7, 1.0e-4],
        'lwp': [-25, 500]
    }

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

    _ = [h.ts_to_dt(ts) for ts in ds['Z']['time'].values]
    idx_ts = [pyLARDA.helpers.argnearest(_, begin_dt), pyLARDA.helpers.argnearest(_, end_dt)]
    ceilo_cbh_interp = ds_ceilo['cbh'].values

    # # Load Voodoo Categorize & Classification File
    ds_voodoo_cat = open_xarray_datasets(cat_file)
    ds_vclass = open_xarray_datasets(class_file)
    # ds_voodoo_cat = open_xarray_datasets(PRED_PATH + f'{date}-{site}-categorize-limrad94-{MODEL}.nc')
    # ds_vclass = open_xarray_datasets(PRED_PATH + f'{date}-{site}-classification-limrad94-{MODEL}.nc')

    ## compute convective index
    # kappa_min, kappa_max = 0, 0.4
    # _kappa = UT.convection_index(ds['v'], dts=2, drg=1)
    # _ = np.ma.masked_less(_kappa, kappa_min)
    # _ = np.ma.masked_greater(_, kappa_max)
    # _kappa_mask = np.ma.getmaskarray(_)

    # copy datetime list and range list for plotting
    dt_list = [h.ts_to_dt(ts) for ts in ds['Z']['time'].values]
    rg_list = ds['Z']['height'].values

    rg_res = np.mean(np.diff(ds.height.values)) * 0.001

    #
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
    ds['voodoo_cd_prob'].attrs['colormap'] = ListedColormap(newcolors)
    ds['voodoo_cd_prob'].attrs['dimlabel'] = ['time', 'range']
    ds['voodoo_cd_prob'].height.attrs['units'] = 'km'

    # # Compute the adiabatic LWP using Karstens et al. 1994: https://link.springer.com/article/10.1007/BF01030057
    #
    # ## 3. Algorithm Development
    # ### 3.1 Parameterization of Cloud Liquid Water Content
    # Since radiosonde observations do not contain direct information about the cloud liquid water content (LWC) we developed a parameterization. If relative humidity exceeds 95% in a layer, it is assumed that  he radiosonde has passed a cloud. For each level in a cloudy layer the adiabatic LWC is calculated with:

    # # interpolate temperature and presser from model to cloudnet res (linear)

    ts, rg = ds['time'].values, ds['height'].values
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

    # if site == 'punta-arenas':
    _ts_from_ceilo = np.array([(dt64 - ds_ceilo['time'][0].values) / np.timedelta64(1, 's') for dt64 in ds_ceilo['time'].values]) / 3600
    # else:
    #    _ts_from_ceilo = ds_ceilo['time'].values

    f = scipy.interpolate.interp1d(
        _ts_from_ceilo,
        ds_ceilo['cbh'][:, 0].values,
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )
    _cbh = f(ts)[:]

    _ccl = ds_cnclass['target_classification'].values
    _cst = ds_cnclass['detection_status'].values
    _vcl = ds_vclass['target_classification'].values
    _mask_voodoo = (ds_voodoo_cat['pred_liquid_prob_1'].values > p) * (ds_voodoo_cat['pred_liquid_prob_1'].values < 0.90)
    _mask_cloudnet = (_ccl == 1) + (_ccl == 3) + (_ccl == 5) + (_ccl == 7)
    _mask_combi = (_vcl == 1) + (_vcl == 3) + (_vcl == 5) + (_vcl == 7)
    _is_clutter = _ccl > 7
    _is_rain = np.array([ds['is_rain'].values] * ds['height'].size, dtype=bool).T
    _cloud_mask = _mask_cloudnet + (_ccl == 2) + (_ccl == 4) + (_ccl == 6)

    _TP_mask = _mask_cloudnet * _mask_voodoo
    _TN_mask = (~_mask_cloudnet * ~_mask_voodoo) * _cloud_mask

#    fig, ax = plt.subplots(nrows=2)
#    ax[0].pcolormesh(_TP_mask.T)
#    ax[1].pcolormesh(_TN_mask.T)
#    fig.savefig('test.png')

    _mask_below_cloudbase = np.full(_ccl.shape, False)
    for its, icb in enumerate(_cbh):
        if icb > 250:
            _mask_below_cloudbase[its, :h.argnearest(rg_list, icb)] = True

    liquid_masks2D = {
        'Voodoo': _mask_voodoo,
        'CloudnetPy': _mask_cloudnet,
        'Combination': _mask_combi * ~_mask_below_cloudbase * ~_is_clutter * ~_is_rain,
        'Voodoo-post': _mask_voodoo * ~_mask_below_cloudbase * ~_is_clutter * ~_is_rain + _TP_mask
    }
    # for key, val in liquid_masks2D.items():
    #    liquid_masks2D[key][:,:] = False

    # # calculate cloud bases, cloud tops, liquid water content, liqudi water lap, liquid layer thickness
    # cloud tops and cloud bases
    bases_tops_dict = {
        alg: UT.find_bases_tops(liquid_masks2D[alg], ds.height.values)
        for alg in liquid_masks2D.keys()
    }
    bases = {
        alg: UT.get_bases_or_tops([h.ts_to_dt(ts) for ts in ds['Z']['time'].values], bases_tops_dict[alg], key='cb')
        for alg in liquid_masks2D.keys()
    }

    # liquid water content
    lwc_dict = {}
    for key, mask in liquid_masks2D.items():
        _lwc = UT.adiabatic_liquid_water_content(_T, _p, mask, delta_h=float(np.mean(np.diff(ds.height.values))))
        _lwc[_lwc > 1000] = np.nan
        _lwc[_lwc < 1] = np.nan
        lwc_dict[key] = np.ma.masked_invalid(_lwc)  # * (1 + np.ma.masked_invalid(ds['voodoo_cd_prob'].values))

    try:
        _lwp = ds['lwp'].values
        _lwp[_lwp > 5000] = np.nan
        _lwp[_lwp < 1] = np.nan
        a = pd.Series(_lwp)
    except:
        print('skiped')
        continue

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
    CBH_dict = {}
    for key in liquid_masks2D.keys():
        _tmp = np.argmax(liquid_masks2D[key] == 1, axis=1)
        CBH_dict[key] = np.ma.masked_less_equal([rg_list[ind_rg] for ind_rg in _tmp], 300.0)
    CBH_dict['CEILO'] = _cbh

    # with plt.style.context(['science', 'ieee']):
    fig, ax = plt.subplots(nrows=2, figsize=(14, 9))
    ax[0].pcolormesh(ts, rg_list, liquid_masks2D['Voodoo'].T)
    ax[0].scatter(ts, _cbh, s=0.01)
    ax[1].pcolormesh(_TN_mask.T)
    fig.savefig('test.png')


    if True:
        fig, ax = plt.subplots(figsize=(12, 6))
        _colors = np.array([
            [255, 255, 255, 255],
            [220, 220, 220, 50],
            [180, 55, 87, 255],
            [32, 159, 243, 255],
            [206, 188, 137, 255],
        ]) / 255
        _colornames = ["clear sky", "no liquid", "both", "ANN only", "Cloudnet only"]
        # loop over the different box thresholds
        combi_liq_mask = np.zeros(_cloud_mask.shape)
        combi_liq_mask[~_cloud_mask] = -1
        combi_liq_mask[liquid_masks2D['Voodoo'] * liquid_masks2D['CloudnetPy']] = 1
        combi_liq_mask[liquid_masks2D['Voodoo'] * ~liquid_masks2D['CloudnetPy']] = 2
        combi_liq_mask[~liquid_masks2D['Voodoo'] * liquid_masks2D['CloudnetPy']] = 3
        # calculate layer thickness for all different categories
        pcmesh = ax.pcolormesh(ts, rg_list, combi_liq_mask.T,
                               cmap=matplotlib.colors.ListedColormap(tuple(_colors), "colors5"))
        ax.contour(ts, rg_list, _cloud_mask.T, colors=['black'], linewidths=[0.1], nchunk=10, alpha=0.2)
        ax.scatter(ts, _cbh, s=0.01, c='black', marker='*')

        cont = ax.contour(ts, rg_list, (_T - 273.15).T, levels=[-38, -25, -15, -10, -5, 0],
                          linestyles='dashed', colors=['black'], linewidths=[0.5], alpha=0.5)
        ax.clabel(cont, inline=1, fmt='%1.0f°C')

        cbar = fig.colorbar(pcmesh, ax=ax, pad=0.05)
        cbar.set_ticks(list(np.arange(-0.5, len(_colornames) - 1)))
        cbar.ax.set_yticklabels(_colornames)
        fig.savefig('test.png', dpi=450)


    try:
        ds = ds.rename({'time': 'ts'})
        ds = ds.rename({'height': 'rg'})
    except:
        pass

    #
    #    n_samples = 100
    #
    #    perm = np.argsort(lwp_dict['mwr_s'])
    #    sorted_lwp = lwp_dict['mwr_s'][perm]
    #    bin_list = []
    #    for k in range(10):
    #        for i in range(len(lwp_dict['mwr_s'])):
    #            cum_sum = np.cumsum(sorted_lwp[:i])
    #            if cum_sum.size < n_samples:
    #                sorted_lwp = sorted_lwp[i:]
    #                bin_list.append(i)
    #                continue
    bin_edges = [[0, 20], [20, 30], [30, 40], [40, 50], [50, 65], [65, 80], [80, 100], [100, 150], [150, 200], [200, 300]
        , [300, 400], [400, 2000]]

    lwp_masks = [(edge[0] < lwp_dict['mwr']) * (lwp_dict['mwr'] < edge[1]) for edge in bin_edges]

    ##########################################################################################################################################################
    ##########################################################################################################################################################
    # CORRELATION COEFFICIENTS
    correlation_LLT, correlaion_LLT_s = {}, {}
    correlation_LWP, correlation_LWP_s = {}, {}
    correlation_LCBH = {}




#    ###### DBG begin
#    print('corrllt: ', UT.correlation_coefficient(lwp_dict['mwr'], llt_dict['Voodoo']))
#    print('corrlwp: ', UT.correlation_coefficient(lwp_dict['mwr'], lwp_dict['Voodoo']))
#
#    import matplotlib.colors as colors
#
#    fig, ax = plt.subplots(nrows=3)
#    ax[0].pcolormesh(np.ma.masked_less_equal(lwc_dict['Voodoo'].T, 0.0),
#                     norm = colors.LogNorm(vmin=1, vmax=lwc_dict['Voodoo'].max()))
#    ax[1].pcolormesh(liquid_masks2D['Voodoo'].T)
#    ax[2].plot(np.histogram(lwc_dict['Voodoo'], range=(0, 20)))
#    fig.savefig('test.png')
#    ######DBG end


    for alg in liquid_masks2D.keys():
        correlation_LLT[alg + 'corr(LLT)'] = [UT.correlation_coefficient(lwp_dict['mwr'], llt_dict[alg]), ] + \
                                             [UT.correlation_coefficient(lwp_dict['mwr'][_msk], llt_dict[alg][_msk]) for _msk in lwp_masks]

        correlaion_LLT_s[alg + 'corr(LLT)-s'] = [UT.correlation_coefficient(lwp_dict['mwr_s'], llt_dict[alg + '_s']), ] + \
                                                [UT.correlation_coefficient(lwp_dict['mwr_s'][_msk], llt_dict[alg + '_s'][_msk]) for _msk in lwp_masks]

        correlation_LWP[alg + 'corr(LWP)'] = [UT.correlation_coefficient(lwp_dict['mwr'], lwp_dict[alg]), ] + \
                                             [UT.correlation_coefficient(lwp_dict['mwr'][_msk], lwp_dict[alg][_msk]) for _msk in lwp_masks]

        correlation_LWP_s[alg + 'corr(LWP)-s'] = [UT.correlation_coefficient(lwp_dict['mwr_s'], lwp_dict[alg + '_s']), ] + \
                                                 [UT.correlation_coefficient(lwp_dict['mwr_s'][_msk], lwp_dict[alg + '_s'][_msk]) for _msk in lwp_masks]

        correlation_LCBH[alg + 'corr((L)CBH)'] = [UT.correlation_coefficient(CBH_dict[alg], CBH_dict['CEILO']), ] + \
                                                 [UT.correlation_coefficient(CBH_dict[alg][_msk], CBH_dict['CEILO'][_msk]) for _msk in lwp_masks]


    #### print('Correlation MWR-LWP with adiabatic LWP from liquid pixel mask')
    ##########################################################################################################################################################
    #########################################################################################################################################################

    # # Scores of predictive performance all Liquid Water Path Values thresholds
    arr = UT.compute_metrics(ds['cloudnet_class'].values, ds['cloudnet_status'].values, liquid_masks2D['Voodoo'], lwp_masks)
    arr2 = UT.compute_metrics(ds['cloudnet_class'].values, ds['cloudnet_status'].values, liquid_masks2D['Voodoo-post'], lwp_masks)

    # create pandas dataframe and save to csv
    print(f'\n')
    thr_names = ['', 'all', ] + [f'{edge[0]}<LWP<{edge[1]}' for edge in bin_edges]

    int_names = ['TP', 'TN', 'FP', 'FN']
    flt_names = ['precision', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score', 'Jaccard-index']
    corr_names = ['Voodoo-rho(LLT)', 'Cloudnet-rho(LLT)', 'Combination-rho(LLT)',
                  'Voodoo-rho(LWP)', 'Cloudnet-rho(LWP)', 'Combination-rho(LWP)']
    stats_list = [[thr_names, ] +
                  [[key, ] + list(val.astype(int)) for key, val in zip(int_names, arr[:, :4].T)] +
                  [[key, ] + list(val) for key, val in zip(flt_names, arr[:, 4:].T)] +
                  [[key + '-post', ] + list(val.astype(int)) for key, val in zip(int_names, arr2[:, :4].T)] +
                  [[key + '-post', ] + list(val) for key, val in zip(flt_names, arr2[:, 4:].T)] +
                  [[alg, ] + val for alg, val in correlation_LLT.items()] +
                  [[alg, ] + val for alg, val in correlaion_LLT_s.items()] +
                  [[alg, ] + val for alg, val in correlation_LWP.items()] +
                  [[alg, ] + val for alg, val in correlation_LWP_s.items()] +
                  [[alg, ] + val for alg, val in correlation_LCBH.items()]]

    table1 = latextable.Texttable()
    table1.set_deco(latextable.Texttable.HEADER)
    table1.set_cols_align(["r", 'c', ] + ['c' for _ in bin_edges])
    table1.add_rows(stats_list[0])
    # print(table1.draw() + "\n")
    print(latextable.draw_latex(table1, caption=f"a"))

    df = pd.DataFrame(
        np.array(stats_list[0][1:])[:, 1:],
        columns=stats_list[0][0][1:],
        index=int_names + flt_names +
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

    y = 5
