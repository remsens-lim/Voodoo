#!/home/sdig/anaconda3/bin/python
import datetime
import glob
import os
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import toml
import torch
import xarray as xr
from cloudnetpy.plotting import generate_figure
from cloudnetpy.products import generate_classification
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter

import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
from libVoodoo.Loader import cn_from_hourly_zarr, tensor_from_hourly_zarr
from libVoodoo.Plot import load_xy_style

sys.path.append('/Users/willi/code/python/larda3/larda/')
import pyLARDA.helpers as h
import pyLARDA.VIS_Colormaps as VIS_colors

import warnings
warnings.filterwarnings("ignore")

pt_models_path = f'torch_models/'

torch.set_num_interop_threads(2) # Inter-op parallelism
torch.set_num_threads(2) # Intra-op parallelism

def dh_to_ts(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh * 3600))


def VoodooPredictor(
        date_str,
        hourly_path,
        categorize_path,
        modelfile,
        liquid_threshold,
        site,
        **torch_settings
):
    torch_settings.update({'dev': 'cpu', 'task': 'test'})

    print(f'Loading date ...... {date_str}')

    hourly_files = sorted(glob.glob(hourly_path + f'{date_str}*.zarr'))

    X, y, Time, status, mask = tensor_from_hourly_zarr(hourly_files)
    n_classes = 2

    X = torch.unsqueeze(X, dim=1)
    X = torch.transpose(X, 3, 2)

    model = TM.VoodooNet(X.shape, n_classes, **torch_settings)
    model.load_state_dict(
        torch.load(
            f'{pt_models_path}/{modelfile[:14]}/{modelfile}',
            map_location=model.device)['state_dict']
    )

    prediction = model.predict(X, batch_size=1024)
    prediction = prediction.to('cpu')

    Nts, Nrg = mask.shape
    probabilities = TM.VoodooNet.reshape(prediction, mask, (Nts, Nrg, n_classes))

    smoothed_probs = np.ma.zeros((Nts, Nrg, n_classes))
    for i in range(n_classes):
        tmp = gaussian_filter(probabilities[:, :, i], sigma=1)
        smoothed_probs[:, :, i] = np.ma.masked_where(mask, tmp)

    # uncertain pixels
    cond1 = smoothed_probs[:, :, 0] > liquid_threshold[0]
    cond2 = (1 - np.sum(smoothed_probs, axis=2)) > 0.15
    voodoo_liquid_mask = cond1 * ~cond2

    # load the original categorize file
    categorize_file = glob.glob(f'/{categorize_path}/{str(date_str)[:4]}/{date_str}*.nc')[0]
    cat_xr = xr.open_dataset(categorize_file, decode_times=False)
    cloud_mask = cat_xr['Z'].values > -100

    def _fill_time_gaps():

        # print('Cloudnetpy has dimensins (ts,rg) :: ', cat_xr['category_bits'].shape)
        # print('   Voodoo has dimensions (ts,rg) :: ', probabilities.shape[:2])
        n_ts_cloudnet_cat, n_rg_cloudnet_cat = cat_xr['category_bits'].shape
        ts_unix_cloudnetpy = np.array([UT.dt_to_ts(dh_to_ts(date_str, dh)) for dh in cat_xr['time'].values])

        _tmp_master_ts = ts_unix_cloudnetpy.astype(int)
        _tmp_slave_ts = Time.astype(int)
        uniq, uniq_idx = np.unique(_tmp_slave_ts, return_index=True, )
        _tmp_slave_ts = _tmp_slave_ts[uniq_idx]
        mask_final = voodoo_liquid_mask[uniq_idx, :]
        status_final = status[uniq_idx, :]
        smoothed_probs_final = smoothed_probs[uniq_idx, :]

        ts_unix_cloundetpy_mask = np.full(n_ts_cloudnet_cat, False)
        m_new = np.zeros((n_ts_cloudnet_cat, n_rg_cloudnet_cat))
        s_new = np.zeros((n_ts_cloudnet_cat, n_rg_cloudnet_cat))
        p_new = np.zeros((n_ts_cloudnet_cat, n_rg_cloudnet_cat, 2))
        cnt = 0
        for i, mts in enumerate(_tmp_master_ts):
            if cnt == _tmp_slave_ts.size:
                break
            if mts == _tmp_slave_ts[cnt]:
                ts_unix_cloundetpy_mask[i] = True
                m_new[i, :]  = mask_final[cnt, :]
                s_new[i, :]  = status_final[cnt, :]
                p_new[i, :]  = smoothed_probs_final[cnt, :]
                cnt += 1
        return m_new, s_new, p_new

    mask_new, status_new, smoothed_probs_new = _fill_time_gaps()


    def _adjust_cloudnetpy_bits():
        n_ts_cloudnet_cat, n_rg_cloudnet_cat = cat_xr['category_bits'].shape
        bits_unit = cat_xr['category_bits'].values.astype(np.uint8)
        new_bits = bits_unit.copy()

        for ind_time in range(n_ts_cloudnet_cat):
            for ind_range in range(n_rg_cloudnet_cat):
                if mask_new[ind_time, ind_range]:
                    if status_new[ind_time, ind_range] in [3]:
                        continue  # skip good radar & lidar echo pixel
                    bit_rep = np.unpackbits(bits_unit[ind_time, ind_range])
                    bit_rep[-1] = 1  # set droplet bit
                    new_bits[ind_time, ind_range] = np.packbits(bit_rep)
        return new_bits

    cat_xr['category_bits'].values = _adjust_cloudnetpy_bits()

    os.makedirs(f'{pt_models_path}/{modelfile[:14]}/nc/', exist_ok=True)
    cat_xr.attrs['postprocessor'] = f'Voodoo_v2.0, Modelname: {modelfile[:-3]}'
    #
    cat_xr[f'liquid_probability'] = cat_xr['Z'].copy()
    cat_xr[f'liquid_probability'].values = np.ma.masked_where(~cloud_mask, smoothed_probs_new[:, :, 0])
    cat_xr[f'liquid_probability'].attrs = {
        'comment': "This variable contains information about the likelihood of cloud droplet\n"
                   f"availability, predicted by the {cat_xr.attrs['postprocessor']} classifier.",
        'definition': "\nProbability 1 means most likely cloud droplets are present,\n"
                      "probability of 0 means no cloud droplets are available, respectively.\n",
        'units': "1",
        'long_name': f"Predicted likelihood of cloud droplet present."}
    #
    #
    cat_xr[f'noliquid_probability'] = cat_xr['Z'].copy()
    cat_xr[f'noliquid_probability'].values = np.ma.masked_where(~cloud_mask, smoothed_probs_new[:, :, 1])
    cat_xr[f'noliquid_probability'].attrs = {
        'comment': "This variable contains information about the likelihood of present cloud droplet,\n"
                   f"predicted by the {cat_xr.attrs['postprocessor']} classifier.",
        'definition': "\nProbability 1 means most likely no cloud droplets are present,\n"
                      "probability of 0 means no cloud droplets are available, respectively.\n",
        'units': "1",
        'long_name': f"Predicted likelihood of no cloud droplet present."}
    #
    CAT_FILE_NAME = f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-categorize-limrad94-{modelfile[:-3]}.nc'
    cat_xr.to_netcdf(path=CAT_FILE_NAME, format='NETCDF4', mode='w')
    print(f"\nnew categorize file saved: {CAT_FILE_NAME}")
    # generate classification with new bit mask

    class_filename = CAT_FILE_NAME.replace('categorize', 'classification')
    generate_classification(CAT_FILE_NAME, class_filename)
    generate_figure(
        class_filename,
        ['target_classification', 'detection_status'],
        max_y=12,
        show=True,
        image_name=f'{class_filename[:-3]}.png'
    )


def VoodooAnalyser(
        date_str,
        site,
        modelfile,
        liquid_threshold,
        n_lwp_smoothing=600,
):

    if int(str(date_str)[:4]) > 2019:
        CLASSIFICATION_PATH = f'/media/sdig/leipzig/cloudnet/products-hatpro/limrad94/classification/{str(date_str)[:4]}'
        CEILOMETE_PATH = f'/media/sdig/leipzig/instruments/ceilim/data/Y{date_str[:4]}/M{date_str[4:6]}/{date_str}_Leipzig_CHM200114_000.nc'
        h0 = 117
    else:
        CLASSIFICATION_PATH = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/products-hatpro/limrad94/classification/{str(date_str)[:4]}'
        CEILOMETE_PATH = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/calibrated/chm15x/{date_str[:4]}/{date_str}_punta-arenas_chm15x.nc'
        h0 = 7

    # original & new categorize file
    CLOUDNET_CLASS_FILE = f'{CLASSIFICATION_PATH}/{date_str}-{site}-classification-limrad94.nc'
    CLOUDNET_VOODOO_CAT_FILE = glob.glob(f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-categorize-limrad94-{modelfile[:-3]}*.nc')[0]
    CLOUDNET_VOODOO_CLASS_FILE = glob.glob(f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-classification-limrad94-{modelfile[:-3]}*.nc')[0]

    class_xr_cloudnet = xr.open_dataset(CLOUDNET_CLASS_FILE, decode_times=False)
    class_xr_voodoo = xr.open_dataset(CLOUDNET_VOODOO_CLASS_FILE, decode_times=False)
    cat_xr_plus = xr.open_dataset(CLOUDNET_VOODOO_CAT_FILE, decode_times=False)
    ceilo = xr.open_mfdataset(CEILOMETE_PATH)

    cat_xr_plus['cloudnet_target_classification'] = class_xr_cloudnet['target_classification']
    cat_xr_plus['voodoo_target_classification'] = class_xr_voodoo['target_classification']
    cat_xr_plus['detection_status'] = class_xr_voodoo['detection_status']
    _ts, _rg = cat_xr_plus['time'].values, cat_xr_plus['height'].values

    # set rain/precip flag
    is_raining = cat_xr_plus['rain_rate'].values > 0
    is_precipitating = np.mean(cat_xr_plus['Z'][:, :5].values, axis=1) > 5
    is_no_lwp_ret = is_raining + is_precipitating
    

    def interpolate_T_p(cn_data):
        def check_boundaries(val_in):
            val =val_in.copy()
            if np.isnan(val[-1, :]).all():
                val[-1, :] = val[-2, :]
            if np.isnan(val[:-1, -1]).all():
                val[:, -1] = val[:, -2]
            return val

        f = scipy.interpolate.interp2d(
            cn_data['model_time'].values,
            cn_data['model_height'].values,
            check_boundaries(cn_data['temperature'].values).T,
            kind='linear',
            bounds_error=False,
            fill_value=None,
        )
        cn_data['Temp'] = cn_data['Z'].copy()
        cn_data['Temp'].values = f(_ts, _rg)[:, :].T
        cn_data['Temp'].attrs = cn_data['temperature'].attrs.copy()

        f = scipy.interpolate.interp2d(
            cn_data['model_time'].values,
            cn_data['model_height'].values,
            check_boundaries(cn_data['pressure'].values).T,
            kind='linear',
            bounds_error=False,
            fill_value=None
        )

        cn_data['Press'] = cn_data['Z'].copy()
        cn_data['Press'].values = f(_ts, _rg)[:, :].T
        cn_data['Press'].attrs = cn_data['pressure'].attrs.copy()

        return cn_data

    cat_xr_plus = interpolate_T_p(cat_xr_plus)

    def datetime64_to_decimalhour(ts_in):
        ts = []
        for i in range(len(ts_in)):
            ts.append(ts_in[i].astype('datetime64[h]').astype(int) % 24 * 3600 +
                      ts_in[i].astype('datetime64[m]').astype(int) % 60 * 60 +
                      ts_in[i].astype('datetime64[s]').astype(int) % 60)
        return np.array(ts) / 3600

    def interpolate_cbh(cn_data, ceilo_data, icb=0):
        ts_cloudnet = cn_data['time'].values
        ts_ceilo = datetime64_to_decimalhour(ceilo_data['time'].values)

        f = scipy.interpolate.interp1d(
            ts_ceilo,
            ceilo_data['cbh'][:, icb].values,
            kind='linear',
            copy=True,
            bounds_error=False,
            fill_value=None
        )
        cbh = f(ts_cloudnet)[:]
        cbh[np.isnan(cbh)] = -1
        return cbh  # ceilo ts needs to be adapted for interpoaltion

    cbh0 = interpolate_cbh(cat_xr_plus, ceilo, icb=0) + h0
    cbh1 = interpolate_cbh(cat_xr_plus, ceilo, icb=1) + h0

    def fetch_liquid_masks(cn_data):
        _ccl = cn_data['cloudnet_target_classification'].values
        _cst = cn_data['detection_status'].values
        _vcl = cn_data['voodoo_target_classification'].values
        mliqVoodoo = cn_data['liquid_probability'].values > liquid_threshold[0]
        mliqCLoudnet = (_ccl == 1) + (_ccl == 3) + (_ccl == 5) + (_ccl == 7)
        _is_lidar_only = _cst == 4
        _is_clutter = _ccl > 7
        _is_rain = np.array([(cn_data['rain_rate'].values > 0.0)] * cn_data['height'].size, dtype=bool).T
        _is_falling = (_ccl == 2) * (cn_data['v'].values < -3)

        # reclassify all fast-falling hydrometeors and insects/clutter to non-CD
        mliqVoodoo[_is_falling] = False
        mliqVoodoo[_is_clutter] = False

        cloud_mask = mliqCLoudnet + (_ccl == 2) + (_ccl == 4) + (_ccl == 6)
        cloud_mask = UT.remove_cloud_edges(cloud_mask, n=3)
        cloud_mask[_is_lidar_only] = False
        cloud_mask[_is_clutter] = False

        # # create dictionary with liquid pixel masks
        masks = {
            'Voodoo': mliqVoodoo * cloud_mask,
            'Cloudnet': mliqCLoudnet * cloud_mask,
        }

        return masks, cloud_mask

    liquid_masks, cloud_mask = fetch_liquid_masks(cat_xr_plus)

    def compute_llt_lwp(cn_data, liquid_masks, n_smoothing=n_lwp_smoothing, idk_factor=1.5):
        from libVoodoo.Utils import adiabatic_liquid_water_content
        import pandas as pd
        ts, rg = cn_data['time'].values, cn_data['height'].values
        rg_res = np.mean(np.diff(rg)) * 0.001

        # liquid water content
        lwc_dict = {}
        for key, mask in liquid_masks.items():
            _lwc = adiabatic_liquid_water_content(
                cat_xr_plus['Temp'].values,
                cat_xr_plus['Press'].values,
                mask,
                delta_h=float(np.mean(np.diff(rg)))
            )
            _lwc[_lwc > 100] = np.nan
            _lwc[_lwc < 1] = np.nan
            lwc_dict[key] = np.ma.masked_invalid(_lwc)

        _lwp = cn_data['lwp'].values
        _lwp[_lwp > 4000] = np.nan
        _lwp[_lwp < 1] = np.nan
        a = pd.Series(_lwp)

        # liquid water path and liquid layer thickness
        llt_dict, lwp_dict = {}, {}
        lwp_dict['mwr'] = a.interpolate(method='nearest').values
        lwp_dict['mwr_s'] = h.smooth(lwp_dict['mwr'], n_smoothing)
        for key in liquid_masks.keys():
            lwp_dict[key] = np.ma.sum(lwc_dict[key], axis=1) * idk_factor
            lwp_dict[key + '_s'] = h.smooth(lwp_dict[key], n_smoothing)
            llt_dict[key] = np.count_nonzero(liquid_masks[key], axis=1) * rg_res
            llt_dict[key + '_s'] = h.smooth(llt_dict[key], n_smoothing)

        return llt_dict, lwp_dict

    llt_dict, lwp_dict = compute_llt_lwp(cat_xr_plus, liquid_masks)
    valid_lwp = (0.0 < lwp_dict['mwr_s']) * (lwp_dict['mwr_s'] < 1000.0)

    def correlation_lwp_llt(llt_dict, lwp_dict):
        correlations = {}
        valid = np.argwhere(valid_lwp * (~is_no_lwp_ret))[:, 0]

        for alg in liquid_masks.keys():
            correlations.update({
                alg + 'corr(LWP)-s': np.corrcoef(lwp_dict['mwr_s'][valid], lwp_dict[alg + '_s'][valid])[0, 1],
                alg + 'corr(LLT)-s': np.corrcoef(lwp_dict['mwr_s'][valid], llt_dict[alg + '_s'][valid])[0, 1],
            })

        print(correlations)
        return correlations

    correlations = correlation_lwp_llt(llt_dict, lwp_dict)

    def fetch_voodoo_status(cn_data, liquid_masks):
        mliqVoodoo = liquid_masks['Voodoo']
        mliqCLoudnet = liquid_masks['Cloudnet']
        # H
        _TP_mask = mliqCLoudnet * mliqVoodoo
        _FP_mask = ~mliqCLoudnet * mliqVoodoo
        _FN_mask = mliqCLoudnet * ~mliqVoodoo
        _TN_mask = ~mliqCLoudnet * ~mliqVoodoo

        for its, icb in enumerate(cbh0):
            if icb > 0:
                idx_cb = h.argnearest(cn_data['height'].values, icb)
                _TN_mask[its, idx_cb+1:] = False
                _FP_mask[its, idx_cb+1:] = False

        combi_liq_mask = np.zeros(mliqVoodoo.shape)
        _non_CD_ext_mask = cloud_mask * ~_TP_mask * ~_TN_mask * ~_FP_mask * ~_FN_mask
        _CD_ext_mask = mliqVoodoo * ~_TP_mask * ~_FP_mask

        combi_liq_mask[_non_CD_ext_mask] = 1
        combi_liq_mask[_CD_ext_mask] = 2
        combi_liq_mask[_TN_mask] = 3
        combi_liq_mask[_TP_mask] = 4
        combi_liq_mask[_FP_mask] = 5
        combi_liq_mask[_FN_mask] = 6
        combi_liq_mask[~cloud_mask] = 0

        return combi_liq_mask

    voodoo_status = fetch_voodoo_status(cat_xr_plus, liquid_masks)

    def fetch_evaluation_metrics(voodoo_status):
        TN, TP, FP, FN = [np.count_nonzero(voodoo_status == i) for i in range(3, 7)]
        b = 0.5
        s = {
            'precision': TP / max(TP + FP, 1.0e-7),
            'npv': TN / max(TN + FN, 1.0e-7),
            'recall': TP / max(TP + FN, 1.0e-7),
            'specificity': TN / max(TN + FP, 1.0e-7),
            'accuracy': (TP + TN) / max(TP + TN + FP + FN, 1.0e-7),
            'F1-score': TP / max(TP + 0.5 * (FP + FN), 1.0e-7),
        }
        s['F1/2-score'] = (1 + b*b) * s['precision']*s['recall'] / max(s['recall'] + b*b*s['precision'], 1.0e-7)
        s['acc-balanced'] = (s['precision'] + s['npv']) / 2

        print(TP, FN, FP, TN)
        for key, val in s.items():
            print(f'{val:.3f}', key)
        return s

    eval_metrics = fetch_evaluation_metrics(voodoo_status)

    def ql_prediction(cn_data, lwp_max=500):
        ccolors = VIS_colors.custom_colormaps
        ts, rg = cat_xr_plus['time'].values, cat_xr_plus['height'].values*0.001

        def make_rainflag(ax, ypos=-0.2):
            raining = np.full(is_no_lwp_ret.size, ypos)
            _m0 = np.argwhere(~is_no_lwp_ret*valid_lwp)
            _m1 = np.argwhere(is_no_lwp_ret+~valid_lwp)
            ax.scatter(ts[_m0], raining[_m0], marker='|', color='green', alpha=0.75)
            ax.scatter(ts[_m1], raining[_m1], marker='|', color='red', alpha=0.75)

        def make_cbar(ax, p, ticks=None, ticklabels=None):
            ax.set(xlim=[0, 24], ylim=[-0.2, 12], xlabel='Time [UTC]', ylabel='Height [km]')
            cbar = fig.colorbar(
                p,
                cax=inset_axes(ax, width="50%", height="5%", loc='upper left'),
                fraction=0.05,
                pad=0.05,
                orientation="horizontal",
                extend='min'
            )

            if ticks is not None:
                if type(ticks) == int:
                    cbar.set_ticks(np.arange(0, ticks)+0.5)
                    cbar.ax.set_xticklabels(np.arange(0, ticks), fontsize=7)
                else:
                    cbar.set_ticks(ticks)
            if ticklabels is not None:
                cbar.ax.set_xticklabels(ticklabels, fontsize=7)
            return cbar

        def add_lwp(ax, lw=1.1, al=0.75):
            ax_right = ax.twinx()

            ax.plot(ts, lwp_dict['mwr'], color='royalblue', linewidth=lw / 2, alpha=al / 2)
            ax.bar(ts, lwp_dict['mwr_s'], linestyle='-', width=0.01, color='royalblue', alpha=0.4, label=r'LWP$_\mathrm{MWR}$')

            ax.plot(
                ts, lwp_dict['Voodoo_s'], linestyle='-', c='red', linewidth=lw, alpha=al,
                label=rf'LWP$_\mathrm{{VOODOO}}$, $r^2 = {correlations["Voodoocorr(LWP)-s"]:.2f}$'
            )
            ax.plot(
                ts, lwp_dict['Cloudnet_s'], linestyle='-', c='black', linewidth=lw, alpha=al,
                label=rf'LWP$_\mathrm{{Cloudnet}}$, $r^2 = {correlations["Cloudnetcorr(LWP)-s"]:.2f}$'
            )

            ax_right.plot(
                ts, llt_dict['Voodoo_s'], linestyle='--', c='red', linewidth=lw/2, alpha=al,
                label=rf'LLT$_\mathrm{{VOODOO}}$, $r^2 = {correlations["Voodoocorr(LLT)-s"]:.2f}$'
            )
            ax_right.plot(
                ts, llt_dict['Cloudnet_s'], linestyle='--', c='black', linewidth=lw/2, alpha=al,
                label=rf'LLT$_\mathrm{{Cloudnet}}$, $r^2 = {correlations["Cloudnetcorr(LLT)-s"]:.2f}$'
            )

            ax.set(ylabel=r'LWP [g$\,$m$^{-2}$]', xlim=[0, 24])
            ax_right.set(ylabel=r'LLT [km]', xlim=[0, 24])
            #ax.set_xlabel(r'Time [UTC]')

            leg0 = ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.99), prop={'size': 8})
            leg1 = ax_right.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), prop={'size': 8})

            for leg in [leg0, leg1]:
                leg.get_frame().set_alpha(None)
                leg.get_frame().set_facecolor((1, 1, 1, 0.9))

            ax.set_ylim([-25, lwp_max])
            ax_right.set_ylim([-0.1, 2.])
            return ax

        def fetch_cmaps():
            from matplotlib import cm
            from matplotlib.colors import ListedColormap

            viridis = cm.get_cmap('viridis', 256)
            newcolors = viridis(np.linspace(0, 1, 256))
            newcolors[:1, :] = np.array([220 / 256, 220 / 256, 220 / 256, 1])
            viridis_new = ListedColormap(newcolors)

            colors = np.array([
                [255, 255, 255, 255],
                [0, 0, 0, 15],
                [70, 74, 185, 255],
                [0, 0, 0, 35],
                [108, 255, 236, 255],
                [220, 20, 60, 255],  # [180, 55, 87, 255],
                [255, 165, 0, 155],
            ]) / 255
            V_status = ListedColormap(tuple(colors), "colors5")

            return viridis_new, V_status

        voodoo_probability_cmap, voodoo_status_cmap = fetch_cmaps()
        p = liquid_threshold[0]

        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 8))
        # 1. row
        p0 = ax[0, 0].pcolormesh(ts, rg, cn_data['Z'].values.T, cmap='jet', vmin=-50, vmax=20)
        p1 = ax[0, 1].pcolormesh(ts, rg, cn_data['v'].values.T, cmap='jet', vmin=-5, vmax=2)
        p2 = ax[0, 2].pcolormesh(ts, rg, cn_data['beta'].values.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1.0e-7, vmax=1.0e-4))

        # 2. row
        tmp = cn_data['liquid_probability'].values
        tmp2 = cn_data['noliquid_probability'].values
        cond1 = tmp > liquid_threshold[0]
        cond2 = (1 - (tmp + tmp2)) > 0.15
        tmp[cond1*cond2] = 0.0
        p3 = ax[1, 0].pcolormesh(ts, rg, tmp.T, cmap=voodoo_probability_cmap, vmin=p, vmax=1)
        ax[1, 0].scatter(ts, cbh0 * 0.001, marker='*', color='black', alpha=0.75, s=0.01)
        p4 = ax[1, 1].pcolormesh(ts, rg, cn_data['v_sigma'].values.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1.0e-2, vmax=1.0))
        ax[1, 2] = add_lwp(ax[1, 2])
        make_rainflag(ax[1, 2], ypos=-25)

        # 3. row
        p5 = ax[2, 0].pcolormesh(ts, rg, cn_data['cloudnet_target_classification'].values.T, cmap=ccolors['cloudnetpy_ds'], vmin=0, vmax=12)
        p6 = ax[2, 1].pcolormesh(ts, rg, cn_data['voodoo_target_classification'].values.T, cmap=ccolors['cloudnetpy_ds'], vmin=0, vmax=12)
        p7 = ax[2, 2].pcolormesh(ts, rg, voodoo_status.T, cmap=voodoo_status_cmap, vmin=0, vmax=7)
        make_cbar(ax[2, 2], p7, ticks=np.arange(0.5, 7.5), ticklabels=["clear\nsky", "no-CD", "CD", "TN", "TP", "FP", "FN"])

        ax[2, 0].text(
            0.1, -4,
            str([f'{key} = {val:.2f}' for key, val in eval_metrics.items()]),
            dict(size=14)
        )

        list_pmesh = [
            [0, 0, 0, 1, 1, 2, 2],
            [0, 1, 2, 0, 1, 0, 1],
            [p0, p1, p2, p3, p4, p5, p6],
            [None, None, None, None, None, 12, 12]
        ]
        for i, j, pmi, ticks in zip(*list_pmesh):
            make_cbar(ax[i, j], pmi, ticks=ticks)
            make_rainflag(ax[i, j])

        fig.subplots_adjust(bottom=0.1, right=0.95, top=0.975, left=0.05, hspace=0.13, wspace=0.15)
        return fig, ax

    fig, ax = ql_prediction(cat_xr_plus)
    fig_name = f'{pt_models_path}/{modelfile[:14]}/plots/{modelfile[:-3]}-{date_str}-Analyzer-QL.png'
    fig.savefig(fig_name, facecolor='white', dpi=400)
    print(f' saved  {fig_name}')


if __name__ == '__main__':
    ''' Main program for testing
    
    TODO: - remove liquid predictions below first (ceilometer) cloud base
          - remove profiles with lwp < 10 g m-2

    '''
    t0 = time.time()
    # setting device on GPU if available, else CPU

    _, agrs, kwargs = UT.read_cmd_line_args()
    # load data
    torch_settings = toml.load('VnetSettings-1.toml')['pytorch']

    trained_model = kwargs['model'] if 'model' in kwargs else 'Vnet2_0-dy0.00-fnXX-cuda2.pt'
    #'Vnet2_0-dy0.00-fnXX-cuda3.pt' # all data dupe=0 epochs=10 drop_cd=0.0
    #'Vnet2_0-dy0.00-fnXX-cuda2.pt' # larger network dupe_cd=0 epochs=10 drop_cd=0.9
    #'Vnet2_0-dy0.00-fnXX-cuda1.pt' # larger network dupe_cd=0 epochs=2
    #'Vnet2_0-dy0.00-fnXX-cuda0.pt' # larger network both dataset

    p = kwargs['p'] if 'p' in kwargs else 0.65
    date_str = str(kwargs['time']) if 'time' in kwargs else '20190801' #'20201230' #'20190801' #
    site = 'LIM' if int(date_str) > 20191001 else 'punta-arenas'
    ifn = str(kwargs['fn']) if 'fn' in kwargs else 'dbg'
    fac = float(kwargs['fac']) if 'fac' in kwargs else 1

    if 'punta' in site:
        hourly_path = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/calibrated/voodoo/hourly-cn133/'
        categorize_path = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/processed-hatpro/limrad94/categorize/'
    else:
        hourly_path = f'/media/sdig/leipzig/cloudnet/calibrated/voodoo/hourly-cn133/'
        categorize_path = f'/media/sdig/leipzig/cloudnet/processed-hatpro/limrad94/categorize/'


    VoodooPredictor(
        date_str,
        hourly_path=hourly_path,
        categorize_path=categorize_path,
        modelfile=trained_model,
        filenumber=ifn,
        liquid_threshold=[p, 1.0],
        site=site,
        **torch_settings
    )
    VoodooAnalyser(
        date_str,
        site,
        modelfile=trained_model,
        liquid_threshold=[p, 1.0],
        n_lwp_smoothing=20,     # in sec
    )