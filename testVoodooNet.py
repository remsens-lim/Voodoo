#!/home/sdig/anaconda3/bin/python
import datetime
import glob
import os
import sys
import time

import latextable
import matplotlib
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import toml
import torch
import xarray as xr
from cloudnetpy.products import generate_classification
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter

import libVoodoo.TorchModel as TM
import libVoodoo.TorchModel2 as TM2
import libVoodoo.TorchResNetModel as TMres
import libVoodoo.Utils as UT
from libVoodoo.Loader import dataset_from_zarr_new, VoodooXR, generate_multicase_trainingset
from libVoodoo.Utils import find_bases_tops

sys.path.append('/Users/willi/code/python/larda3/larda/')
import pyLARDA.helpers as h

import warnings
warnings.filterwarnings("ignore")

voodoo_path = os.getcwd()
pt_models_path = os.path.join(voodoo_path, f'torch_models/')

n_cloud_edge = 1
BATCH_SIZE  = 128
CLOUDNET = 'CLOUDNETpy94'
NCLASSES = 3
LIQ_EXT = '_noliquidextension'

line_colors = ['red', 'orange']
lw, al = 1.75, 0.85


def dh_to_ts(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh * 3600))


def VoodooPredictor(
        date_str,
        tomlfile,
        datafile,
        modelfile,
        liquid_threshold,
        **torch_settings
):
    torch_settings.update({'dev': 'cpu'})

    generate_multicase_trainingset(
        [datetime.datetime.strptime(f'{date_str} 0000', '%Y%m%d %H%M'),
         datetime.datetime.strptime(f'{date_str} 2359', '%Y%m%d %H%M')],
        60.0,  # minutes,
        60.0,
        '/home/sdig/code/Voodoo/tomls/',
    )

    print(f'Loading zarr files ...... {tomlfile}')
    ds_ND, ds_2D = dataset_from_zarr_new(
        DATA_PATH=datafile,
        TOML_PATH=tomlfile,
        CLOUDNET=CLOUDNET,
        RADAR='limrad94',
        TASK='predict',
    )
    X = ds_ND['features'][:, :, :, 0].values  # good!
    X = X[:, :, :, np.newaxis]

    classes = ds_2D['CLASS'].values
    status = ds_2D['detection_status'].values
    mask = ds_2D['masked'].values
    ts = ds_2D['ts'].values
    rg = ds_2D['rg'].values

    X = X.transpose(0, 3, 2, 1)
    X_test = torch.Tensor(X)

    #fig, ax_CNclass = tr.plot_timeheight2(ds_2D['CLASS'], rg_converter=True, labelsize=6)
    #fig.savefig('test2.png')

    # combine multiple ML models
    resnet = modelfile[-6:-3]
    if resnet == '-RN':
        model = TMres.ResNet18(img_channels=X_test.shape[1], num_classes=NCLASSES)
    elif resnet == 'VRN':
        model = TM2.VoodooNet(X_test.shape, NCLASSES, **torch_settings)
    else:
        model = TM.VoodooNet(X_test.shape, NCLASSES, **torch_settings)

    model.load_state_dict(
        torch.load(f'{pt_models_path}/{modelfile[:14]}/{modelfile}', map_location=model.device)['state_dict']
    )
    #print(modelfile)
    #model.print_nparams()

    prediction = model.predict(X_test, batch_size=BATCH_SIZE)
    prediction = prediction.to('cpu')

    predicted_values = TM.VoodooNet.new_classification(prediction, mask)



    Vpost = VoodooXR(ts, rg)
    Vpost.add_nD_variable('VCLASS', ('ts', 'rg'), predicted_values, **{
        'colormap': 'cloudnet_target_new',
        'rg_unit': 'km',
        'var_unit': '',
        'system': 'VOODOO',
        'var_lims': [0, 10],
        'range_interval': [0, 5]
    })

    Vpost['CNCLASS'] = Vpost['VCLASS'].copy()
    Vpost['CNCLASS'].attrs['system'] = 'CLOUDNETpy94'
    Vpost['CNCLASS'].values = classes

    Nts, Nrg = mask.shape
    probabilities = TM.VoodooNet.reshape(prediction, mask, (Nts, Nrg, NCLASSES))

    smoothed_probs = np.zeros((Nts, Nrg, NCLASSES))
    for i in range(NCLASSES):
        smoothed_probs[:, :, i] = gaussian_filter(probabilities[:, :, i], sigma=1)

    values = np.zeros((Nts, Nrg))
    voodoo_liquid_mask = (smoothed_probs[:, :, 1] > liquid_threshold[0]) * (smoothed_probs[:, :, 1] < liquid_threshold[1])
    values[voodoo_liquid_mask] = 1.0

    # again for smoothed predictions
    print('\nMetrics for smoothed output:')
    _ = TM.evaluation_metrics(values, classes, status)

    # load categorization and write new bitmask
    if int(str(date_str)[:4]) > 2019:
        CATEGORIZE_PATH = f'/media/sdig/leipzig/cloudnet/processed/limrad94/categorize-py{LIQ_EXT}/'
        site = 'LIM'
    else:
        CATEGORIZE_PATH = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/processed/limrad94/categorize-py{LIQ_EXT}/'
        site = 'punta-arenas'

    # load the original categorize file
    CAT_FILES_PATH = f'/{CATEGORIZE_PATH}/{str(date_str)[:4]}/'
    CLOUDNET_CAT_FILE = f'{CAT_FILES_PATH}/{date_str}-{site}-categorize-limrad94.nc'
    cat_xr = xr.open_dataset(CLOUDNET_CAT_FILE, decode_times=False)


    def _fill_time_gaps():
        # print('Cloudnetpy has dimensins (ts,rg) :: ', cat_xr['category_bits'].shape)
        # print('   Voodoo has dimensions (ts,rg) :: ', probabilities.shape[:2])
        n_ts_cloudnet_cat, n_rg_cloudnet_cat = cat_xr['category_bits'].shape
        ts_unix_cloudnetpy = np.array([UT.dt_to_ts(dh_to_ts(date_str, dh)) for dh in cat_xr['time'].values])

        _tmp_master_ts = ts_unix_cloudnetpy.astype(int)
        _tmp_slave_ts = ts.astype(int)
        uniq, uniq_idx = np.unique(_tmp_slave_ts, return_index=True, )
        _tmp_slave_ts = _tmp_slave_ts[uniq_idx]
        values_final = values[uniq_idx, :]
        status_final = status[uniq_idx, :]
        smoothed_probs_final = smoothed_probs[uniq_idx, :]

        ts_unix_cloundetpy_mask = np.full(n_ts_cloudnet_cat, False)
        v_new = np.zeros((n_ts_cloudnet_cat, n_rg_cloudnet_cat))
        s_new = np.zeros((n_ts_cloudnet_cat, n_rg_cloudnet_cat))
        p_new = np.zeros((n_ts_cloudnet_cat, n_rg_cloudnet_cat, 3))
        cnt = 0
        for i, mts in enumerate(_tmp_master_ts):
            if cnt == _tmp_slave_ts.size:
                break
            if mts == _tmp_slave_ts[cnt]:
                ts_unix_cloundetpy_mask[i] = True
                v_new[i, :]  = values_final[cnt, :]
                s_new[i, :]  = status_final[cnt, :]
                p_new[i, :]  = smoothed_probs_final[cnt, :]
                cnt += 1
        return v_new, s_new, p_new

    values_new, status_new, smoothed_probs_new = _fill_time_gaps()

    def _adjust_cloudnetpy_bits():
        n_ts_cloudnet_cat, n_rg_cloudnet_cat = cat_xr['category_bits'].shape
        bits_unit = cat_xr['category_bits'].values.astype(np.uint8)
        new_bits = bits_unit.copy()

        for ind_time in range(n_ts_cloudnet_cat):
            for ind_range in range(n_rg_cloudnet_cat):
                if values_new[ind_time, ind_range] == 1:
                    if status_new[ind_time, ind_range] in [1, 2]:
                        continue  # skip good radar & lidar echo pixel
                    if cat_xr['v'][ind_time, ind_range] < -3:
                        continue
                    bit_rep = np.unpackbits(bits_unit[ind_time, ind_range])
                    bit_rep[-1] = 1  # set droplet bit
                    new_bits[ind_time, ind_range] = np.packbits(bit_rep)
        return new_bits

    cat_xr['category_bits'].values = _adjust_cloudnetpy_bits()

    os.makedirs(pt_models_path + f'{modelfile[:14]}/nc/', exist_ok=True)
    CAT_FILE_NAME = f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-categorize-limrad94-{modelfile[:-3]}.nc'
    cat_xr.attrs['postprocessor'] = f'Voodoo_v1.0, Modelname: {modelfile[:-3]}'

    for i in range(3):
        cat_xr[f'pred_liquid_prob_{i}'] = xr.DataArray(
            smoothed_probs_new[:, :, i],
            dims=('time', 'height'),
            attrs={'comment': "This variable contains information about the likelihood of cloud droplet\n"
                              f"availability, predicted by the {cat_xr.attrs['postprocessor']}\n"
                              "classifier.",
                   'definition': "\nProbability 1 means most likely cloud droplets are available,\n"
                                 "probability of 0 means no cloud droplets are available, respectively.\n",
                   'units': "",
                   'long_name': f"Predicted likelihood of cloud droplet availability _{i}"}
        )
    cat_xr.to_netcdf(path=CAT_FILE_NAME, format='NETCDF4', mode='w')
    print(f"\nfig saved: {CAT_FILE_NAME}")
    # generate classification with new bit mask
    generate_classification(CAT_FILE_NAME, CAT_FILE_NAME.replace('categorize', 'classification'))

    # ======================================================================================================================================================

def VoodooAnalyser(
        date_str,
        site,
        modelfile,
        liquid_threshold,
        time_lwp_smoothing=600,
        entire_day='yes',
):

    assert 'Vnet' in modelfile, 'no Vnet model given'

    ifn = modelfile[-1]
    idk_factor = 1.0

    hour_start, hour_end = 0, 23
    range_start, range_end = 0, 12000
    if entire_day == 'no':
        if date_str == '20190801':
            hour_start, hour_end = 0, 9
            range_start, range_end = None, 6000
        elif date_str == '20210310':
            hour_start, hour_end = 0, 11
            range_start, range_end = None, 4000
        elif date_str == '20201230':
            hour_start, hour_end = 12, 20
            range_start, range_end = None, 4000
        elif date_str == '20210216':
            hour_start, hour_end = 0, 15
            range_start, range_end = None, 8000
        elif date_str == '20210217':
            hour_start, hour_end = 1, 5
            range_start, range_end = None, 8000
        elif date_str == '20190313':
            hour_start, hour_end = 3, 21
            range_start, range_end = None, 6000
        else:
            hour_start, hour_end = 0, None
            range_start, range_end = 0, None

    if int(str(date_str)[:4]) > 2019:
        CLASSIFICATION_PATH = f'/media/sdig/leipzig/cloudnet/products/limrad94/classification-cloudnetpy{LIQ_EXT}/{str(date_str)[:4]}'
        ds_ceilo_all = xr.open_mfdataset(f'/media/sdig/leipzig/instruments/ceilim/data/Y{date_str[:4]}/M{date_str[4:6]}/{date_str}_Leipzig_CHM200114_000.nc')
    else:
        CLASSIFICATION_PATH = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/products/limrad94/classification-cloudnetpy{LIQ_EXT}/{str(date_str)[:4]}'
        ds_ceilo_all = xr.open_mfdataset(f'/media/sdig/LACROS/cloudnet/data/punta-arenas/calibrated/chm15x/{date_str[:4]}/{date_str}_punta-arenas_chm15x.nc')

    h0 = 117 if site == 'LIM' else 7
    """
        ____ ___  ____ _  _    ____ ____ ___ ____ ____ ____ ____ _ ____ ____    ____ _ _    ____ 
        |  | |__] |___ |\ |    |    |__|  |  |___ | __ |  | |__/ | [__  |___    |___ | |    |___ 
        |__| |    |___ | \|    |___ |  |  |  |___ |__] |__| |  \ | ___] |___    |    | |___ |___ 

    """
    # original & new categorize file

    CLOUDNET_CLASS_FILE = f'{CLASSIFICATION_PATH}/{date_str}-{site}-classification-limrad94.nc'
    CLOUDNET_VOODOO_CAT_FILE = glob.glob(f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-categorize-limrad94-{modelfile[:-3]}*.nc')[0]
    CLOUDNET_VOODOO_CLASS_FILE = glob.glob(f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-classification-limrad94-{modelfile[:-3]}*.nc')[0]

    # interpolate model variable
    _cat_mod = xr.open_dataset(CLOUDNET_VOODOO_CAT_FILE, decode_times=False)
    _ts, _rg = _cat_mod['time'].values, _cat_mod['height'].values


    if hour_start is None:
        hour_start = 0
    dt_list_all = [
        datetime.datetime.strptime(date_str + str(hour_start).zfill(2), '%Y%m%d%H') +
        datetime.timedelta(seconds=int(tstep * 60 * 60)) for tstep in _ts
    ]
    ts_list_all = [(dt - datetime.datetime(1970, 1, 1)).total_seconds() for dt in dt_list_all]

    f = scipy.interpolate.interp2d(
        _cat_mod['model_time'].values,
        _cat_mod['model_height'].values,
        _cat_mod['temperature'].values.T,
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )
    _cat_mod['Temp'] = _cat_mod['Z'].copy()
    _cat_mod['Temp'].values = f(_ts, _rg)[:, :].T
    _cat_mod['Temp'].attrs = _cat_mod['temperature'].attrs.copy()

    f = scipy.interpolate.interp2d(
        _cat_mod['model_time'].values,
        _cat_mod['model_height'].values,
        _cat_mod['pressure'].values.T,
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )

    _cat_mod['Press'] = _cat_mod['Z'].copy()
    _cat_mod['Press'].values = f(_ts, _rg)[:, :].T
    _cat_mod['Press'].attrs = _cat_mod['pressure'].attrs.copy()

    i0 = (dt_list_all[0].replace(hour=0, minute=0, second=0) - datetime.datetime(1970, 1, 1)).total_seconds()
    ts, rg = np.array(ts_list_all), _cat_mod['height'].values
    _ts_from_ceilo = np.array([
        (dt64 - ds_ceilo_all['time'][0].values) / np.timedelta64(1, 's') for dt64 in ds_ceilo_all['time'].values
    ]) / 3600

    f = scipy.interpolate.interp1d(
        _ts_from_ceilo,
        ds_ceilo_all['cbh'][:, 0].values,
        kind='linear',
        copy=True,
        bounds_error=False,
        fill_value=None
    )
    _cbh1_all = f((ts - i0) / 3600)[:]  # ceilo ts needs to be adapted for interpoaltion

    fig_name = f'{pt_models_path}/{modelfile[:14]}/plots/{modelfile[:-3]}-{date_str}-Analyzer-QL.png'


    # SLICING
    if entire_day == 'no':
        time_range_slicer = {'time': slice(hour_start, hour_end), 'height': slice(range_start, range_end)}
        cat_post_xr = _cat_mod.sel(**time_range_slicer)
        class_post_xr = xr.open_dataset(CLOUDNET_VOODOO_CLASS_FILE, decode_times=False).sel(**time_range_slicer)
        class_orig_xr = xr.open_dataset(CLOUDNET_CLASS_FILE, decode_times=False).sel(**time_range_slicer)
    else:
        cat_post_xr = _cat_mod.copy()
        class_post_xr = xr.open_dataset(CLOUDNET_VOODOO_CLASS_FILE, decode_times=False)
        class_orig_xr = xr.open_dataset(CLOUDNET_CLASS_FILE, decode_times=False)


    if hour_start is None:
        hour_start = 0
    dt_list = [
        datetime.datetime.strptime(date_str+str(hour_start).zfill(2), '%Y%m%d%H') +
        datetime.timedelta(seconds=int(tstep * 60 * 60)) for tstep in cat_post_xr['time'].values
    ]
    ts_list = [(dt - datetime.datetime(1970, 1, 1)).total_seconds() for dt in dt_list]


    # slice cbh
    if entire_day ==  'no':
        begin_dt = datetime.datetime.strptime(f'{dt_list_all[0]:%Y%m%d}{str(hour_start).zfill(2)}01', '%Y%m%d%H%M')
        end_dt = datetime.datetime.strptime(f'{dt_list_all[0]:%Y%m%d}{str(hour_end).zfill(2)}59', '%Y%m%d%H%M')
    else:
        begin_dt = dt_list_all[0]
        end_dt = dt_list_all[-1]
    idx_start = h.argnearest(dt_list_all, begin_dt)
    idx_stop = h.argnearest(dt_list_all, end_dt)
    if len(dt_list_all) == idx_stop - idx_start + 1:
        idx_ts = slice(h.argnearest(dt_list_all, begin_dt), idx_stop + 1)
    else:
        idx_ts = slice(h.argnearest(dt_list_all, begin_dt), idx_stop)

    _cbh1 = _cbh1_all[idx_ts] + h0

    ts_arr, rg_arr = cat_post_xr['pred_liquid_prob_1']['time'].values, cat_post_xr['pred_liquid_prob_1']['height'].values
    all_targets_mask = cat_post_xr['Z'].values > 1.0e10
    voodoo_liq_mask = (cat_post_xr['pred_liquid_prob_1'].values  > liquid_threshold[0]) * (cat_post_xr['pred_liquid_prob_1'].values  < liquid_threshold[1])

    cloudnet_classification = class_orig_xr['target_classification'].values.copy()
    cloudnet_status = class_orig_xr['detection_status'].values.copy()

    n_smoothing = int(time_lwp_smoothing/60)

    """
        _    ____ ___ ____ _  _    ___ ____ ___  _    ____    _  _ ____ ___ ____ _ ____ ____ 
        |    |__|  |  |___  \/      |  |__| |__] |    |___    |\/| |___  |  |__/ | |    [__  
        |___ |  |  |  |___ _/\_     |  |  | |__] |___ |___    |  | |___  |  |  \ | |___ ___] 


    """

    print('\n\nScores of predictive performancea\n\n')
    metrics = TM.evaluation_metrics(voodoo_liq_mask, cloudnet_classification, cloudnet_status)
    del metrics['npv'], metrics['specificity'], metrics['Jaccard-index']

    table1 = latextable.Texttable()
    table1.set_deco(latextable.Texttable.HEADER)
    table1.set_cols_align(["r", 'c'])
    table1.add_row(["metric", f'fn{ifn}'])
    table1.add_rows([[key, val] for key, val in metrics.items()])
    print("\n" + table1.draw() + "\n")
    #print(latextable.draw_latex(table1,
    #                            caption=f"Performance metrics from model: {fig_name}",
    #                            label=f"tab:metrics-fn{igpu}-gpu{igpu}") + "\n")

    """
        _  _ ____ _  _ ____    _  _ ____ ____ ____ ____ _   _ ____ 
        |\/| |__| |_/  |___     \/  |__| |__/ |__/ |__|  \_/  [__  
        |  | |  | | \_ |___    _/\_ |  | |  \ |  \ |  |   |   ___] 

    """
    # VOODOO cloud droplet likelyhood colorbar (viridis + grey below minimum value)
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    viridis = cm.get_cmap('viridis', 6)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[:1, :] = np.array([220 / 256, 220 / 256, 220 / 256, 1])


    Vpost = VoodooXR(ts_list, rg_arr)
    Vpost.add_nD_variable('CCLASS', ('ts', 'rg'), class_orig_xr['target_classification'].values.copy(), **{
            'colormap': 'cloudnet_target_new',
            'rg_unit': 'km',
            'var_unit': '',
            'system': 'CloudnetPy',
            'var_lims': [0, 10],
            'range_interval': [0, 12]
        })
    Vpost.add_nD_variable('VCLASS', ('ts', 'rg'), class_post_xr['target_classification'].values.copy(), **{
            'colormap': 'cloudnet_target_new',
            'rg_unit': 'km',
            'var_unit': '',
            'system': 'CloudnetPy',
            'var_lims': [0, 10],
            'range_interval': [0, 12]
        })
    Vpost.add_nD_variable('CSTATUS', ('ts', 'rg'), class_orig_xr['detection_status'].values.copy(), **{
            'colormap': 'cloudnetpy_detection_status',
            'rg_unit': 'km',
            'var_unit': '',
            'system': 'CloudnetPy',
            'var_lims': [0, 7],
            'range_interval': [0, 12]
        })
    Vpost.add_nD_variable('liquid_probability', ('ts', 'rg'), cat_post_xr['pred_liquid_prob_1'].values.copy(), **{
            'colormap': ListedColormap(newcolors),
            'rg_unit': 'km',
            'var_unit': '',
            'system': 'CloudnetPy',
            'var_lims': liquid_threshold,
            'range_interval': [0, 12]
        })
    Vpost['liquid_probability'].values = np.ma.masked_where(all_targets_mask, Vpost['liquid_probability'].values)


    ### NEW
    rg_list = Vpost['rg'].values

    cloudnet_cbh = np.ma.masked_greater_equal(class_post_xr['cloud_base_height_agl'].values, 15000)
    _ccl = class_orig_xr['target_classification'].values
    _cst = class_orig_xr['detection_status'].values
    _mask_voodoo = (Vpost['liquid_probability'].values > liquid_threshold[0])  # * (ds_voodoo_cat['pred_liquid_prob_1'].values < 0.90)
    _mask_cloudnet = (_ccl == 1) + (_ccl == 3) + (_ccl == 5) + (_ccl == 7)
    _is_lidar_only = _cst == 4
    _is_radar_lidar = _cst == 1
    _is_clutter = _ccl > 7
    _is_rain = np.array([cat_post_xr['is_rain'].values] * Vpost['rg'].size, dtype=bool).T
    _is_falling = (_ccl == 2) * (cat_post_xr['v'].values < -3)

    # reclassify falling to non-CD
    _mask_voodoo[_is_falling] = False
    _mask_voodoo[_is_clutter] = False

    _cloud_mask = _mask_cloudnet + (_ccl == 2) + (_ccl == 4) + (_ccl == 6)
    _cloud_mask[_is_lidar_only] = False
    _cloud_mask = UT.remove_cloud_edges(_cloud_mask, n=3)

    _TP_mask = _cloud_mask * (_mask_cloudnet * _mask_voodoo)
    _FP_mask = _cloud_mask * (~_mask_cloudnet * _mask_voodoo)
    _FN_mask = _cloud_mask * (_mask_cloudnet * ~_mask_voodoo)
    _TN_mask = _cloud_mask * (~_mask_cloudnet * ~_mask_voodoo)

    _mask_below_cloudbase = np.full(_ccl.shape, False)
    for its, icb in enumerate(_cbh1):
        if icb > 0:
            idx_cb = h.argnearest(rg_list, icb)
            _mask_below_cloudbase[its, :idx_cb] = True

            _TN_mask[its, idx_cb + 1:] = False
            _FP_mask[its, idx_cb + 1:] = False

    # # create dictionary with liquid pixel masks
    liquid_masks2D = {
        'Voodoo': _mask_voodoo * _cloud_mask,
        'CloudnetPy': _mask_cloudnet * _cloud_mask,
    }

    # liquid water content
    rg_res = float(np.mean(np.diff(Vpost.rg.values)))
    lwc_dict = {}
    for key, mask in liquid_masks2D.items():
        tmp = UT.adiabatic_liquid_water_content(
            _cat_mod['Temp'].values, _cat_mod['Press'].values, mask, delta_h=rg_res
        )
        tmp = np.ma.masked_less_equal(tmp, 0.0)
        tmp = np.ma.masked_greater_equal(tmp, 50.0)
        lwc_dict[key] = tmp

    lwp_nan = cat_post_xr['lwp'].values
    lwp_nan[lwp_nan > 2000] = np.nan
    lwp_nan[lwp_nan < 0] = np.nan
    a = pd.Series(lwp_nan)

    # liquid water path
    lwp_dict = {
        # MWR-LWP from cloudnet
        'mwr': a.interpolate(method='nearest').values,
        # adiabatic LWP for
        **{key: np.ma.sum(val, axis=1) for key, val in lwc_dict.items()},
    }
    lwp_dict.update({key + '_s': h.smooth(val, n_smoothing) for key, val in lwp_dict.items()})  # smooth all variables, add to the dictionary

    # liquid layer thickness
    llt_dict = {key: np.count_nonzero(liquid_masks2D[key], axis=1) * rg_res for key in liquid_masks2D.keys()}
    llt_dict.update({key + '_s': h.smooth(val, n_smoothing) for key, val in llt_dict.items()})  # smooth all variables, add to the dictionary

    # cloud tops and cloud bases
    CBH_dict = {'CEILO1': np.ma.masked_less_equal(_cbh1, h0)}

    for key in liquid_masks2D.keys():
        idx_cb = np.argmax(liquid_masks2D[key] == 1, axis=1)
        CBH_dict[key] = np.ma.masked_less_equal([rg_list[ind_rg] for ind_rg in idx_cb], 200.0)
        CBH_dict[key] = np.ma.masked_invalid(CBH_dict[key])

    '''
    ___  ____ ____ ___    ___  ____ ____ ____ ____ ____ ____ _ _  _ ____
    |__] |  | [__   |     |__] |__/ |  | |    |___ [__  [__  | |\ | | __
    |    |__| ___]  |     |    |  \ |__| |___ |___ ___] ___] | | \| |__]

    '''
    # CORRELATION COEFFICIENTS
    bin_edges = [
        [0, 2000],
        [0, 20], [20, 30], [30, 40], [40, 50],
        [50, 65], [65, 80], [80, 100], [100, 150],
        [150, 200], [200, 300], [300, 400], [400, 2000]
    ]
    lwp_masks = [
        (edge[0] < lwp_dict['mwr']) * (lwp_dict['mwr'] < edge[1]) for edge in bin_edges
    ]

    correlation_LLT, correlation_LLT_s = {}, {}
    correlation_LWP, correlation_LWP_s = {}, {}
    correlation_LCBH = {}
    correlations = {}

    for alg in liquid_masks2D.keys():
        correlation_LLT_s[alg + 'corr(LLT)-s'] = [
            UT.correlation_coefficient(lwp_dict['mwr_s'][_msk], llt_dict[alg + '_s'][_msk]) for _msk in lwp_masks
        ]
        correlation_LWP_s[alg + 'corr(LWP)-s'] = [
            UT.correlation_coefficient(lwp_dict['mwr_s'][_msk], lwp_dict[alg + '_s'][_msk]) for _msk in lwp_masks
        ]
        _cbh_mask = CBH_dict['CEILO1'] > 150
        correlation_LCBH[alg + 'corr((L)CBH)'] = [
            UT.correlation_coefficient(CBH_dict[alg][_msk*_cbh_mask], CBH_dict['CEILO1'][_msk*_cbh_mask]) for _msk in lwp_masks
        ]

        correlations[alg] = np.array([
            correlation_LLT_s[alg + 'corr(LLT)-s'],
            correlation_LWP_s[alg + 'corr(LWP)-s'],
            correlation_LCBH[alg + 'corr((L)CBH)'],
        ])

    int_columns = ['TP', 'TN', 'FP', 'FN']
    flt_columns = ['precision', 'npv', 'recall', 'specificity', 'accuracy', 'F1-score']
    corr_columns = ['Vr2(LLT)', 'Vr2(LWP)', 'Vr2(LCBH)', 'Cr2(LLT)', 'Cr2(LWP)', 'Cr2(LCBH)']
    extra_columns = ['n_time_steps']

    num_columns = len(int_columns) + len(flt_columns) + len(corr_columns) + len(extra_columns)
    ##### ds

    title_list = ['TP', 'TN', 'FP', 'FN']
    mask_list = [_TP_mask, _TN_mask, _FP_mask, _FN_mask]
    n_range_bins = _cloud_mask.shape[1]

    arr0 = np.zeros((len(lwp_masks), num_columns), dtype=float)
    for i in range(len(lwp_masks)):
        _lwp_bin_mask = np.array([lwp_masks[i]] * n_range_bins).T
        _lwp_bin_mask = _lwp_bin_mask * _cloud_mask

        n_masks = []
        for i_mask in mask_list:
            n = np.count_nonzero(i_mask * _lwp_bin_mask)
            n_masks.append(n)

        sum_stats = UT.performance_metrics(*n_masks)
        sum_stats_list = [val for val in sum_stats.values()]
        arr0[i, :] = np.array(
            n_masks +
            sum_stats_list +
            list(correlations['Voodoo'][:, i]) +
            list(correlations['CloudnetPy'][:, i]) +
            [np.count_nonzero(np.any(_lwp_bin_mask, axis=1)), ]
        )

    # create pandas dataframe and save to csv
    print(f'\n')

    stats_list = [
        [['', ] + int_columns + flt_columns + corr_columns + extra_columns] +
        [[f'{site}-lwp-bin{i}', ] + list(val) for i, val in enumerate(arr0[:, :])]
    ]
    stats_list = stats_list[0]


    """
        _    ____ ___ ____ _  _    ___ ____ ___  _    ____ ____ 
        |    |__|  |  |___  \/      |  |__| |__] |    |___ [__                  
        |___ |  |  |  |___ _/\_     |  |  | |__] |___ |___ ___] 

    """

    def pmatrix(a):
        """Returns a LaTeX pmatrix

        :a: numpy array
        :returns: LaTeX bmatrix as a string
        """
        if len(a.shape) > 2:
            raise ValueError('pmatrix can at most display two dimensions')
        lines = str(a).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{pmatrix}']
        rv += ['  ' + ' & '.join(l.split()) for l in lines]
        rv += [r'\end{pmatrix}']
        return ' '.join(rv)

    em = np.array([[stats_list[1][1], stats_list[1][3]],
                   [stats_list[1][4], stats_list[1][2]]])


    perf = [stats_list[1][5], stats_list[1][7], stats_list[1][9], stats_list[1][10]]

    corr_v = [stats_list[1][11], stats_list[1][12], stats_list[1][13]]
    corr_c = [stats_list[1][14], stats_list[1][15], stats_list[1][16]]
    for i in range(len(corr_v)):
        corr_v[i] = f'{corr_v[i]:.3f}'
        corr_c[i] = f'{corr_c[i]:.3f}'

    print(f'case-{site}\n')
    print(pmatrix(em))
    print()
    print('prec, recall ... ', f'{perf[0]:.3f} & {perf[1]:.3f} & {perf[2]:.3f} & {perf[3]:.3f} & ', end='')
    print(f'{corr_v[0]} / {corr_c[0]} & {corr_v[1]} / {corr_c[1]} & {corr_v[2]} / {corr_c[2]} \\')
    print(f'\n       maximum calculated adiabatic liquid water path value = {np.nanmax(lwp_dict["Voodoo"])*idk_factor:.2f} [g m-2]')
    print(f'maximum actual microwave radiometer liquid water path value = {np.nanmax(lwp_dict["mwr"]):.2f} [g m-2]')

    """
        ____ _  _ ____ _    _   _ ___  ____ ____    ____ _
        |__| |\ | |__| |     \_/    /  |___ |__/    |  | |
        |  | | \| |  | |___   |    /__ |___ |  \    |_\| |___
    """

    title = False
    _FONT_SIZE = 14
    line_colors = ['red', 'black', 'orange']
    _colornames = ["clear\nsky", "non-CD\next.", "CD\next.", "TN", "TP", "FP", "FN"]
    _colors = np.array([
        [255, 255, 255, 255],
        [0, 0, 0, 45],
        [70, 74, 185, 255],
        [0, 0, 0, 15],
        [108, 255, 236, 255],
        [180, 55, 87, 255],
        [255, 165, 0, 155],
    ]) / 255
    print(len(_colornames))
    lw, al = 1.75, 0.85
    diff_LCBH_v = np.ma.masked_invalid(CBH_dict['CEILO1'] - CBH_dict['Voodoo'])*0.001
    diff_LCBH_c = np.ma.masked_invalid(CBH_dict['CEILO1'] - CBH_dict['CloudnetPy'])*0.001

    # Generating Data
    def _plot_all_together():
        dt_np = np.array(dt_list)
        with plt.style.context(['science', 'ieee']):
            fig = plt.figure(figsize=(19, 12))
            gs = fig.add_gridspec(4, 6)
            ax_Z = fig.add_subplot(gs[0, :2])
            ax_beta = fig.add_subplot(gs[0, 2:4])
            ax_violin_left = fig.add_subplot(gs[1, 0])
            ax_violin_right = fig.add_subplot(gs[1, 1])
            ax_LCBH_diff = fig.add_subplot(gs[1, 2:4])
            ax_prob = fig.add_subplot(gs[2:3, :2])
            ax_liq_overlapp = fig.add_subplot(gs[3:4, :2])
            ax_LLT = fig.add_subplot(gs[2:3, 2:4])
            ax_LLT_right = ax_LLT.twinx()
            ax_LWPad = fig.add_subplot(gs[3:4, 2:4])
            ax_v = fig.add_subplot(gs[0, 4:])
            ax_wd = fig.add_subplot(gs[1, 4:])
            ax_ldr = fig.add_subplot(gs[2, 4:])
            ax_text = fig.add_subplot(gs[3, 4:])
            # C
            pcmesh = ax_v.pcolormesh(
                dt_list, rg_list * 0.001, np.ma.masked_greater_equal(cat_post_xr['v'].values.T, 50),
                cmap='jet', vmin=-4, vmax=2
            )
            cbaxes = inset_axes(ax_v, width="50%", height="5%", loc='upper left')
            cbar = fig.colorbar(pcmesh, cax=cbaxes, pad=0.1, orientation="horizontal", extend='min')
            bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5)
            cbar.ax.set_ylabel(r'$v_D$ [m\,s$^{-1}$]', labelpad=-190, rotation=0, y=0, bbox=bbox)
            # cbar.ax.setp(ax2.get_xticklabels(), backgroundcolor="limegreen")

            # B
            pcmesh = ax_beta.pcolormesh(
                dt_list, rg_list * 0.001, np.ma.masked_greater_equal(cat_post_xr['beta'].values.T, 50),
                cmap='jet', norm=mpl.colors.LogNorm(vmin=1.0e-7, vmax=1.0e-4)
            )
            cbaxes = inset_axes(ax_beta, width="50%", height="5%", loc='upper left')
            cbar = fig.colorbar(pcmesh, cax=cbaxes, pad=0.1, orientation="horizontal", extend='min')
            cbar.ax.set_ylabel(r'$\beta$ [m$^{-1}\,$sr$^{-1}$]', labelpad=-195, rotation=0, y=0)

            # A
            pcmesh = ax_Z.pcolormesh(
                dt_list, rg_list * 0.001, np.ma.masked_greater_equal(cat_post_xr['Z'].values.T, 50),
                cmap='jet', vmin=-50, vmax=20
            )
            cbaxes = inset_axes(ax_Z, width="50%", height="5%", loc='upper left')
            cbar = fig.colorbar(pcmesh, cax=cbaxes, pad=0.1, orientation="horizontal", extend='min')
            bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5)
            cbar.ax.set_ylabel(r'$Z_e$ [dBZ]', labelpad=-190, rotation=0, y=0, bbox=bbox)

            # J
            pcmesh = ax_wd.pcolormesh(
                dt_list, rg_list * 0.001, np.ma.masked_greater_equal(cat_post_xr['width'].values.T, 50),
                cmap='jet',  norm=mpl.colors.LogNorm(vmin=1.0e-1, vmax=1.e-0)
            )
            cbaxes = inset_axes(ax_wd, width="50%", height="5%", loc='upper left')
            cbar = fig.colorbar(pcmesh, cax=cbaxes, pad=0.1, orientation="horizontal", extend='min')
            bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5)
            cbar.ax.set_ylabel(r'$\sigma_w$ [m\,s$^{-1}$]', labelpad=-190, rotation=0, y=0, bbox=bbox)

            # L
            pcmesh = ax_ldr.pcolormesh(
                dt_list, rg_list * 0.001, np.ma.masked_greater_equal(cat_post_xr['ldr'].values.T, 50),
                cmap='jet', vmin=-30, vmax=0
            )
            cbaxes = inset_axes(ax_ldr, width="50%", height="5%", loc='upper left')
            bounds = np.linspace(-30, 0, 500)
            cbar = fig.colorbar(pcmesh, cax=cbaxes, pad=0.1, orientation="horizontal", extend='min',
                                boundaries=bounds, ticks=[-30, -25, -20, -15, -10, -5, 0])
            bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.5)
            cbar.ax.set_ylabel(r'ldr [dB]', labelpad=-190, rotation=0, y=0, bbox=bbox)
            cbar.set_ticklabels([-30, -25, -20, -15, -10, -5, 0])
            ax_ldr.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(3))

            # D and E
            _cbh_mask = CBH_dict[f'CEILO1'] > h0
            for ax in [ax_violin_left, ax_violin_right]:
                # ax = sns.violinplot(x=df["LCBH"], ax=ax, scale="count", palette="Pastel1")
                z1  = [y for y in diff_LCBH_v if y]
                z2  = [y for y in diff_LCBH_c if y]
                ax.boxplot([z1, z2], notch=True)
                ax.set_xticklabels(['Voodoo', 'Cloudnet'])
                ax.grid()
            ax_violin_left.set_ylabel('(CBH - LCBH) [km]')
            ax_violin_left.set_ylim([-3, 3])
            ax_violin_right.set_ylim([-.5, .5])
            ax_LCBH_diff.set_ylim([-3, 3])
            if title:
                ax_violin_left.set_title('LCBH(Cloudnet) - LCBH(Voodoo)')
                ax_LWPad.set_title('MWR-LWP vs.  adiabatic LWP')
                ax_LLT.set_title('MWR-LWP vs. liquid layer thickness')
                ax_LCBH_diff.set_title('Difference in Liquid Cloud Base Height')
                ax_prob.set_title('Voodoo output - probabiltiy for cloud droplets')
                ax_liq_overlapp.set_title('Cloud Droplet Overlapp - Cloudnet VS Voodoo')
            # f
            ax_LCBH_diff.plot(dt_list, diff_LCBH_v, color='red', linewidth=lw / 2, alpha=al / 2,
                              label=f'Voodoo, $r^2=$ {correlation_LCBH["Voodoocorr((L)CBH)"][0]:.3f}')
            ax_LCBH_diff.plot(dt_list, diff_LCBH_c, color='black', linewidth=lw / 2, alpha=al / 2,
                              label=f'CloudnetPy,  $r^2=$ {correlation_LCBH["CloudnetPycorr((L)CBH)"][0]:.3f}')
            # and I
            ax_LLT.plot(dt_list, lwp_dict['mwr'], color='royalblue', linewidth=lw / 2, alpha=al / 2)
            ax_LLT.bar(dt_list, lwp_dict['mwr_s'], linestyle='-', width=0.001, color='royalblue', alpha=0.4, label='MWR')
            ax_LWPad.plot(dt_list, lwp_dict['mwr'], color='royalblue', linewidth=lw / 2, alpha=al / 2)
            ax_LWPad.bar(dt_list, lwp_dict['mwr_s'], linestyle='-', width=0.001, color='royalblue', alpha=0.4, label='MWR')
            for alg, color in zip(liquid_masks2D.keys(), line_colors):
                ax_LWPad.plot(dt_list, lwp_dict[alg], color=color, linewidth=lw / 2, alpha=al / 2)
                ax_LWPad.plot(dt_list, lwp_dict[alg + '_s'], linestyle='-', c=color, linewidth=lw, alpha=al,
                              label=alg + f' $r^2=$ {correlation_LWP_s[alg + "corr(LWP)-s"][0]:.3f}')
                ax_LLT_right.plot(dt_list, llt_dict[alg]*0.001, color=color, linewidth=lw / 2, alpha=al / 2)
                ax_LLT_right.plot(dt_list, llt_dict[alg + '_s']*0.001, linestyle='-', c=color, linewidth=lw, alpha=al,
                                  label=alg + f' $r^2=$ {correlation_LLT_s[alg + "corr(LLT)-s"][0]:.3f}')
            ax_LLT_right.set_ylabel(r'LLT [km]')
            for ax in [ax_LLT, ax_LWPad]:
                ax.set_ylabel(r'LWP [g\,m$^{-2}$]')
            ax_LWPad.legend(loc='upper left', facecolor='white')
            ax_LLT_right.legend(loc='upper left', facecolor='white')
            ax_LLT.legend(loc='center left', bbox_to_anchor=(0.001, 0.77), facecolor='white')
            ax_LCBH_diff.legend(loc='upper left', facecolor='white')
            ax_LWPad.set_ylim([-25, 750])
            ax_LLT.set_ylim([-25, 750])
            ax_LLT_right.set_ylim([-0.12, 3.])
            ax_LCBH_diff.grid()
            for ax in [ax_LCBH_diff, ax_LLT, ax_LLT_right, ax_LWPad, ax_prob]:
                ax.set_xlabel('Time [UTC]')
                ax.set_xlim([dt_list[0], dt_list[-1]])
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

            # rainflag
            raining = np.full(cat_post_xr['is_rain'].size, -10)
            _m0 = np.argwhere(cat_post_xr['is_rain'].values == 1)
            _m1 = np.argwhere(cat_post_xr['is_rain'].values == 0)
            for ax in [ax_LLT, ax_LWPad]:
                ax.scatter(dt_np[_m0], raining[_m0], marker='|', color='red', alpha=0.75)
                ax.scatter(dt_np[_m1], raining[_m1], marker='|', color='green', alpha=0.75)
            for ax in [ax_prob, ax_liq_overlapp]:
                ax.scatter(dt_np[_m0], raining[_m0] + 10.1, marker='|', color='red', alpha=0.75)
                ax.scatter(dt_np[_m1], raining[_m1] + 10.1, marker='|', color='green', alpha=0.75)

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
            pcmesh = ax_liq_overlapp.pcolormesh(
                dt_list, rg_list * 0.001, combi_liq_mask.T,
                cmap=matplotlib.colors.ListedColormap(tuple(_colors), "colors5"),
                vmin=0, vmax=7
            )
            cbaxes = inset_axes(ax_liq_overlapp, width="50%", height="5%", loc='upper left')
            cbar = fig.colorbar(pcmesh, cax=cbaxes, pad=0.05, orientation="horizontal")
            cbar.set_ticks(np.arange(0.5, len(_colornames) + 0.5))
            cbar.ax.set_xticklabels(_colornames, fontsize=8)
            # H
            values = np.ma.masked_where(~_cloud_mask, Vpost['liquid_probability'].values).T
            vlim = Vpost['liquid_probability'].attrs['var_lims']
            pcmesh = ax_prob.pcolormesh(
                dt_list, rg_list * 0.001, values,
                cmap=Vpost['liquid_probability'].attrs['colormap'],
                vmin=vlim[0], vmax=vlim[1]
            )
            cbaxes = inset_axes(ax_prob, width="50%", height="5%", loc='upper left')
            cbar = fig.colorbar(pcmesh, cax=cbaxes, fraction=0.05, pad=0.05, orientation="horizontal", extend='min')
            cbar.set_ticks(np.linspace(0.5, 1, 6))
            cbar.ax.set_ylabel(r'$P$ for CD [-]', labelpad=-180, rotation=0, y=0)

            # for icb in range(1, 4):
            ax_prob.scatter(dt_np[_cbh_mask], CBH_dict[f'CEILO1'][_cbh_mask] * 0.001, marker='*', color='red', alpha=0.75, s=0.1)

            # isotherms, axis labels
            for ax in [ax_Z, ax_beta, ax_liq_overlapp, ax_prob, ax_v, ax_wd, ax_ldr]:
                cont = ax.contour(
                    dt_list, rg_list * 0.001, (cat_post_xr['Temp'].values - 273.15).T, levels=[-40, -25, -15, -10, -5, 0, 5],
                    linestyles='dashed', colors=['black'], linewidths=[0.5], alpha=1
                )
                clabels = ax.clabel(cont, inline=1, fmt='%1.1f°C', fontsize=8)
                [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7)) for txt in clabels]
                ylim = [rg_list[0] * 0.001, rg_list[-1] * 0.001]
                ax.set_ylim(ylim)
                ax.set_ylabel('Height [km]')
                ax.set_xlabel('Time [UTC]')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            for letter, ax in zip('DE', [ax_violin_left, ax_violin_right]):
                ax.text(0.8, 0.8, rf'({letter})', transform=ax.transAxes, fontsize=16, fontweight='bold')
            for letter, ax in zip('ABCFGHIJKLM', [ax_Z, ax_beta, ax_v,
                                               ax_LCBH_diff, ax_wd,
                                               ax_prob, ax_LLT, ax_ldr,
                                               ax_liq_overlapp, ax_LWPad, ax_text]):
                ax.text(0.9, 0.8, rf'({letter})', transform=ax.transAxes, fontsize=16, fontweight='bold')

            ax_text.axis('off')

            text_kwars = {'transform': ax.transAxes, 'fontsize': 14, 'fontweight': 'bold'}
            ax_text.text(0.0, 1.0, f' {em}',)
            ax_text.text(0.0, 0.8, f' prec {perf[0]:.3f}  recall {perf[1]:.3f}', **text_kwars)
            ax_text.text(0.0, 0.7,f' acc {perf[2]:.3f}  f1 {perf[3]:.3f} ', **text_kwars)
            ax_text.text(0.0, 0.6,f' llt {corr_v[0]} / {corr_c[0]}', **text_kwars)
            ax_text.text(0.0, 0.5,f' lwp {corr_v[1]} / {corr_c[1]}', **text_kwars)
            ax_text.text(0.0, 0.4,f'lcbh {corr_v[2]} / {corr_c[2]}', **text_kwars)
            ax_text.text(0.0, 0.2,f'diff lcbh: mean = {np.ma.mean(diff_LCBH_v):.3f} / {np.ma.mean(diff_LCBH_c):.3f}', **text_kwars)
            ax_text.text(0.0, 0.1,f'diff lcbh: std = {np.ma.std(diff_LCBH_v):.3f} / {np.ma.std(diff_LCBH_c):.3f}', **text_kwars)

            print(f'case-{site}\n')
            print(pmatrix(em))
            print()
            print('prec, recall ... ', f'{perf[0]:.3f} & {perf[1]:.3f} & {perf[2]:.3f} & {perf[3]:.3f} & ', end='')
            print(f'{corr_v[0]} / {corr_c[0]} & {corr_v[1]} / {corr_c[1]} & {corr_v[2]} / {corr_c[2]} \\')
            print(f'\nmaximum calculated adiabatic liquid water path value = {np.nanmax(lwp_dict["Voodoo"]) * idk_factor:.2f} [g m-2]')
            print(f'diff lcbh: mean_v = {np.ma.mean(diff_LCBH_v):.3f}  std_v = {np.ma.std(diff_LCBH_v):.3f}')
            print(f'diff lcbh: mean_c = {np.ma.mean(diff_LCBH_c):.3f}  std_c = {np.ma.std(diff_LCBH_c):.3f}')
            print(f'maximum actual microwave radiometer liquid water path value = {np.nanmax(lwp_dict["mwr"]):.2f} [g m-2]')

            if title:
                fig.subplots_adjust(bottom=0.03, right=0.9, top=0.975, left=0.1, hspace=0.25, wspace=0.25)
            else:
                fig.subplots_adjust(bottom=0.03, right=0.9, top=0.975, left=0.1, hspace=0.15, wspace=0.25)
            # fig_name = f'{begin_dt:%Y%m%d-%H%M}-PRED_QL.png'
            fig.savefig(fig_name, facecolor='white', dpi=400)
            print(f' saved  {fig_name}')
            # print(f'correlation CloudnetPy versus Voodoo first liquid cloud base height = {correlation_LCBH:.3f}')

    _plot_all_together()

    return None




if __name__ == '__main__':
    ''' Main program for testing
    
    TODO: - remove liquid predictions below first (ceilometer) cloud base
          - remove profiles with lwp < 10 g m-2

    '''
    t0 = time.time()
    # setting device on GPU if available, else CPU

    _, agrs, kwargs = UT.read_cmd_line_args()
    # load data
    setup = kwargs['setup'] if 'setup' in kwargs else 1
    torch_settings = toml.load(os.path.join(voodoo_path, f'VnetSettings-{setup}.toml'))['pytorch']

    trained_model = kwargs['model'] if 'model' in kwargs else 'Vnet0x60de1687-fnX-gpu0-VN.pt'
    #'Vnet0x61555a5f-fn1-gpu0-VN.pt' #'Vnet0x6155c37c-fn1-gpu1-VN.pt'

    p = kwargs['p'] if 'p' in kwargs else 0.5
    date_str = str(kwargs['time']) if 'time' in kwargs else '20190801' #'20201230' #'20190801' #
    site = 'LIM' if int(date_str) > 20191001 else 'punta-arenas'
    ifn = str(kwargs['fn']) if 'fn' in kwargs else 'dbg'
    fac = float(kwargs['fac']) if 'fac' in kwargs else 1
    entire_day = kwargs['ed'] if 'ed' in kwargs else 'yes'
    lwp_smooth = float(kwargs['slwp']) if 'slwp' in kwargs else 20
    if 'resnet' in kwargs:
        torch_settings['resnet'] = kwargs['resnet']

    VoodooPredictor(
        date_str,
        tomlfile=f'{voodoo_path}/tomls/auto-trainingset-{date_str}-{date_str}.toml',
        datafile=f'{voodoo_path}/data/Vnet_6ch_noliqext/hourly/',
        modelfile=trained_model,
        filenumber=ifn,
        liquid_threshold=[p, 1.0],
        **torch_settings
    )
    cut_fn = trained_model.find('-fn') if '-fn' in trained_model else None
    VoodooAnalyser(
        date_str,
        site,
        modelfile=trained_model,
        liquid_threshold=[p, 1.0],
        time_lwp_smoothing=lwp_smooth * 60,     # in sec
        entire_day=entire_day,
    )