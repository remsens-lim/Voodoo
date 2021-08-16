#!/home/sdig/anaconda3/bin/python
import datetime
import os
import sys
import time
import glob

import numpy as np
import xarray as xr
import torch
import toml
import latextable
import scipy.interpolate


import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from libVoodoo.Plot import load_xy_style, load_cbar_style
from libVoodoo.Utils import find_bases_tops, load_training_mask
import pandas as pd

from cloudnetpy import plotting
from cloudnetpy.products import generate_classification

from scipy.ndimage import gaussian_filter


import libVoodoo.TorchModel as TM
import libVoodoo.TorchModel2 as TM2
import libVoodoo.TorchResNetModel as TMres
import libVoodoo.Utils as UT
from libVoodoo.Loader import dataset_from_zarr_new, VoodooXR, generate_multicase_trainingset

sys.path.append('/Users/willi/code/python/larda3/larda/')
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

import warnings
warnings.filterwarnings("ignore")

voodoo_path = os.getcwd()
pt_models_path = os.path.join(voodoo_path, f'torch_models/')

BATCH_SIZE  = 128
CLOUDNET = 'CLOUDNETpy94'
NCLASSES = 3
LIQ_EXT = '_noliquidextension'

line_colors = ['red', 'black', 'orange']
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

    print(f'Loading multiple zarr files ...... {tomlfile}')
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
#    outM = TM.evaluation_metrics(predicted_values, classes, status)
#
#    print(f'\nMetrics for raw output: {CAT_FILE_NAME}')
#    print(f'                   True   |          False   ')
#    print(f'positive   {outM["TP"]:12d}   |   {outM["FP"]:12d}')
#    print(f'negative   {outM["TN"]:12d}   |   {outM["FN"]:12d}')
#    print(f'')
#    print(f'   Precision = {outM["precision"]:.4f}')
#    print(f'         NPV = {outM["npv"]:.4f}')
#    print(f'      Recall = {outM["recall"]:.4f}')
#    print(f' Specificity = {outM["specificity"]:.4f}')
#    print(f' Overall acc = {outM["accuracy"]:.4f}')
#    print(f'   F1-scorer = {outM["F1-score"]:.4f}')
#    print(f'Jacard Index = {outM["Jaccard-index"]:.4f}')
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

    if entire_day == 'yes':
        hour_start, hour_end = None, None
        range_start, range_end = None, None
    else:
        if date_str == '20190801':
            hour_start, hour_end = 0, 9
            range_start, range_end = None, 6000
        elif date_str == '20210310':
            hour_start, hour_end = 0, 11
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
            hour_start, hour_end = 0, 5
            range_start, range_end = None, 8000

    if int(str(date_str)[:4]) > 2019:
        CLASSIFICATION_PATH = f'/media/sdig/leipzig/cloudnet/products/limrad94/classification-cloudnetpy{LIQ_EXT}/{str(date_str)[:4]}'
    else:
        CLASSIFICATION_PATH = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/products/limrad94/classification-cloudnetpy{LIQ_EXT}/{str(date_str)[:4]}'

    """
        ____ ___  ____ _  _    ____ ____ ___ ____ ____ ____ ____ _ ____ ____    ____ _ _    ____ 
        |  | |__] |___ |\ |    |    |__|  |  |___ | __ |  | |__/ | [__  |___    |___ | |    |___ 
        |__| |    |___ | \|    |___ |  |  |  |___ |__] |__| |  \ | ___] |___    |    | |___ |___ 

    """
    # original & new categorize file

    CLOUDNET_CLASS_FILE = f'{CLASSIFICATION_PATH}/{date_str}-{site}-classification-limrad94.nc'
    CLOUDNET_VOODOO_CAT_FILE = glob.glob(f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-categorize-limrad94-{modelfile[:-3]}*.nc')[0]
    CLOUDNET_VOODOO_CLASS_FILE = glob.glob(f'{pt_models_path}/{modelfile[:14]}/nc/{date_str}-{site}-classification-limrad94-{modelfile[:-3]}*.nc')[0]

    # interpolate model variables
    _cat_mod = xr.open_dataset(CLOUDNET_VOODOO_CAT_FILE, decode_times=False)
    _ts, _rg = _cat_mod['time'].values, _cat_mod['height'].values
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

    fig_name = f'{pt_models_path}/{modelfile[:14]}/plots/{modelfile[:-3]}-{date_str}-Analyzer-QL.png'

    time_range_slicer = {'time': slice(hour_start, hour_end), 'height': slice(range_start, range_end)}
    cat_post_xr = _cat_mod.sel(**time_range_slicer)
    class_post_xr = xr.open_dataset(CLOUDNET_VOODOO_CLASS_FILE, decode_times=False).sel(**time_range_slicer)
    class_orig_xr = xr.open_dataset(CLOUDNET_CLASS_FILE, decode_times=False).sel(**time_range_slicer)

    n_time, n_range = cat_post_xr['category_bits'].shape
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

    n_folds = 10
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
    if hour_start is None:
        hour_start = 0
    dt_list = [
        datetime.datetime.strptime(date_str+str(hour_start).zfill(2), '%Y%m%d%H') +
        datetime.timedelta(seconds=int(tstep * 60 * 60)) for tstep in ts_arr
    ]
    ts_list = [(dt - datetime.datetime(1970, 1, 1)).total_seconds() for dt in dt_list]

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
            'colormap': 'coolwarm',
            'rg_unit': 'km',
            'var_unit': '',
            'system': 'CloudnetPy',
            'var_lims': liquid_threshold,
            'range_interval': [0, 12]
        })
    Vpost['liquid_probability'].values = np.ma.masked_where(all_targets_mask, Vpost['liquid_probability'].values)

    ##### NEW
    _ccl = Vpost['CCLASS'].values
    _vcl = Vpost['VCLASS'].values
    _m0 = (Vpost['liquid_probability'].values > liquid_threshold[0]) \
          * (Vpost['liquid_probability'].values < liquid_threshold[1])
    _m1 = (_ccl == 1) + (_ccl == 3) + (_ccl == 5) + (_ccl == 7)
    _m2 = (_vcl == 1) + (_vcl == 3) + (_vcl == 5) + (_vcl == 7)

    #mdv = class_orig_xr['v'].copy()
    #kappa = UT.convection_index(mdv, dts=2, drg=1)
    #kappa_min, kappa_max = -0.3, 0.3
    #_ = np.ma.masked_less(kappa, kappa_min)
    #_ = np.ma.masked_greater(_, kappa_max)
    #kappa_mask = np.ma.getmaskarray(_)

    liquid_masks2D = {
        # Voodoo smoothed output gaussian filter with sigma=1
        'Voodoo': _m0,

        # ClouenetPy liquid droplet mask (classes= {1, 3, 5, 7})
        'CloudnetPy': _m1,

        # Combination of Voodoo liquid prediction and CloudnetPy liquid mask
        'Combination': _m2,

        #'both': _m0 * _m1
    }
    algos = liquid_masks2D.keys()

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
    lwp_nan[lwp_nan > 5000] = np.nan
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
    bases_tops_dict = {alg: find_bases_tops(liquid_masks2D[alg], Vpost.rg.values) for alg in liquid_masks2D.keys()}


    if True:
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(0, 1, 256))
        grey = np.array([220/256, 220/256, 220/256, 1])

        dimgrey = np.array([105/256, 105/256, 105/256, 0.75])
        newcolors[:1, :] = grey
        newcolors[-1:, :] = dimgrey
        newcmp = ListedColormap(newcolors)
        Vpost['liquid_probability'].attrs['colormap'] = newcmp

    '''
    ___  ____ ____ ___    ___  ____ ____ ____ ____ ____ ____ _ _  _ ____
    |__] |  | [__   |     |__] |__/ |  | |    |___ [__  [__  | |\ | | __
    |    |__| ___]  |     |    |  \ |__| |___ |___ ___] ___] | | \| |__]

    '''
    # cloud base/top height dict
    cloudnet_cbh = np.ma.masked_greater_equal(class_post_xr['cloud_base_height_agl'].values.copy(), 15000)

    # correlations mwr-lwp vs adiabatic-lwp
    correlation = {
        alg: [UT.correlation_coefficient(lwp_dict['mwr'], lwp_dict[alg]),
              UT.correlation_coefficient(lwp_dict['mwr_s'], lwp_dict[alg + '_s'])]
        for alg in algos
    }

    # correlations first cloud base height from liquid mask
    bases = {}
    for alg in algos:
        bases[alg] = np.zeros(n_time)
        for ind_time in range(n_time):
            if len(bases_tops_dict[alg][0][ind_time]['val_cb']) > 0:
                bases[alg][ind_time] = bases_tops_dict[alg][0][ind_time]['val_cb'][0]

    correlation_ceilo = {
        alg: UT.correlation_coefficient(cloudnet_cbh, base) for alg, base in bases.items()
    }


    """
        _    ____ ___ ____ _  _    ___ ____ ___  _    ____ ____ 
        |    |__|  |  |___  \/      |  |__| |__] |    |___ [__                  
        |___ |  |  |  |___ _/\_     |  |  | |__] |___ |___ ___] 

    """

    print_table = True
    if print_table:
        print('\n\nCorrelation MWR-LWP with adiabatic LWP from liquid pixel mask')
        print(f'{fig_name}\n\n')
        table = latextable.Texttable()
        table.set_deco(latextable.Texttable.HEADER)
        table.set_cols_dtype(['t',  # text
                              'f',  # float
                              'f',  # float
                              ])  # automatic
        table.set_cols_align(["l", "c", "c", ])
        table.add_rows(
            [["droplets mask", r"corr($\text{LWP}$)", "corr(LCBH)"], ] +
            [[
                alg,
                rf'{correlation[alg][1]:.2f}',  # / {correlation_lwp_50to1000[alg][1]:.2f}',
                rf'{correlation_ceilo[alg]:.2f}'  # / {correlation_ceilo_50to1000lwp[alg]:.2f}'
            ] for alg in algos]
        )
        print(table.draw() + "\n")
        #print(latextable.draw_latex(table,
        #                            caption="Correlation of MWR-LWP with calculated LWP from Voodoo predictions using adiabatic assumption"+
        #                            f" from model: {CLOUDNET_VOODOO_CAT_FILE}",
        #                            label=f"tab:correlations-fn{ifn}-gpu{igpu}") + "\n")
        ###################
        print(f'\n       maximum calculated adiabatic liquid water path value = {np.nanmax(lwp_dict["Voodoo"])*idk_factor:.2f} [g m-2]')
        print(f'maximum actual microwave radiometer liquid water path value = {np.nanmax(lwp_dict["mwr"]):.2f} [g m-2]')

    """
        ____ _  _ ____ _    _   _ ___  ____ ____    ____ _    
        |__| |\ | |__| |     \_/    /  |___ |__/    |  | |    
        |  | | \| |  | |___   |    /__ |___ |  \    |_\| |___ 

    """
    def voodoo_analyser_ql(ts_mask=None, title=''):

        nrows = 4
        xlim = [dt_list[0], dt_list[-1]]

        if ts_mask is None:
            ts_mask = np.full(len(dt_list), False)

        ts_mask2D = np.array([ts_mask for _ in range(Vpost['rg'].size)]).T

        fig, ax = plt.subplots(nrows=nrows, figsize=(12, nrows * 4))

        ax[0].bar(dt_list, np.ma.masked_where(ts_mask, lwp_dict['mwr_s']), width=0.001, color='royalblue', alpha=0.4, label='MWR')
        ax[0].plot(dt_list, np.ma.masked_where(ts_mask, lwp_dict['mwr']), color='royalblue', linewidth=lw / 2, alpha=al / 2)
        for alg, color in zip(algos, line_colors):
            ax[0].plot(dt_list, np.ma.masked_where(ts_mask, lwp_dict[alg + '_s']), color=color, linewidth=lw, alpha=al, label=alg)
            ax[0].plot(dt_list, np.ma.masked_where(ts_mask, lwp_dict[alg]), color=color, linewidth=lw / 2, alpha=al / 2)
        load_xy_style(ax[0], ylabel='Liquid Water Path [g m-2]')
        ax[0].set(xlabel='', xlim=xlim, ylim=[-25, 500], title=title)
        ax[0].legend()

        """
            _    _ ____ _  _ _ ___     _    _ _  _ ____ _    _   _ _  _ ____ ____ ___  
            |    | |  | |  | | |  \    |    | |_/  |___ |     \_/  |__| |  | |  | |  \ 
            |___ | |_\| |__| | |__/    |___ | | \_ |___ |___   |   |  | |__| |__| |__/ 

        """
        fig, ax[1] = tr.plot_timeheight2(
            Vpost['liquid_probability'],
            fig=fig, ax=ax[1],
            title=f"Voodoo Predicted Liquid Likelyhood {date_str}",
            fontweight='normal',
            rg_converter=True,
            label='',
            mask=ts_mask2D,
        )
        # cloud bases scatter plots
        bases = {alg: UT.get_bases_or_tops(dt_list, bases_tops_dict[alg], key='cb') for alg in algos}
        for alg, color in zip(algos, line_colors):
            ax[1].scatter(*bases[alg]['first'], c=color, s=1.1, alpha=0.55, edgecolor='face', label=alg + ' LCBH')
        ax[1].legend(loc='upper left')
        squeeze = ["32%", "11%"]
        for i in range(2):
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size=squeeze[i], pad=0.2)
            cax.axis('off')
            fig.add_axes(cax)

        # cloudnet target classification
        new_class = Vpost['CCLASS'].copy()
        tr_mask = load_training_mask(Vpost['CCLASS'].values, Vpost['CSTATUS'].values)
        new_class.values[tr_mask] = 0

        # cloudnet + voodoo classes
        plot_kwargs = {'var_lims': [0, 10], 'fontweight': 'normal', 'rg_converter': True, 'mask': ts_mask2D}
        fig, ax[2] = tr.plot_timeheight2(Vpost['CCLASS'], fig=fig, ax=ax[2], title=f"CloudnetPy target class quicklook {date_str}", **plot_kwargs)
        fig, ax[3] = tr.plot_timeheight2(Vpost['VCLASS'], fig=fig, ax=ax[3], title=f"Voodoo target class quicklook {date_str}", **plot_kwargs)

        return fig, ax

    fig, ax = voodoo_analyser_ql(title=fig_name[fig_name.rfind('Vnet'):-4])
    fig.savefig(fig_name, dpi=200)

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
    p = kwargs['p'] if 'p' in kwargs else 0.4
    date_str = str(kwargs['time']) if 'time' in kwargs else '20190223'
    site = 'LIM' if int(date_str) > 20191001 else 'punta-arenas'
    ifn = str(kwargs['fn']) if 'fn' in kwargs else 'dbg'
    fac = float(kwargs['fac']) if 'fac' in kwargs else 1
    entire_day = kwargs['ed'] if 'ed' in kwargs else 'yes'
    lwp_smooth = float(kwargs['slwp']) if 'slwp' in kwargs else 10
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
        liquid_threshold=[p, 0.99],
        time_lwp_smoothing=lwp_smooth * 60,     # in sec
        entire_day=entire_day,
    )