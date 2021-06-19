#!/home/sdig/anaconda3/bin/python
import os
import sys
import time

import latextable
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
from libVoodoo.Loader import preproc_ini

sys.path.append(preproc_ini['larda']['path'])

VOODOO_PATH = os.getcwd()
TORCH_MODELS_PATH = os.path.join(VOODOO_PATH, f'torch_models/')

if __name__ == '__main__':
    t0 = time.time()
    SL = '_noliqext'

    _, agrs, kwargs = UT.read_cmd_line_args()

    date_str = str(kwargs['time']) if 'time' in kwargs else '20190801'

    # fni trained models
    fni_list = ['Vnet0x6060b481',
                'Vnet0x6076cb4b',
                #'Vnet0x60793ea5',
                'Vnet0x607ad73a',
                'Vnet0x607c1277',
                'Vnet0x607c3d16',
                'Vnet0x607d2875',
                'Vnet0x607d5c8d',
                #'Vnet0x607f3e98',
                'Vnet0x607fd4b2',
                'Vnet0x60811e16'
                ]
    # fnX trained models
    fnX_list = ['Vnet0x60757189',
                'Vnet0x6076da29',
                #'Vnet0x60794daf',
                'Vnet0x607ae5d7',
                'Vnet0x607c2141',
                'Vnet0x607c4bc4',
                'Vnet0x607d3708',
                'Vnet0x607d6b39',
                #'Vnet0x607f4d25',
                'Vnet0x607ff89f',
                'Vnet0x60812cc3'
                ]
    site = 'LIM' if int(date_str[:4]) > 2019 else 'punta-arenas'

    fn_strings = [f'fn{ifn}' for ifn in range(10)] + ['fnX.3', 'fnX.6', 'fnX.9']


    def create_table(model, flag=False):
        if int(str(date_str)[:4]) > 2019:
            CLASSIFICATION_PATH = f'/media/sdig/leipzig/cloudnet/products/limrad94/classification-cloudnetpy/{str(date_str)[:4]}'
        else:
            CLASSIFICATION_PATH = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/products/limrad94/classification-cloudnetpy/{str(date_str)[:4]}'

        ################################
        # original classification
        hour_start, hour_end = None, 9
        range_start, range_end = None, 6000
        time_range_slicer = {'time': slice(hour_start, hour_end), 'height': slice(range_start, range_end)}
        CLOUDNET_CLASS_FILE = f'{CLASSIFICATION_PATH}/{date_str}-{site}-classification-limrad94.nc'
        class_orig_xr = xr.open_dataset(CLOUDNET_CLASS_FILE, decode_times=False).sel(**time_range_slicer)

        metrics = []
        n = 3 if flag else 10

        for i in range(n):
            if flag:
                CLOUDNET_VOODOO_CAT_FILE = f'{TORCH_MODELS_PATH}/{date_str}-{site}-categorize-limrad94-{model}-fnX-gpu{i}{SL}.nc'
                CLOUDNET_VOODOO_CLASS_FILE = f'{TORCH_MODELS_PATH}/{date_str}-{site}-classification-limrad94-{model}-fnX-gpu{i}{SL}.nc'
            else:
                igpu = np.mod(i, 4)
                CLOUDNET_VOODOO_CAT_FILE = f'{TORCH_MODELS_PATH}/{date_str}-{site}-categorize-limrad94-{model}-fn{i}-gpu{igpu}{SL}.nc'
                CLOUDNET_VOODOO_CLASS_FILE = f'{TORCH_MODELS_PATH}/{date_str}-{site}-classification-limrad94-{model}-fn{i}-gpu{igpu}{SL}.nc'

            try:
                cat_post_xr = xr.open_dataset(CLOUDNET_VOODOO_CAT_FILE, decode_times=False).sel(**time_range_slicer)
                class_post_xr = xr.open_dataset(CLOUDNET_VOODOO_CLASS_FILE, decode_times=False).sel(**time_range_slicer)
            except:
                continue

            n_time, n_range = cat_post_xr['category_bits'].shape
            voodoo_liq_mask = (cat_post_xr['pred_liquid_prob_1'].values - cat_post_xr['pred_liquid_prob_0'].values) > 0.5

            voodoo_post_classification = class_post_xr['target_classification'].values.copy()
            voodoo_post_liq_mask = np.full((n_time, n_range), False)
            voodoo_post_liq_mask[
                (voodoo_post_classification == 1) +
                (voodoo_post_classification == 3) +
                (voodoo_post_classification == 5) +
                (voodoo_post_classification == 7)] = True

            cloudnet_classification = class_orig_xr['target_classification'].values.copy()
            cloudnet_status = class_orig_xr['detection_status'].values.copy()
            cloudnet_liq_mask = np.full((n_time, n_range), False)
            cloudnet_liq_mask[
                (cloudnet_classification == 1) +
                (cloudnet_classification == 3) +
                (cloudnet_classification == 5) +
                (cloudnet_classification == 7)
                ] = True

            tmp = TM.evaluation_metrics(voodoo_liq_mask, cloudnet_classification, cloudnet_status)
            del tmp['array'], tmp['specificity'], tmp['Jaccard-index'], tmp['npv']
            metrics.append(tmp)

        arr = np.zeros((n, 8))
        for i, met in enumerate(metrics):
            for j, (key, imet) in enumerate(met.items()):
                arr[i, j] = imet

        return metrics, arr

    conc_list =[]
    for fni, fnX in tqdm(zip(fni_list, fnX_list)):
        metrics, arr = create_table(fni)
        metricsfnX, arrfnX = create_table(fnX, True)

        conc = np.concatenate((arr, arrfnX), axis=0).T
#        table = latextable.Texttable()
#        table.set_deco(latextable.Texttable.HEADER)
#        table.set_cols_align(["r", ] + ['c'] * conc.shape[1])
#        table.add_rows(
#            [["", ] + fn_strings] +
#            [[key, ] + list(val) for key, val in zip(metrics[0].keys(), conc)]
#        )
#        # print(table.draw() + "\n")
#        print(latextable.draw_latex(
#            table,
#            caption=f"Performance metrics from model: {date_str}: models fni: {fni}   fnX: {fnX}",
#            label=f"tab:metrics-{date_str}") + "\n")

        conc_list.append(conc)

    # mean, std
    int_strings = ['TP', 'TN', 'FP', 'FN']
    float_strings = ['precision', 'recall', 'accuracy', 'F1-score']
    mean = np.mean(conc_list, axis=0)
    std = np.std(conc_list, axis=0)
    table = latextable.Texttable()
    table.set_deco(latextable.Texttable.HEADER)
    table.set_cols_align(["r", ] + ['c'] * mean.shape[1])
    table.add_rows(
        [["", ] + fn_strings] +
        [[key, ] + list(val.astype(int)) for key, val in zip(int_strings, mean[:4, :])] +
        [[key, ] + list(val) for key, val in zip(float_strings, mean[4:, :])]
    )
    # print(table.draw() + "\n")
    print(latextable.draw_latex(
        table,
        caption=f"Performance metrics from model: {date_str}: mean of all models ",
        label=f"tab:metrics-{date_str}mean") + "\n")

    table = latextable.Texttable()
    table.set_deco(latextable.Texttable.HEADER)
    table.set_cols_align(["r", ] + ['c'] * std.shape[1])
    table.add_rows(
        [["", ] + fn_strings] +
        [[key, ] + list(val.astype(int)) for key, val in zip(int_strings, std[:4, :])] +
        [[key, ] + list(val) for key, val in zip(float_strings, std[4:, :])]
    )
    # print(table.draw() + "\n")
    print(latextable.draw_latex(
        table,
        caption=f"Performance metrics from model: {date_str}: standart deviation of all models ",
        label=f"tab:metrics-{date_str}std") + "\n")

    print()

    import matplotlib.pyplot as plt

    boxes = np.array(conc_list).transpose((0, 2, 1))
    n_models, n_fn, n_metrics = boxes.shape
    import seaborn as sb

    Vnet_labels = [f'Vnet{k}' for k in range(boxes.shape[0])]

    fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(16, 9))
    for i in range(8):
        ax[i, 0].boxplot(boxes[:, :, i])
        ax[i, 0].set(ylabel=(int_strings + float_strings)[i])
        for k in range(boxes.shape[0]):
            ax[i, 1].plot(boxes[k, :, i], label=f'Vnet-fni: {fni_list[k]} :: Vnet-fnX: {fnX_list[k]}', linewidth=0.5)
        for j in range(2):
            if i == 7:
                ax[i, 0].set(xticklabels=fn_strings)
                ax[i, 1].set_xticklabels(Vnet_labels, rotation=45, ha="right")
            else:
                ax[i, j].set(xticklabels=[])
            if i > 3:
                ax[i, j].set(ylim=[0, 1])

    ax[0, 1].legend()
    fig.savefig(f'{date_str}-boxplots.png')

    n_models, n_fn, n_metrics = boxes.shape
    szenarios = [[0, 1], [2, 3], [4, 5], [6, 7]]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(4 * 4, 4))
    for ii, szen in enumerate(szenarios):
        im, jm = szen[0], szen[1]
        ax[ii].scatter(boxes[:, :10, im], boxes[:, :10, jm], label=f'fni')
        ax[ii].scatter(boxes[:, 10:, im], boxes[:, 10:, jm], label=f'fnX')
        ax[ii].set(xlabel=(int_strings + float_strings)[im], ylabel=(int_strings + float_strings)[jm])

    ax[-1].plot(105, 200, 'ro')

    ax[-1].legend()

    fig.savefig(f'{date_str}-cluster.png')
    # plot data




