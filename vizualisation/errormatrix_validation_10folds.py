#!/home/sdig/anaconda3/bin/python
import latextable
import time
from tqdm.auto import tqdm

t0_marker = time.time()  # first thing = get unix time stamp for Vnet file name

import logging
import os
from os.path import join

import numpy as np
import toml
import torch

import libVoodoo.TorchModel as TM
import libVoodoo.TorchResNetModel as TMres
import libVoodoo.Utils as UT
from libVoodoo.Plot import create_quicklook, create_acc_loss_graph
from libVoodoo.Utils import change_dir
import matplotlib.pyplot as plt


SL = '_noliqext'
voodoo_path = os.getcwd()
# model_path = join(voodoo_path, 'Vnet2021.toml')
model_path = join(voodoo_path, 'HP_12chdp2.toml')
# model_path = join(voodoo_path, 'Vnet_experimental01.toml')
pt_models_path = join(voodoo_path, f'torch_models/')
valid_path = join(voodoo_path, f'data/Vnet_6ch{SL}/validation/validation_testingset-1-10folds_all-ND.zarr')
valid_path = join('/home/sdig/code/Voodoo/data/Vnet_6ch_noliqext/10folds_all',
                  'debugger_trainingset-1-10folds_all-ND.zarr')

PT_settings = toml.load(f'{model_path}')['pytorch']
BATCH_SIZE = 4096
EPOCHS = 2
DROP = 0.0
BALANCE = -1
mode = '10folds_all'
remove_classes = []
# remove_classes = [3, 6, 7, 8, 9, 10]
garbage_classes = []
# garbage_classes = [2, 3, 4, 6, 7, 8, 9, 10]
groups = {1: [1, 3, 5, 7], 2: [0, 2, 4, 6, 8, 9, 10]}
NCLASSES = len(groups) + 1
class_name_list = ['droplets available', 'no droplets available']

# next try multilabel classification https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/

TM.log_TM.setLevel(logging.CRITICAL)

if __name__ == '__main__':

    # ./TorchTrain.py fn=0 gpu=0
    _, agrs, kwargs = UT.read_cmd_line_args()

    data_path = join(voodoo_path, f'data/Vnet_6ch{SL}/{mode}/')
    # train_path = join(data_path, f'debugger_trainingset-1-10folds_all-ND.zarr') # debug case

    # setting device on GPU if available, else CPU
    PT_settings.update({'dev': torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')})

    # load data
    p = kwargs['p'] if 'p' in kwargs else 0.4
    igpu = kwargs['gpu'] if 'gpu' in kwargs else 1
    date_str = str(kwargs['time']) if 'time' in kwargs else '20201220'
    site = 'LIM' if int(date_str) > 20191001 else 'punta-arenas'
    ifn = str(kwargs['fn']) if 'fn' in kwargs else 'X'
    fac = float(kwargs['fac']) if 'fac' in kwargs else 5
    entire_day = bool(kwargs['ed']) if 'ed' in kwargs else None
    lwp_smooth = float(kwargs['slwp']) if 'slwp' in kwargs else 10
    device = torch.device(f'cuda:{igpu}' if torch.cuda.is_available() else 'cpu')
    trained_model = f"{kwargs['model']}" if 'model' in kwargs else ''
    trained_model = pt_models_path + trained_model

    trained_models = [
        'Vnet0x60757189',
        'Vnet0x6076da29',
        'Vnet0x60794daf',
        'Vnet0x607ae5d7',
        'Vnet0x607c2141',
        'Vnet0x607c4bc4',
        'Vnet0x607d3708',
        'Vnet0x607d6b39',
        'Vnet0x607f4d25',
        'Vnet0x607ff89f',
        'Vnet0x60812cc3'
    ]

    dp_list = ['.3', '.6', '.9']
    int_strings = ['TP', 'TN', 'FP', 'FN']
    float_strings = ['precision', 'recall', 'accuracy', 'F1-score']
    fn_strings = [f'fn{ifn}' for ifn in range(10)] + ['fnX.3', 'fnX.6', 'fnX.9']
    n_mod = len(trained_models)
    n_fn = 11


    Vnet_labels = [f'Vnet{k}' for k in range(len(trained_models))]

    a = [('', f'{trained_models[i][6:]}', '') for i in range(len(trained_models))]
    b = [('', f'{fn_strings[i]}', '') for i in range(10)]
    flat_model_list = [item for sublist in a for item in sublist]
    flat_fn_list = [item for sublist in b for item in sublist]


    def print_latextable(arr, model, ifn=0):
        cut = model.rfind('/') + 1
        newline_labels = [rf'{a[8:]}' for a in trained_models]
        table1 = latextable.Texttable()
        table1.set_deco(latextable.Texttable.HEADER)
        table1.set_cols_align(["r", ] + ['c'] * arr.shape[0])
        table1.add_rows(
            [["", ] + newline_labels] +
            [[key, ] + list(val.astype(int)) for key, val in zip(int_strings, arr[:, :4].T)] +
            [[key, ] + list(val) for key, val in zip(float_strings, arr[:, 4:].T)]
        )
        print(table1.draw() + "\n")
        print(latextable.draw_latex(
            table1,
            caption=f"Performance metrics from model: {model[cut:].replace('_', '-')}",
            label=f"tab:metrics-fn{ifn}-gpu{igpu}") + "\n"
              )


    def save_to_npy():
        arr = np.zeros((n_fn + 1, len(trained_models), 8))
        for ifn in tqdm(range(n_fn), unit=' fold'):
            valid_path = join(data_path, f'20181127-20190927-{ifn}-{mode}-ND.zarr')
            try:
                X_test, y_test = TM.VoodooNet.fetch_data(valid_path, shuffle=False, merge_classes=groups)
            except:
                continue  # skip if not available

            class_dist = UT.log_number_of_classes(np.array(y_test))
            print(f'\nGPU:{igpu} {np.sum(class_dist, dtype=np.int):12d}   total samples: fn{ifn}')
            for i in range(class_dist.size):
                print(f'{class_dist[i]:18d}   {class_name_list[i]}')

            for ind_mod, mod in enumerate(trained_models):
                mod_path = pt_models_path + mod + f'-fnX-eps2-bs512-bl-1-gpu{igpu}_noliqext-VN.pt'

                # new model
                model = TM.VoodooNet.load(mod_path)
                print(mod_path)
                model.print_nparams()

                ann_output = model.testing(X_test, batch_size=BATCH_SIZE, dev=device)
                ann_output = ann_output.to('cpu').numpy()
                prediction = np.full(y_test.shape, 2)
                prediction[ann_output[:, 1] > p] = 1

                print('Scores of predictive performance\n')
                metr = TM.evaluation_metrics(prediction, y_test.numpy())
                del metr['npv'], metr['specificity'], metr['Jaccard-index'], metr['array']

                for j, (key, imet) in enumerate(metr.items()):
                    arr[ifn, ind_mod, j] = imet

                print_latextable(arr[ifn, :, :], mod_path, ifn=ifn)
            np.save(f'numpy-allmodels-fn{ifn}-gpu{igpu}.npy', arr[ifn, :, :])

        return arr


    def load_from_npy():
        arr = np.zeros((n_fn, 3, len(trained_models), 8))
        for ifn in tqdm(range(n_fn), unit=' fold'):
            for igpu in range(3):
                name=f'numpy-allmodels-fn{ifn}-gpu{igpu}.npy'
                try:
                    arr[ifn, igpu, :, :] = np.load(name)[:n_mod, :]
                except:
                    print(f'skip: {name}')
                    continue  # skip if not available
        return np.ma.masked_less_equal(arr, 0)


    # arr = save_to_npy()

    # dimensions: (n_fn=10, n_alpha=3, n_models=11, n_stats=8)
    arr = load_from_npy()
    arr = arr[:10, :, :, :]

    mean_arr = np.ma.mean(arr, axis=(0, 2))  # mean of all folds
    sum_arr = np.ma.sum(arr, axis=(0, 2))  # sum of all folds

    table1 = latextable.Texttable()
    table1.set_deco(latextable.Texttable.HEADER)
    table1.set_cols_align(["r", ] + ['c'] * mean_arr.shape[0])
    table1.add_rows(
        [["", ] + fn_strings[-mean_arr.shape[0]:]] +
        [[key, ] + list(val.astype(int)) for key, val in zip(int_strings, mean_arr[:, :4].T)] +
        [[key, ] + list(val) for key, val in zip(float_strings, mean_arr[:, 4:].T)]
    )
    print(latextable.draw_latex(
        table1,
        caption=f"Performance metrics from model: {trained_models[0][:6]}",
        label=f"tab:metrics-alldata-fnX-stats") + "\n"
          )

    # 2nd table
    mean_arr = np.ma.mean(arr, axis=(0, 1))  # mean of all folds
    table2 = latextable.Texttable()
    table2.set_deco(latextable.Texttable.HEADER)
    table2.set_cols_align(["r", ] + ['c'] * arr.shape[2])
    table2.add_rows(
        [[trained_models[0][:8],] + [tm[8:] for tm in trained_models]] +
        [[key, ] + list(val.astype(int)) for key, val in zip(int_strings, mean_arr[:, :4].T)] +
        [[key, ] + list(val) for key, val in zip(float_strings, mean_arr[:, 4:].T)]
    )
    print(latextable.draw_latex(
        table2,
        caption=f"Performance metrics from model: {trained_models[0][:6]}",
        label=f"tab:metrics-alldata-Vnet0x60-stats") + "\n"
          )

    # 3rd table
    mean_arr = np.ma.mean(arr, axis=(1, 2))  # mean of all folds
    table2 = latextable.Texttable()
    table2.set_deco(latextable.Texttable.HEADER)
    table2.set_cols_align(["r", ] + ['c'] * arr.shape[0])
    table2.add_rows(
        [["", ] + fn_strings[:arr.shape[0]]] +
        [[key, ] + list(val.astype(int)) for key, val in zip(int_strings, mean_arr[:, :4].T)] +
        [[key, ] + list(val) for key, val in zip(float_strings, mean_arr[:, 4:].T)]
    )
    print(latextable.draw_latex(
        table2,
        caption=f"Performance metrics from model: {trained_models[0][:6]}",
        label=f"tab:metrics-alldata-fni-stats") + "\n"
          )



    ############# 0th plot
    fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(4, 12))
    for ivalue in range(8):
        tmp = arr[:, :, :, ivalue].transpose(1, 0, 2)
        ax[ivalue].boxplot(tmp.reshape((3, arr.shape[0] * arr.shape[2])).T, widths=0.2)
        ax[ivalue].set(ylabel=(int_strings + float_strings)[ivalue])
        # ax[ivalue].set(xticklabels=[])
        if ivalue > 3:
            ax[ivalue].set(ylim=[0, 1])
        ax[ivalue].set(xticklabels=fn_strings[-3:])
    # ax[0].legend()
    fig.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.05)
    fig.savefig(f'{date_str}-boxplots-fnXVSmetrics.png')

    ############# 1st plot
    fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(12, 12))
    for ivalue in range(8):
        for imod in range(len(trained_models)):
            ax[ivalue].boxplot(arr[:, :, imod, ivalue], positions=np.array([0.25, 0.5, 0.75]) + imod, widths=0.2)
            ax[ivalue].set(ylabel=(int_strings + float_strings)[ivalue])

        if ivalue > 3:
            ax[ivalue].set(ylim=[0, 1])
        ax[ivalue].set(xticklabels=flat_model_list)
    # ax[0].legend()
    fig.subplots_adjust(left=0.075, right=0.975, top=0.975, bottom=0.025)
    fig.savefig(f'{date_str}-boxplots-modelsVSmetrics.png')

    ############# 2nd plot
    fig, ax = plt.subplots(nrows=8, ncols=1, figsize=(12, 12))
    for ivalue in range(8):
        for ifn in range(10):
            ax[ivalue].boxplot(arr[ifn, :, :, ivalue].T, positions=np.array([0.25, 0.5, 0.75]) + ifn, widths=0.2)
            ax[ivalue].set(ylabel=(int_strings + float_strings)[ivalue])

        if ivalue > 3:
            ax[ivalue].set(ylim=[0, 1])

        ax[ivalue].set(xticklabels=flat_fn_list)
    # ax[0].legend()
    fig.subplots_adjust(left=0.075, right=0.975, top=0.975, bottom=0.025)
    fig.savefig(f'{date_str}-boxplots-fnVSmetrics.png')

