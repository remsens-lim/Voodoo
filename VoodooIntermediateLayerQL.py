#!/home/sdig/anaconda3/bin/python
import glob
import os
import sys
import time
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch

import torch.nn as nn
from tqdm.auto import tqdm

import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
from libVoodoo.Loader import dataset_from_zarr_new
import sys
VOODOO_PATH = os.getcwd()

pt_models_path = os.path.join(VOODOO_PATH, f'torch_models/')
MODEL_TOML = os.path.join(VOODOO_PATH, 'VnetSettings-1.toml')

#MODEL_TOML = os.path.join(VOODOO_PATH, 'HP_12chdp2.toml')

SL = ''
BATCH_SIZE = 512
CLOUDNET = 'CLOUDNETpy94'
NCLASSES = 3

if __name__ == '__main__':
    t0 = time.time()
    # setting device on GPU if available, else CPU

    _, agrs, kwargs = UT.read_cmd_line_args()
    # load data
    trained_model = os.path.join(f"{kwargs['model']}" if 'model' in kwargs else 'Vnet0x60de1687-fnX-gpu0-VN.pt')
    date_str = str(kwargs['time']) if 'time' in kwargs else '20190223' # 20190801
    TOML_FILE = f'{VOODOO_PATH}/tomls/{date_str}.toml'
    DATA_PATH = os.path.join(VOODOO_PATH, f'data/Vnet_6ch_noliqext{SL}/hourly/')
    torch_settings = toml.load(os.path.join(VOODOO_PATH, f'VnetSettings-1.toml'))['pytorch']
    torch_settings.update({'dev': 'cpu'})

    start = 5000 #9032  # 37 #3057

    nfeatures = 150 #170


    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[:1, :] = np.array([220 / 256, 220 / 256, 220 / 256, 1])
    newcmp = ListedColormap(newcolors)

    print(f'Loading multiple zarr files ...... {TOML_FILE}')
    ds_ND, ds_2D = dataset_from_zarr_new(
        DATA_PATH=DATA_PATH,
        TOML_PATH=TOML_FILE,
        CLOUDNET=CLOUDNET,
        RADAR='limrad94',
        TASK='predict',
    )
    X = ds_ND['features'][:, :, :, 0].values  # good!
    X = X[:, :, :, np.newaxis]
    y = ds_ND['targets'].values

    X = X.transpose(0, 3, 2, 1)
    X_test = torch.Tensor(X)
    y_test = torch.Tensor(y)

    trmodel = f'{pt_models_path}/{trained_model[:14]}/{trained_model}'

    model = TM.VoodooNet(X_test.shape, NCLASSES, **torch_settings)
    model.load_state_dict(torch.load(trmodel, map_location=model.device)['state_dict'])

    print(X.shape)
    print(trmodel)
    model.print_nparams()

    interm0 = model.convolution_network.conv2d_0(X_test)
    interm1 = model.convolution_network.conv2d_1(interm0)
    interm2 = model.convolution_network.conv2d_2(interm1)
    interm3 = model.convolution_network.conv2d_3(interm2)
    interm4 = model.convolution_network.conv2d_4(interm3)
    intermediate = [interm0, interm1, interm2, interm3, interm3]

    input_features = X_test.detach().numpy()
    input_labels = y_test.detach().numpy()

    flattened = model.flatten(interm4)
    dense0 = model.dense_network.dense_0(flattened)
    dense1 = model.dense_network.dense_1(dense0)
    intermediate_dense = [dense0, dense1]

    min_f, max_f = X_test.min(), X_test.max()

    norm = matplotlib.colors.SymLogNorm(linthresh=0.1, linscale=0.1, vmin=-1.0, vmax=1.0)
    
    UT.change_dir(f'{pt_models_path}/voodoo_intermediate_feature/{trained_model[:trained_model.find("-")]}/')
    plot_conv = True
    if plot_conv:

        for iconv in [0]: #range(len(model.convolution_network)):

            interm_feautures = intermediate[iconv].detach().numpy()
            interm_feautures = np.transpose(interm_feautures, axes=(0, 1, 3, 2))
            fig = plt.figure(constrained_layout=True, figsize=(26, 10))
            gs = fig.add_gridspec(nrows=nfeatures, ncols=24)
            for irow in tqdm(range(nfeatures)):  # ref_dsND['features'].shape[0]):


                #if irow == 0:
                perm = np.argsort(np.mean(interm_feautures[irow +start, :, :, :], axis=(1, 2)))

                min_f = input_features[irow + start, 0, :, :].min()
                max_f = input_features[irow + start, 0, :, :].max()

                concatin = np.concatenate(interm_feautures[irow + start, perm, :, :], axis=0).T

                axcol0 = fig.add_subplot(gs[nfeatures - irow - 1, :6])
                axcolN = fig.add_subplot(gs[nfeatures - irow - 1, 6:-1])
                axcol0.set_facecolor('silver')
                axcolN.set_facecolor('silver')
                axcol0.pcolormesh(input_features[irow + start, 0, :, :], vmin=min_f, vmax=max_f, cmap='coolwarm')
                pcmesh = axcolN.pcolormesh(concatin, norm=norm, cmap='coolwarm')
                axcol0.set_ylabel(f'{irow} - {UT.cloudnetpy_classes[int(input_labels[irow + start])]}', rotation=0)
                axcolN.set(yticklabels=[], ylabel='', )
                if irow > 0:
                    axcol0.set(xticklabels=[], yticklabels=[], xlabel='')
                    axcolN.set(xticklabels=[], xlabel='')
            axcolcbar = fig.add_subplot(gs[:, -1])
            cbar = fig.colorbar(pcmesh, cax=axcolcbar)

            cbar.set_label(label=f'conv-layer-{iconv} output')
            fig.subplots_adjust(bottom=0.1, right=0.95, top=0.95, left=0.05, hspace=0.05, wspace=0.05)
            fig.suptitle(f'input: ({input_features.shape[2:]}) to ({interm_feautures.shape[1:]}) to ({np.concatenate(interm_feautures[0, :, :, :], axis=0).shape})')
            fig.savefig(f'{date_str}-spec_intermediate-conv1d_{iconv}-output.png', dpi=500)
            #fig.savefig(f'{date_str}-spec_intermediate-conv1d_{iconv}-output-T.eps', format='eps')
            continue

    plot_dense = True
    if plot_dense:
        n_dense_layers = len(model.dense_network)
        for idense in [1]: #range(n_dense_layers):
            if idense == n_dense_layers - 1:
                smax = nn.Softmax(dim=1)
                intermediate_dense[idense] = smax(intermediate_dense[idense])

            interm_feautures = intermediate_dense[idense].detach().numpy()
            n_sqrt = int(np.sqrt(interm_feautures.shape[1]))

            fig = plt.figure(constrained_layout=True, figsize=(26, 10))
            gs = fig.add_gridspec(nrows=nfeatures, ncols=24)
            for irow in tqdm(range(nfeatures)):  # ref_dsND['features'].shape[0]):

                min_f = input_features[irow + start, :].min()
                max_f = input_features[irow + start, :].max()

                concatin = interm_feautures[irow + start, :, np.newaxis].T

                axcol0 = fig.add_subplot(gs[nfeatures - irow - 1, :6])
                axcolN = fig.add_subplot(gs[nfeatures - irow - 1, 6:-1])
                axcol0.set_facecolor('silver')
                axcolN.set_facecolor('silver')
                axcol0.pcolormesh(input_features[irow + start, 0, :, :], vmin=min_f, vmax=max_f, cmap='coolwarm')
                if idense < n_dense_layers - 1:
                    pcmesh = axcolN.pcolormesh(concatin, norm=norm, cmap='coolwarm')
                else:
                    pcmesh = axcolN.pcolormesh(concatin[:, 1:], vmin=0.4, vmax=1, cmap=newcmp)
                axcol0.set_ylabel(f'{irow} - {UT.cloudnetpy_classes[int(input_labels[irow + start])]}', rotation=0)
                axcolN.set(yticklabels=[], ylabel='', )
                if irow > 0:
                    axcol0.set(xticklabels=[], yticklabels=[], xlabel='')
                    axcolN.set(xticklabels=[], xlabel='')
            axcolcbar = fig.add_subplot(gs[:, -1])
            cbar = fig.colorbar(pcmesh, cax=axcolcbar)
            axcol0.grid('off')
            axcolN.grid('off')

            cbar.set_label(label=f'dense-layer-{idense} output')
            fig.subplots_adjust(bottom=0.1, right=0.95, top=0.95, left=0.05, hspace=0.05, wspace=0.05)
            #fig.suptitle(f'input: ({input_features.shape[2:]}) to ({interm_feautures.shape[1:]}) to ({})')
            fig.savefig(f'{date_str}-spec_intermediate-dense_{idense}-output-T.png', dpi=500)
            #fig.savefig(f'{date_str}-spec_intermediate-dense_{idense}-output-T.eps', format='eps')
