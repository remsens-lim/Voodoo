#!/home/sdig/anaconda3/bin/python
import time

t0_marker = time.time()  # first thing = get unix time stamp for Vnet file name

import logging
import os
from os.path import join

import toml

from typing import List, Tuple, Dict
import xarray as xr
import numpy as np
np.random.seed(42069)

import torch
torch.manual_seed(42069)

import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT

voodoo_path = os.getcwd()
data_path = join('/media/sdig/Voodoo_Data/10folds_all/')

groups = {1: [1, 3, 5, 7],  2: [0, 2, 4, 6, 8, 9, 10]}
NCLASSES = len(groups)+1
class_name_list = ['droplets available', 'no droplets available']

# next try multilabel classification https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
TM.log_TM.setLevel(logging.CRITICAL)

if __name__ == '__main__':
    ''' Main program for training
    
    '''
    print('start')
    # ./TorchTrain.py fn=0 gpu=0
    _, agrs, kwargs = UT.read_cmd_line_args()

    setup = kwargs['setup'] if 'setup' in kwargs else 1
    torch_settings = toml.load(os.path.join(voodoo_path, f'VnetSettings-{setup}.toml'))['pytorch']

    if 'fn' in kwargs:
        torch_settings['fn'] = kwargs['fn']
    if 'balance' in kwargs:
        torch_settings['balance'] = int(kwargs['balance'])

    torch_settings['dp'] = kwargs['dp'] if 'dp' in kwargs else 0

    if torch_settings['fn'] == 'dbg':
        train_path = join(data_path, f'debugger_trainingset-1-10folds_all-ND.zarr')
    else:
        #date_range = '20181127-20190927' # PA
        date_range = '20201216-20211001' # LIM
        train_path = join(data_path, f'{date_range}-{torch_settings["fn"]}-10folds_all-ND.zarr')

    ts_kwargs = {
        'balance': torch_settings['balance'],
        'shuffle': True,
        'merge_classes': groups,
        'n_split': 10
    }

    def fetch_data(
            path: str,
            shuffle: bool = True,
            balance: int = False,
            n_split: int = 0,
            merge_classes: Dict = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        t0 = time.time()
        with xr.open_zarr(path, consolidated=False) as data:
            X = data['features'].values
            y = data['targets'].values

        print(f'time elapsed reading {path} :: {int(time.time() - t0)} sec')

        assert len(X) != 4, f'Input data has wrong shape: {X.shape}'

        ichannel = 0
        X = X[:, :, :, ichannel]  # vertical + horizontal polarization
        X = X[:, :, :, np.newaxis]

        # nsamples, npolarization, ntimesteps, nDopplerbins
        X = X.transpose(0, 3, 2, 1)
        UT.log_number_of_classes(y, text=f'\nsamples per class in {path}')

        if merge_classes is not None:
            tmp = y.copy()
            for key, val in merge_classes.items(): # i from 0, ..., ngroups-1
                for jclass in val:
                    tmp[y == jclass] = key
            y = tmp

        if balance > 0:
            for i in range(11):
                X, y = TM.VoodooNet.remove_randomely(balance, i, X, y)
            UT.log_number_of_classes(y, text=f'\nsamples per class balanced')

        X = torch.Tensor(X)
        y = torch.Tensor(y)
        y = y.type(torch.LongTensor)

        if shuffle:
            perm = torch.randperm(len(y))
            X, y = X[perm], y[perm]

        if 1 < n_split:
            n = X.shape[0]//n_split
            X = list(torch.split(X, n, dim=0))
            y = list(torch.split(y, n, dim=0))

            if len(X) > n_split:
                X = X[:n_split-len(X)]
                y = y[:n_split-len(y)]

        return X, y

    n_fnX = 10
    TDX, TDy = [], []
    for iFN in range(n_fnX):
        train_path = join(data_path, f'{date_range}-{iFN}-10folds_all-ND.zarr')
        X_tmp, y_tmp = fetch_data(train_path, **ts_kwargs)
        TDX.append(X_tmp)
        TDy.append(y_tmp)

    X_train = []
    y_train = []
    for i in range(n_fnX):
        tmp_sets_X = []
        tmp_sets_y = []
        for j in range(n_fnX):
            tmp_sets_X.append(TDX[j][i])
            tmp_sets_y.append(TDy[j][i])

        X_train.append(torch.cat(tmp_sets_X, axis=0))
        y_train.append(torch.cat(tmp_sets_y, axis=0))

    for i in range(n_fnX):
        class_dist = UT.log_number_of_classes(np.array(y_train[i]))
        print(f'\nportion {i}: {np.sum(class_dist, dtype=np.int):12d}   total samples: fn{torch_settings["fn"]}')
        for j in range(class_dist.size):
            print(f'{class_dist[j]:18d}   {class_name_list[j]}')

        torch.save(X_train[i], data_path + f'portion{i}_bl_{torch_settings["balance"]}_X.pt')
        torch.save(y_train[i], data_path + f'portion{i}_bl_{torch_settings["balance"]}_y.pt')
