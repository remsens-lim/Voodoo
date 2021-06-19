#!/home/sdig/anaconda3/bin/python
import os
import sys
import time
from tqdm.auto import tqdm, trange
import pandas as pd

import toml
import torch
import datetime
import xarray as xr
import numpy as np
import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
from libVoodoo.Loader import features_from_nc, VoodooXR

VOODOO_PATH = os.getcwd()
DATA_PATH = os.path.join(VOODOO_PATH, 'data/Vnet_6ch_noliqext/10folds_all/')
pt_models_path = os.path.join(VOODOO_PATH, 'torch_models/')
MODEL_TOML = os.path.join(VOODOO_PATH, 'HP_12chdp2.toml')
TEST_PATH = os.path.join(VOODOO_PATH, f'data/Vnet_6ch_noliqext/validation/validation_testingset-1-10folds_all-ND.zarr')
# model=model-1609964168-20eps.pt
# model=model-1610033363-4eps.pt
iGPU = 0
DEVICE_train = f'cuda:{iGPU}'
BATCH_SIZE = 512
CLOUDNET = 'CLOUDNETpy94'
NCLASSES = 2

if __name__ == '__main__':
    # setting device on GPU if available, else CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)
    print()

    _, agrs, kwargs = UT.read_cmd_line_args()

    n_folds = 10
    X_list = []
    y_list = []
    zero_column = np.zeros(n_folds, dtype=int)

    df_nclasses = pd.DataFrame(
        {class_: zero_column for class_ in ['CD', 'non-CD']},
        index=[f'fold{i}' for i in range(n_folds)]
    )

    NPOL = 1
    NFOLDS = 1
    NCHANNELS = 6
    NDBINS = 256
    batch_size = 512
    df_meandist = np.zeros((NCLASSES, NPOL, NCHANNELS, NDBINS))
    df_stddist = np.zeros((NCLASSES, NPOL, NCHANNELS, NDBINS))

    iFN_mean = []
    iFN_std = []
    for iFN in trange(NFOLDS):
        if NFOLDS > 1:
            TEST_PATH = os.path.join(DATA_PATH, f'20181127-20190927-{iFN}-10folds_all-ND.zarr')
        print(f'Loading compressed zarr file ...... {TEST_PATH}')
        X, y = TM.VoodooNet.fetch_data(TEST_PATH, shuffle=True)
        class_dist = np.array([np.count_nonzero(y==i) for i in range(NCLASSES)])
        df_nclasses.loc[f'fold{iFN}'] = class_dist[1:]
        print(f'\nGPU:{iGPU} {np.sum(class_dist, dtype=np.int):12d}   total samples: fn{iFN}')

        iterator = tqdm(
            range(0, len(X), batch_size),
            ncols=100,
            unit=f' batches'
        )

        iFN_batches_mean = []
        iFN_batches_std = []
        for ii, i in enumerate(iterator):
            if ii > 100:
                break
            batch_X = X[i:i + batch_size]
            batch_y = y[i:i + batch_size]

            for iclass in range(class_dist.size):
                if class_dist[iclass] > 0:
                    df_meandist[iclass, :, :, :] = np.array(torch.mean(batch_X[batch_y == iclass, :, :, :], 0))
                    df_stddist[iclass, :, :, :] = np.array(torch.std(batch_X[batch_y == iclass, :, :, :], 0))

            iFN_batches_mean.append(df_meandist)
            iFN_batches_std.append(df_stddist)

        iFN_mean.append(np.stack(iFN_batches_mean))
        iFN_std.append(np.stack(iFN_batches_std))

    df_nclasses.to_csv(os.path.join(DATA_PATH, 'TargetDistribution_nosl.csv'))
    np.save(os.path.join(DATA_PATH, 'TargetMeanDistribution_nosl.npy'), np.stack(iFN_mean))
    np.save(os.path.join(DATA_PATH, 'TargetStdDistribution_nosl.npy'), np.stack(iFN_std))
    dummy = 5
