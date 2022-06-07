#!/home/sdig/anaconda3/bin/python
import glob
import logging
import sys, os
import re

import torch
from Voodoo.Utils import read_cmd_line_args
from Voodoo.Loader import dataset_from_zarr_new

PARAMETER = {
    #'system': 'limrad94',
    'system': 'KAZR',
    'cloudnet': 'CLOUDNET',
    #'cloudnet': 'CLOUDNETpy94',
    'larda_path': '../larda3/larda',
    'save': True,
    'n_channels': 6,

    'hourly_path': '/projekt2/ac3data/B07-data/arctic-mosaic/CloudNet/input/voodoo/hourly_zarr/',
    #'hourly_path': '/projekt2/remsens/data_new/site-campaign/punta-arenas_dacapo-peso/cloudnet/calibrated/voodoo/hourly_zarr/',
    #'hourly_path': '/projekt2/remsens/data_new/site-campaign/leipzig-lim/cloudnet/calibrated/voodoo/hourly_zarr/',

    'site': 'mosaic_rs01',  # ('leipzig_rs01', 'lacros_dacapo_rs01', 'mosaic_rs01')
}

if __name__ == '__main__':

    ANN_INI_FILE = 'VnetSettings-1.toml'
    task = 'train'

    _, agrs, kwargs = read_cmd_line_args()

    data_list = sorted(glob.glob(PARAMETER['hourly_path'] + '2019/*.zarr'))[:10]

    X, y = dataset_from_zarr_new(
        data_list,
        TASK='train',
        PLOT=False,
    )

    #numbers = log_number_of_classes(y, text=f'\nsamples per class')
    #print(numbers)
    X_T = torch.tensor(X)
    y_T = torch.tensor(y)

    # ND variables
    FILE_NAME = f'{1}-fold-.nc'
    ND_ds.to_netcdf(store=FILE_NAME, mode='w')
    print(f'save :: {FILE_NAME}')
