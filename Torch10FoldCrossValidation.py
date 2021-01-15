#!/home/sdig/anaconda3/bin/python
import os
from os.path import join
import re
import sys
import time
import toml
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import logging

import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
import libVoodoo.Plot as PLT

PATH_TO_LARDA = '/home/sdig/code/larda3/larda/'
sys.path.append(PATH_TO_LARDA)
import pyLARDA.helpers as h
import pyLARDA.Transformations as TR

if __name__ == '__main__':

    voodoo_path = '/home/sdig/code/larda3/voodoo/'
    data_path = join(voodoo_path, 'data_12chdp/xarray_zarr/')

    model_path = join(voodoo_path, 'HP_12chdp2.toml')
    pt_models_path = join(voodoo_path, f'torch_models/')
    valid_path = join(data_path, '20190801-20190801-allclasses-X-12ch2pol.zarr')

    BATCH_SIZE = 512
    EPOCHS = 10

    RUN_NAME = 'TEST/'

    # ./TorchTrain.py fn=0 gpu=0
    _, agrs, kwargs = UT.read_cmd_line_args()

    h.change_dir(join(pt_models_path, RUN_NAME))

    for i in range(1, 4):
        os.system(f'/bin/bash {voodoo_path}torch_training_parallel.sh {i}')


