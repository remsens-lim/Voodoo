#!/home/sdig/anaconda3/bin/python
import os
import re
import sys
import time
import toml
import torch
import xarray as xr
import logging
import numpy as np
import matplotlib.pyplot as plt

import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
import libVoodoo.Plot as PLT
from generate_trainingset import VoodooXR

PATH_TO_LARDA = '/home/sdig/code/larda3/larda/'
sys.path.append(PATH_TO_LARDA)
import pyLARDA.helpers as h
import pyLARDA.Transformations as TR

trained_model = 'model-1610638465-deepconv2only-10eps-FN1.pt'
voodoo_path = '/home/sdig/code/larda3/voodoo/'
model_path = os.path.join(voodoo_path, 'HP_12chdp2.toml')
pt_models_path = os.path.join(voodoo_path, f'torch_models/TEST_desp/')
PT_settings = toml.load(f'{model_path}')['pytorch']
BATCH_SIZE = 512
NCLASSES = 11
dc = 'deepconv2only-'


if __name__ == '__main__':

    TM.log_TM.setLevel(logging.INFO)

    _, agrs, kwargs = UT.read_cmd_line_args()
    # load data
    m_name = kwargs['model'] if 'model' in kwargs else trained_model
    FNtrain = re.search('FN\d', m_name).group(0)

    # setting device on GPU if available, else CPU
    PT_settings.update({'dev': torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')})

    h.change_dir(pt_models_path)

    model = TM.VoodooNet.load(m_name)
    print(model)
    model.print_nparams()

    ####################################################################################################################
    #
    #
    #
    ### more testing
    case = {
        '20190801': os.path.join(voodoo_path, 'data_12chdp/xarray_zarr/20190801-20190801-allclasses-X-12ch2pol.zarr'),
        #'20190223': os.path.join(voodoo_path, 'data_12chdp/xarray_zarr/20190223-20190223-allclasses-X-12ch2pol.zarr'),
        #'20190313': os.path.join(voodoo_path, 'data_12chdp/xarray_zarr/20190313-20190313-allclasses-X-12ch2pol.zarr'),
        #'20190331': os.path.join(voodoo_path, 'data_12chdp/xarray_zarr/20190331-20190331-allclasses-X-12ch2pol.zarr'),
    }

    class_attrs = {
        'colormap': 'cloudnet_target_new',
        'rg_unit': 'km',
        'var_unit': '',
        'system': 'VOODOO',
        'var_lims': [0, 10],
        'batch_size': BATCH_SIZE,
        'model_path': m_name
    }
    ncases = len(case)

    for key, val in case.items():
        print(f'Plot test cases {key}')
        X_test, y_test = TM.VoodooNet.fetch_data(val, shuffle=False)
        xrTest = TM.VoodooNet.fetch_2Ddata(
            os.path.join(voodoo_path, f'data_12chdp/xarray_zarr/{key}-{key}-X-12ch2pol2D.zarr')
        )
        prediction = model.testing(X_test, batch_size=BATCH_SIZE, dev=PT_settings["dev"])
        prediction = prediction.to('cpu')
        V_prediction = TM.VoodooNet.to_nc(X_test, xrTest, prediction, **class_attrs)

        indices = TM.VoodooNet.random_subset(V_prediction,  var='CLASS')
        f, ax = PLT.grid_plt(V_prediction, xrTest, indices)
        f.suptitle(f'training-file-number: {FNtrain},  batch-size: {BATCH_SIZE}', fontsize=12, fontweight='semibold')
        fig_name = m_name.replace('.pt', f'Validation-{key}-Cases-{FNtrain}-ql.png')
        f.subplots_adjust(top=0.95, bottom=0.05, right=1, left=0)
        f.savefig(fig_name, dpi=450)
        print(f"fig saved: {fig_name}")
