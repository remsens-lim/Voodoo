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

voodoo_path = '/home/sdig/code/larda3/voodoo/'
model_path = join(voodoo_path, 'HP_12chdp2.toml')
pt_models_path = join(voodoo_path, f'torch_models/')
data_path = join(voodoo_path, 'data_12chdp/xarray_zarr/')
valid_path = join(data_path, '20190801-20190801-allclasses-X-12ch2pol.zarr')

PT_settings = toml.load(f'{model_path}')['pytorch']
BATCH_SIZE = 500
EPOCHS = 5

if __name__ == '__main__':

    # ./TorchTrain.py fn=0 gpu=0
    _, agrs, kwargs = UT.read_cmd_line_args()

    FN = int(kwargs['fn']) if 'fn' in kwargs else 1

    dc = 'deepconv2only-'

    #train_path = join(data_path, f'20190801-20190801-allclasses-X-12ch2pol.zarr') # debug case
    train_path = join(data_path, f'20181127-20190927-{FN}-allclasses-{dc}X-12ch2pol.zarr')
    date_str = re.search('\d{8}-\d{8}', train_path).group(0)

    # setting device on GPU if available, else CPU
    iGPU = int(kwargs['gpu']) if 'gpu' in kwargs else 0
    PT_settings.update({'dev': torch.device(f'cuda:{iGPU}' if torch.cuda.is_available() else 'cpu')})

    h.change_dir(join(pt_models_path, 'TEST_desp/'))

    test_only = False
    if test_only:

        m_name = 'model-1610545407-3eps-FN7.pt'
        model = TM.VoodooNet.load(m_name)
        print(model)
        model.print_nparams()

    else:

        m_name = f'model-{int(time.time())}-{dc}{EPOCHS}eps-FN{FN}.pt'

        garbage = [3, 6, 7, 8, 9, 10]
        # load data training features & labels
        X_train, y_train = TM.VoodooNet.fetch_data(
            train_path,
            balance=-1,
            remove_classes=[],
            garbage=garbage,
            shuffle=True,
            mask_below=-1,
            despeckle=True,
        )

        # load data validation features & labels
        X_test, y_test = TM.VoodooNet.fetch_data(
            valid_path,
            shuffle=False
        )

        class_dist = UT.log_number_of_classes(np.array(y_train)).astype(np.int)

        print(f'\nGPU:{iGPU} {np.sum(class_dist, dtype=np.int):12d}   total for FN{FN}')
        for i in range(class_dist.size):
            print(f'{class_dist[i]:18d}   {UT.cloudnetpy_classes[i]}')

        # new model
        torch.cuda.set_device(iGPU)
        model = TM.VoodooNet(X_train.shape, 11, **PT_settings)
        #print(model)
        model.print_nparams()
        stat = model.optimize(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE, epochs=EPOCHS, dev=PT_settings["dev"])
        model.save(m_name)
        print(f'\nmodel saved: {m_name}')
        model.create_acc_loss_graph(stat, m_name)


    ####################################################################################################################
    #
    #
    #
    ### more testing
    case_list = {
        key: join(data_path, f'{key}-{key}-allclasses-X-12ch2pol.zarr')
        for key in ['20190801', '20190223', '20190313', '20190331']
    }

    class_attrs = {
                'colormap': 'cloudnet_target_new',
                'rg_unit': 'km',
                'var_unit': '',
                'system': 'VOODOO',
                'var_lims': [0, 10],
                #'statistics': stat,
                'batch_size': BATCH_SIZE,
                'training_set': train_path,
                'model_path': m_name
            }
    ncases = len(case_list)
    iax = 0
    for key, val in case_list.items():
        print(f'Plot test cases {key}')
        X_test, y_test = TM.VoodooNet.fetch_data(val, shuffle=False, batch_size=BATCH_SIZE)
        xrTest = TM.VoodooNet.fetch_2Ddata(
            join(voodoo_path, f'data_12chdp/xarray_zarr/{key}-{key}-X-12ch2pol2D.zarr')
        )
        prediction = model.testing(X_test, batch_size=BATCH_SIZE, dev=PT_settings["dev"])
        prediction = prediction.to('cpu')
        V_prediction = model.to_nc(X_test, xrTest, prediction, **class_attrs)

        indices = TM.VoodooNet.random_subset(V_prediction, var='CLASS')
        f, ax = PLT.grid_plt(V_prediction, xrTest, indices)
        f.suptitle(f'training-file-number: FN{FN},  batch-size: {BATCH_SIZE}', fontsize=12, fontweight='semibold')
        fig_name = f'{key}-FN{FN}-{dc}QL.png'
        f.subplots_adjust(top=0.95, bottom=0.05, right=1, left=0)  # , hspace = 0, wspace = 0)
        f.savefig(fig_name, dpi=250)  # , bbox_inches = 'tight')
        print(f"fig saved: {fig_name}")
