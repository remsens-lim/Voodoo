#!/home/sdig/anaconda3/bin/python
import logging
import os
import time
from os.path import join

import numpy as np
import toml
import torch

import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
from libVoodoo.Utils import change_dir
from libVoodoo.Plot import create_quicklook, create_acc_loss_graph, grid_plt

voodoo_path = os.getcwd()
model_path = join(voodoo_path, 'HP_12chdp2.toml')
pt_models_path = join(voodoo_path, f'torch_models/')
data_path = join(voodoo_path, 'data/HP_12chdp2/10folds_all/wSL/')
valid_path = join(voodoo_path, 'data/20190801-20190801-allclasses-X-12ch2pol.zarr')

PT_settings = toml.load(f'{model_path}')['pytorch']
BATCH_SIZE = 256
EPOCHS = 2
balance = 225_000
remove_classes = [3, 6, 7, 8, 9, 10]

log_TM = TM.log_TM
log_TM.setLevel(logging.CRITICAL)

if __name__ == '__main__':

    # ./TorchTrain.py fn=0 gpu=0
    _, agrs, kwargs = UT.read_cmd_line_args()

    fn = int(kwargs['fn']) if 'fn' in kwargs else 0
    bl = int(kwargs['bl']) if 'bl' in kwargs else balance
    iGPU = int(kwargs['gpu']) if 'gpu' in kwargs else 0
    md = kwargs['model'] if 'model' in kwargs else ''
    nf = kwargs['mkdir'] if 'mkdir' in kwargs else ''
    addmore = kwargs['addmore'] if 'addmore' in kwargs else True

    #train_path = join(data_path, f'20190801-20190801-allclasses-X-12ch2pol.zarr') # debug case
    train_path = join(data_path, f'20181127-20190927-{fn}-10folds_all-ND.zarr')

    # setting device on GPU if available, else CPU
    PT_settings.update({'dev': torch.device(f'cuda:{iGPU}' if torch.cuda.is_available() else 'cpu')})

    # make new dir
    change_dir(join(pt_models_path, nf))

    if len(md) > 0:
        m_name = md
        model = TM.VoodooNet.load(m_name)
        print(model)
        model.print_nparams()
        Vnet_label = m_name.split('-')[0]

    else:
        Vnet_label = f'Vnet{hex(int(time.time()))}'
        m_name = f'{Vnet_label}-fn{fn}-eps{EPOCHS}-bs{BATCH_SIZE}-bl{bl}.pt'
        ts_kwargs = {
            'balance': balance,
            'garbage': None,
            'remove_classes': remove_classes,
            'shuffle': True,
            'mask_below': -1,
            'despeckle': False
        }

        X_train, y_train = TM.VoodooNet.fetch_data(train_path, **ts_kwargs)
        if addmore:
            train_path2 = join(data_path, f'20181127-20190927-{fn+5}-10folds_all-ND.zarr')
            X_train2, y_train2 = TM.VoodooNet.fetch_data(train_path2, **ts_kwargs)
            X_train = torch.cat((X_train, X_train2), 0)
            y_train = torch.cat((y_train, y_train2), 0)

        X_test, y_test = TM.VoodooNet.fetch_data(valid_path, shuffle=False)

        class_dist = UT.log_number_of_classes(np.array(y_train))
        print(f'\nGPU:{iGPU} {np.sum(class_dist, dtype=np.int):12d}   total samples: fn{fn}')
        for i in range(class_dist.size):
            print(f'{class_dist[i]:18d}   {UT.cloudnetpy_classes[i]}')

        # new model
        torch.cuda.set_device(iGPU)
        model = TM.VoodooNet(X_train.shape, 11, **PT_settings)

        model.print_nparams()
        stat = model.optimize(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE, epochs=EPOCHS, dev=PT_settings["dev"])
        model.save(m_name)
        print(f'\nmodel saved: {m_name}')

        # creat accuracy/loss graph
        fig, ax = create_acc_loss_graph(stat)
        fig.savefig(m_name.replace('.pt', '.png'))
        print(f"fig saved: {m_name.replace('.pt', '.png')}")


    ####################################################################################################################
    #
    #
    #
    ### more testing
    case_list = {
        key: join(voodoo_path, f'data/{key}-{key}-allclasses-X-12ch2pol.zarr')
        for key in ['20190801']#, '20190223', '20190313', '20190331']
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

        # @cpu
        X_test, y_test = TM.VoodooNet.fetch_data(val, shuffle=False, batch_size=BATCH_SIZE)
        xrTest = TM.VoodooNet.fetch_2Ddata(
            join(voodoo_path, f'data/{key}-{key}-X-12ch2pol2D.zarr')
        )

        # @gpu
        prediction = model.testing(X_test, batch_size=BATCH_SIZE, dev=PT_settings["dev"])
        #lrp = model.lrp(X_test, prediction, dev=PT_settings["dev"])

        # @cpu
        prediction = prediction.to('cpu')
        V_prediction = model.to_nc(X_test, xrTest, prediction, **class_attrs)

        indices = TM.VoodooNet.random_subset(V_prediction, var='CLASS')

        #f, ax = grid_plt(V_prediction, xrTest, indices)

        f, ax = create_quicklook(V_prediction['CLASS'])
        f.suptitle(f'training-file-number: fn{fn},  batch-size: {BATCH_SIZE}', fontsize=12, fontweight='semibold')
        fig_name = f'{key}-{Vnet_label}-fn{fn}-QL.png'
        #f.subplots_adjust(top=0.95, bottom=0.05, right=1, left=0)  # , hspace = 0, wspace = 0)
        f.savefig(fig_name, dpi=250)  # , bbox_inches = 'tight')
        print(f"\nfig saved: {fig_name}")
