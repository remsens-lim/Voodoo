#!/home/sdig/anaconda3/bin/python
import time

t0_marker = time.time()  # first thing = get unix time stamp for Vnet file name

import logging
import os
from os.path import join

import numpy as np
import toml
import torch

from testVoodooNet import VoodooPredictor, VoodooAnalyser

import libVoodoo.TorchModel as TM
import libVoodoo.TorchModel2 as TM2
import libVoodoo.TorchResNetModel as TMres
import libVoodoo.Utils as UT
from libVoodoo.Plot import create_quicklook, create_acc_loss_graph
from libVoodoo.Utils import change_dir

def gen_Vnet_aux(kwargs):
    name = f'{kwargs["Vnet_label"]}\n' \
           f'fn{kwargs["fn"]}\n' \
           f'eps{kwargs["epochs"]}\n' \
           f'bs{torch_settings["batch_size"]}\n' \
           f'bl{kwargs["balance"]}\n' \
           f'gpu{kwargs["iGPU"]}\n' \
           f'noliqext'
    return name

voodoo_path = os.getcwd()
data_path = join('/media/sdig/Voodoo_Data/10folds_all/')
pt_models_path = join(voodoo_path, f'torch_models/')
valid_path = join(voodoo_path, f'data/Vnet_6ch_noliqext/validation/validation_testingset-1-10folds_all-ND.zarr')

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
    if 'dropout' in kwargs:
        torch_settings['dropout'] = kwargs['dropout']
    if 'gpu' in kwargs:
        torch_settings['iGPU'] = int(kwargs['gpu'])
    if 'model' in kwargs:
        torch_settings['model'] = kwargs['model']
    if 'batch_size' in kwargs:
        torch_settings['batch_size'] = kwargs['batch_size']
    if 'epochs' in kwargs:
        torch_settings['epochs']  = kwargs['epochs']
    if 'resnet' in kwargs:
        torch_settings['resnet'] = kwargs['resnet']

    torch_settings['dp'] = kwargs['dp'] if 'dp' in kwargs else 0

    if 'p' in kwargs:
        torch_settings['p'] = kwargs['p']

    if torch_settings['fn'] == 'dbg':
        train_path = join(data_path, f'debugger_trainingset-1-10folds_all-ND.zarr')
    else:
        #date_range = '20181127-20190927' # PA
        date_range = '20201216-20211001' # LIM
        train_path = join(data_path, f'{date_range}-{torch_settings["fn"]}-10folds_all-ND.zarr')


    # setting device on GPU if available, else CPU
    torch_settings['dev'] = torch.device(f'cuda:{torch_settings["iGPU"]}' if torch.cuda.is_available() else 'cpu')

    # make new dir
    change_dir(pt_models_path)

    # name the model
    torch_settings['Vnet_label'] = f'Vnet{hex(int(t0_marker))}'
    m_name = f'{torch_settings["Vnet_label"]}-fn{torch_settings["fn"]}-gpu{torch_settings["iGPU"]}-{torch_settings["resnet"]}.pt'
    torch_settings['auxiliary'] = gen_Vnet_aux(torch_settings)

    ts_kwargs = {
        'balance': torch_settings['balance'],
        'garbage': torch_settings['garbage_classes'],
        'remove_classes': torch_settings['remove_classes'],
        'shuffle': True,
        'drop_pct': torch_settings["dp"],
        'merge_classes': groups,
    }

    X_test, y_test = TM.VoodooNet.fetch_data(valid_path, shuffle=False, merge_classes=groups)
    X_test, y_test = X_test[:10000], y_test[:10000]
    if torch_settings['fn'] == 'X':
        TDX, TDy = [], []
        for iFN in range(10):
            train_path = join(data_path, f'20181127-20190927-{iFN}-10folds_all-ND.zarr')
            X_tmp, y_tmp = TM.VoodooNet.fetch_data(train_path, **ts_kwargs)
            TDX.append(X_tmp)
            TDy.append(y_tmp)

        X_train, y_train = torch.cat(TDX, 0), torch.cat(TDy, 0)
    else:
        X_train, y_train = TM.VoodooNet.fetch_data(train_path, **ts_kwargs)

    class_dist = UT.log_number_of_classes(np.array(y_train))
    print(f'\nGPU:{torch_settings["iGPU"]} {np.sum(class_dist, dtype=np.int):12d}   total samples: fn{torch_settings["fn"]}')
    for i in range(class_dist.size):
        print(f'{class_dist[i]:18d}   {class_name_list[i]}')


    # new model
    torch.cuda.set_device(torch_settings["iGPU"])
    if torch_settings["resnet"] == 'RN':
        resnet_flag = '-RN'
        model = TMres.ResNet18(img_channels=X_train.shape[1], num_classes=NCLASSES)
    elif torch_settings["resnet"] == 'VRN':
        resnet_flag = f'-VRN'
        model = TM2.VoodooNet(X_train.shape, NCLASSES, **torch_settings)
    else:
        resnet_flag = f'-VN'
        model = TM.VoodooNet(X_train.shape, NCLASSES, **torch_settings)


    model.print_nparams()

    stat = model.optimize(
        X_train, y_train, X_test, y_test,
        batch_size=torch_settings['batch_size'],
        epochs=torch_settings['epochs'],
    )

    _path = pt_models_path + torch_settings['Vnet_label']
    os.makedirs(_path, exist_ok=True)
    model.save(path=_path+'/'+m_name, aux=torch_settings)
    print(f'\nmodel saved: {m_name}')

    # creat accuracy/loss graph

    os.makedirs(_path+'/plots/', exist_ok=True)
    fig, ax = create_acc_loss_graph(stat)
    fig.savefig(_path+'/plots/'+m_name.replace('.pt', '.png'))
    print(f"fig saved: {m_name.replace('.pt', '.png')}")


    if True:
        date_str = '20190801'
        VoodooPredictor(
            date_str,
            tomlfile=f'{voodoo_path}/tomls/auto-trainingset-{date_str}-{date_str}.toml',
            datafile=f'{voodoo_path}/data/Vnet_6ch_noliqext/hourly/',
            modelfile=m_name,
            filenumber=torch_settings["fn"],
            liquid_threshold=[torch_settings["p"], 1],
            **torch_settings
        )
        cut_fn = m_name.find('-fn') if '-fn' in m_name else None
        VoodooAnalyser(
            date_str,
            'punta-arenas',
            modelfile=m_name,
            liquid_threshold=[torch_settings["p"], 1],
            time_lwp_smoothing=10 * 60,     # in sec
            entire_day='false',
        )
