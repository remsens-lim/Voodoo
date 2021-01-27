#!/home/sdig/anaconda3/bin/python
import os
import sys

import toml
import torch
import datetime
import xarray as xr
import numpy as np
import libVoodoo.TorchModel as TM
import libVoodoo.Utils as UT
from generate_trainingset import load_features_from_nc, VoodooXR

VOODOO_PATH = '/home/sdig/code/larda3/voodoo/'
DATA_PATH = f'{VOODOO_PATH}/data_12chdp/'
pt_models_path = os.path.join(VOODOO_PATH, f'torch_models/')
MODEL_TOML = '/home/sdig/code/larda3/voodoo/HP_12chdp2.toml'
# model=model-1609964168-20eps.pt
# model=model-1610033363-4eps.pt
igpu = 0
DEVICE_train = f'cuda:{igpu}'
DEVICE_test = f'cuda:{igpu}'

_DEFAULT_CHANNELS = 12
_DEFAULT_DOPPBINS = 256
BATCH_SIZE = 500
CLOUDNET= 'CLOUDNETpy94'
NCLASSES = 11

if __name__ == '__main__':
    # setting device on GPU if available, else CPU
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', DEVICE)
    print()
    
    _, agrs, kwargs = UT.read_cmd_line_args()
    # load data
    trained_model = os.path.join(
        pt_models_path,
        f"{kwargs['model']}" if 'model' in kwargs else 'TEST_desp/Vnet0x6007eab3-dc2-fn6-eps3-bs128-bl200000.pt'
    )
    task = 'inference' if 'task' not in kwargs else kwargs['task']

    if task == 'inference':
        date_str = str(kwargs['time'])
        dt_begin = datetime.datetime.strptime(f"{date_str}-0001", '%Y%m%d-%H%M')
        dt_end = dt_begin + datetime.timedelta(minutes=24*60-2)
        print(f'Loading data from nc ...... {dt_begin:%Y%m%d %H:%M} to {dt_end:%Y%m%d %H:%M}')

        X, y, _, mask, _, ts, rg = load_features_from_nc(
            time_span=[dt_begin, dt_end],
            VOODOO_PATH=VOODOO_PATH,
            data_path=DATA_PATH,
            system='limrad94',
            cloudnet=CLOUDNET,
            save=False,
            n_channels=_DEFAULT_CHANNELS,
            ann_settings_file=MODEL_TOML,
            site='leipzig_gpu' if dt_begin.year > 2019 else 'lacros_dacapo_gpu',
            dual_polarization=True,
        )

        X = X[:, :, 4:10, 0]  # good!
        X = X[:, :, :, np.newaxis]
        X = X.transpose(0, 3, 2, 1)
        X_test = torch.Tensor(X)
        y_test = torch.Tensor(y)
    elif task == 'test':
        date_str = kwargs['time'] if 'time' in kwargs else '20190801'
        toml_file = f'{VOODOO_PATH}/tomls/auto-trainingset-{date_str}-{date_str}.toml'
        chunk_heads = [chunk for chunk in UT.load_case_file(toml_file).keys()]

        print(f'Loading multiple zarr files ...... {toml_file}')
        args = UT.load_dataset_from_zarr(
            DATA_PATH=f'{DATA_PATH}/{CLOUDNET}/xarray/',
            TOML_PATH=toml_file,
            RADAR='limrad94',
            TASK='predict',
        )
        X, y, classes, mask, ts, rg = args[0], args[1], args[3], args[8], args[12], args[13]

        X = X[:, :, 4:10, 0]  # good!
        X = X[:, :, :, np.newaxis]
        X = X.transpose(0, 3, 2, 1)
        X_test = torch.Tensor(X)
        y_test = torch.Tensor(y)
    else:
        TEST_PATH = os.path.join(
            VOODOO_PATH,
            'data_12chdp/xarray_zarr/20190801-20190801-X-12ch2pol.zarr'
        )

        print(f'Loading compressed zarr file ...... {VOODOO_PATH}')
        X_test, y_test = TM.VoodooNet.fetch_data(TEST_PATH, shuffle=False)
        xrTest = TM.VoodooNet.fetch_2Ddata(TEST_PATH)

        mask = xrTest['mask'].values.copy()
        ts, rg = xrTest['mask'].ts.copy(), xrTest['mask'].rg.copy()
        import re

        date_test = re.search('\d{8}-\d{8}', TEST_PATH)
        date_str = date_test.group(0)

        TM.VoodooNet.create_quicklook(xrTest, trained_model)

    model = TM.VoodooNet.load(trained_model)
    print(model)
    model.print_nparams()

    prediction = model.testing(X_test, batch_size=BATCH_SIZE, dev=DEVICE_test)
    prediction = prediction.to('cpu')
    values = TM.VoodooNet.new_classification(prediction, mask)


    Vclasses = VoodooXR(ts, rg)
    Vclasses.add_nD_variable('CLASS', ('ts', 'rg'), values, **{
        'colormap': 'cloudnet_target_new',
        'rg_unit': 'km',
        'var_unit': '',
        'system': 'VOODOO',
        'var_lims': [0, 10],
        'range_interval': [0, 5]
    })

    f, ax = TM.VoodooNet.create_quicklook(Vclasses['CLASS'])
    fig_name = trained_model.replace('.pt', f'-{date_str}-V.png')
    f.savefig(fig_name, dpi=250)  # , bbox_inches = 'tight')
    print(f"\nfig saved: {fig_name}")

    if task != 'inference':
        CNclasses = Vclasses.copy()
        CNclasses['CLASS'].attrs['system'] = 'CLOUDNETpy94'
        CNclasses['CLASS'].values = classes
        f, ax = TM.VoodooNet.create_quicklook(CNclasses['CLASS'])
        fig_name = trained_model.replace('.pt', f'-{date_str}-CN.png')
        f.savefig(fig_name, dpi=250)  # , bbox_inches = 'tight')
        print(f"\nfig saved: {fig_name}")

    # plot smoothed data
    from scipy.ndimage import gaussian_filter

    Nts, Nrg = mask.shape
    probabilities = TM.VoodooNet.reshape(prediction, mask, (Nts, Nrg, NCLASSES))
    probabilities_post = probabilities.copy()

    smoothed_probs = np.zeros((Nts, Nrg, NCLASSES))
    for i in range(NCLASSES):
        smoothed_probs[:, :, i] = gaussian_filter(probabilities[:, :, i], sigma=1)


    Vclasses.add_nD_variable('sCLASS', ('ts', 'rg'), values, **{})
    Vclasses['sCLASS'].attrs = Vclasses['CLASS'].attrs.copy()
    Vclasses['sCLASS'].values = np.argmax(smoothed_probs, axis=2)
    Vclasses['sCLASS'].values[mask] = 0

    f, ax = TM.VoodooNet.create_quicklook(Vclasses['sCLASS'])
    fig_name = trained_model.replace('.pt', f'-{date_str}-sV.png')
    f.savefig(fig_name, dpi=250)  # , bbox_inches = 'tight')
    print(f"\nfig saved: {fig_name}")

#######
#    # post stuff
#    smoothed_probs_post = smoothed_probs.copy()
#    all_liquid = probabilities_post[:, :, 0] + probabilities_post[:, :, 2] + probabilities_post[:, :, 5] + probabilities_post[:, :, 7]
#    liuqid_most_probable = np.max(probabilities_post, axis=2) < all_liquid
#
#
#    Vclasses.add_nD_variable('sxCLASS', ('ts', 'rg'), values, **{})
#    Vclasses['sxCLASS'].attrs = Vclasses['CLASS'].attrs.copy()
#    Vclasses['sxCLASS'].values = np.argmax(smoothed_probs_post, axis=2)
#    Vclasses['sxCLASS'].values[liuqid_most_probable] = 5
#    Vclasses['sxCLASS'].values[mask] = 0
#
#    TM.VoodooNet.create_quicklook(Vclasses['sxCLASS'], trained_model.replace('.pt', f'-{date_str}-VO0DOO_smoothedX.pt'))
#######

