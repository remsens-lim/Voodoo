import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import xarray as xr
from tqdm.auto import tqdm
from datetime import timedelta, datetime

from .Utils import ts_to_dt, lin2z, decimalhour2unix, argnearest
from .TorchModel import VoodooNet

def replace_fill_value(data, newfill):
    """
    Replaces the fill value of an spectrum container by their time and range specific mean noise level.
    Args:
        data (numpy.array) : 3D spectrum array (time, range, velocity)
        newfill (numpy.array) : 2D new fill values for 3rd dimension (velocity)

    Return:
        var (numpy.array) : spectrum with mean noise
    """

    n_ts, n_rg, _ = data.shape
    var = data.copy()
    masked = np.all(data <= 0.0, axis=2)

    for iT in range(n_ts):
        for iR in range(n_rg):
            if masked[iT, iR]:
                var[iT, iR, :] = newfill[iT, iR]
            else:
                var[iT, iR, var[iT, iR, :] <= 0.0] = newfill[iT, iR]
    return var


def open_xarray_datasets(path):
    import re
    ds = xr.open_mfdataset(path, parallel=True, decode_times=False, )
    x = re.findall("\d{8}", path)[0]
    # convert time to unix
    ds = ds.assign_coords(time = decimalhour2unix(str(x), ds['time'].values))
    ds['time'].attrs['units'] = 'UTC'
    return ds


def hyperspectralimage(ts, vhspec, hspec, msk, n_channels, new_ts):
    n_ts_new = len(new_ts) if len(new_ts) > 0 else ValueError('Needs new_time array!')
    n_ts, n_rg, n_vel = vhspec.shape
    mid = n_channels // 2

    ip_var = np.full((n_ts_new, n_rg, n_vel, n_channels, 2), fill_value=-999.0, dtype=np.float32)
    ip_msk = np.full((n_ts_new, n_rg, n_vel, n_channels), fill_value=True)

    # for iBin in range(n_vel):
    for iBin in range(n_vel):
        for iT_cn in range(n_ts_new):
            iT_rd0 = argnearest(ts, new_ts[iT_cn])

            for itmp in range(-mid, mid):
                iTdiff = itmp if iT_rd0 + itmp < n_ts else 0
                ip_var[iT_cn, :, iBin, iTdiff + mid, 0] = vhspec[iT_rd0 + iTdiff, :, iBin]
                ip_var[iT_cn, :, iBin, iTdiff + mid, 1] = hspec[iT_rd0 + iTdiff, :, iBin]
                ip_msk[iT_cn, :, iBin, iTdiff + mid] = msk[iT_rd0 + iTdiff, :, iBin]

    return ip_var, ip_msk

def VoodooPredictor(X):
    """
    Predict probability disribution over discrete set of 3 classes (dummy, CD, non-CD) .
    
    Args:
        X: list of [256, 6, 1] time spectrograms, dimensitons: (N_Doppler_bins, N_time_steps, N_range_gate)
        
        
    Return:
        predicitons: list of predictions 
    """
    import torch
    import toml

    
    # load architecture
    model_setup_file = f'model/VnetSettings.toml'
    trained_model = 'model/Vnet0x615580bf-fn1-gpu0-VN.pt'
    
    torch_settings = toml.load(os.path.join(model_setup_file))['pytorch']
    torch_settings.update({'dev': 'cpu'})
    
    # (n_samples, n_Doppler_bins, n_time_steps)
    X = X[:, :, :, np.newaxis]
    X = X.transpose(0, 3, 2, 1)
    X_test = torch.Tensor(X)

    model = VoodooNet(X_test.shape, 3, **torch_settings)
    model.load_state_dict(torch.load(trained_model, map_location=model.device)['state_dict'])

    prediction = model.predict(X_test, batch_size=256)
    prediction = prediction.to('cpu')
    return prediction
