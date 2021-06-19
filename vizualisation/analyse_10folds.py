#!/home/sdig/anaconda3/bin/python

import logging
import sys, os
import re
from tqdm.auto import tqdm
import xarray as xr
from datetime import datetime
import numpy as np
import scipy.stats

import traceback
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from libVoodoo.Plot import load_xy_style, load_cbar_style
from libVoodoo.Utils import log_number_of_classes, change_dir, cloudnetpy_classes, load_training_mask
from libVoodoo.Loader import VoodooXR, validation_fold_to_zarr, logger, load_case_file, load_case_list
import libVoodoo.TorchModel as TM
from libVoodoo.Utils import traceback_error, read_cmd_line_args

from libVoodoo.Loader import preproc_ini

sys.path.append(preproc_ini['larda']['path'])
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import pyLARDA.VIS_Colormaps as cmaps

""" /home/sdig/code/larda3/voodoo/tomls/
- `auto-trainingset-20181127-20190927.toml` —>	60/600 = 723 files
- `auto-trainingset-20181127-20190928.toml` —>	60/300 = 1469 files
- `auto-trainingset-20181128-20190927.toml` —>	60/180 = 2432 files
- `auto-trainingset-20181128-20190928.toml` —>	60/60 = 7320 files
"""

if __name__ == '__main__':

    logger.setLevel(logging.CRITICAL)
    VOODOO_PATH = os.getcwd()
    mode = '10folds_all'
    ANN_INI_FILE = 'HP_12chdp2.toml'
    task = 'train'
    method_name, args, kwargs = read_cmd_line_args(sys.argv)
    ifold = kwargs['fn'] if 'fn' in kwargs else 0


    DATA_PATH = f'{VOODOO_PATH}/data/Vnet_6ch_noliqext/'
    TOMLS_PATH = f'{VOODOO_PATH}/tomls/{mode}/'

    nfiles = 10
    n_rg = 292  # 358, 292
    n_cl = 14
    plot_kwargs = {'var_lims': [0, 10], 'fontweight': 'normal', 'rg_converter': True}
    plot = False

    # reprocess entire trainingset
    # for ifold in range(nfiles):
    N_NOT_AVAILABLE = 0
    toml_file = f'{TOMLS_PATH}/auto-trainingset-20181127-20190927-{ifold}.toml'
    m = re.search('\d{8}-\d{8}', toml_file).group(0)
    print(f'\nLoad toml file: {toml_file}')

    data_chunk_heads = [chunk for chunk in load_case_file(toml_file).keys()]
    n_hour_files = len(data_chunk_heads)

    xr_ds = VoodooXR(None, None)
    # add coordinates
    xr_ds.add_coordinate({'ts': np.arange(n_hour_files)}, 'Number of time steps')
    xr_ds.add_coordinate({'rg': np.arange(n_rg)}, 'Number of range bins')
    xr_ds.add_coordinate({'cl': np.arange(n_cl)}, 'Number of classes bins')
    xr_ds.add_nD_variable('dist', ('ts', 'rg', 'cl'), np.full((n_hour_files, n_rg, n_cl), 0, dtype=int), **{})

    for icase, case_str in tqdm(enumerate(data_chunk_heads), total=n_hour_files, unit='files', ncols=100):

        case = load_case_list(toml_file, case_str)
        TIME_SPAN = [datetime.strptime(t, '%Y%m%d-%H%M') for t in case['time_interval']]
        dt_str = f'{TIME_SPAN[0]:%Y%m%d_%H%M}-{TIME_SPAN[1]:%H%M}'
        zarr_file = f'{DATA_PATH}/hourly/{dt_str}_limrad94-CLOUDNETpy94-2D.zarr'

        # check if a mat files is available
        try:
            with xr.open_zarr(zarr_file) as zarr_data:

                _classes = zarr_data['CLASS'].values.copy()
                _status = zarr_data['detection_status'].values
                _dt = [datetime.utcfromtimestamp(_ts) for _ts in zarr_data['CLASS']['ts'].values]

                if plot:
                    fig_name = f'{DATA_PATH}/{mode}/{dt_str}-fn-{ifold}-cn{icase}-CloudnetCLASS-QL.png'
                    fig, ax = plt.subplots(nrows=3, figsize=(14, 10))
                    fig, ax[0] = tr.plot_timeheight2(zarr_data['CLASS'], fig=fig, ax=ax[0], title=f"target class quicklook {dt_str}", **plot_kwargs)
                    fig, ax[1] = tr.plot_timeheight2(zarr_data['detection_status'], fig=fig, ax=ax[1], title=f"target status quicklook {dt_str}", **plot_kwargs)
                for i in range(11):
                    xr_ds['dist'][icase, :, i] = np.sum((_classes == i), axis=0)
                droplets = (_classes == 1) + (_classes == 3) + (_classes == 5) + (_classes == 7)
                others = (_classes == 2) + (_classes == 4) + (_classes == 6) + (_classes == 8) + (_classes == 9) + (_classes == 10)
                xr_ds['dist'][icase, :, -4] = np.sum(droplets, axis=0)
                xr_ds['dist'][icase, :, -3] = np.sum(others, axis=0)
                _mask = load_training_mask(_classes, _status, 'CLOUDNETpy94')
                _classes[_mask] = 0
                droplets = (_classes == 1) + (_classes == 3) + (_classes == 5) + (_classes == 7)
                others = (_classes == 2) + (_classes == 4) + (_classes == 6) + (_classes == 8) + (_classes == 9) + (_classes == 10)
                xr_ds['dist'][icase, :, -2] = np.sum(droplets, axis=0)
                xr_ds['dist'][icase, :, -1] = np.sum(others, axis=0)
                # plot both quicklooks
                if plot:
                    zarr_data['newCLASS'] = zarr_data['CLASS'].copy()
                    zarr_data['newCLASS'].values = _classes
                    fig, ax[2] = tr.plot_timeheight2(
                        zarr_data['newCLASS'], fig=fig, ax=ax[2], title=f"target class quicklook {dt_str} only trainingsamples", **plot_kwargs)
                    for i in range(2):
                        ax[i].set(xticklabels=[], xlabel='')
                    fig.subplots_adjust(bottom=0.1, right=1, top=0.95, left=0.05, hspace=0.175)
                    fig.savefig(fig_name, dpi=200)

        except Exception as e:
            logger.warning(f"Unexpected error: Check folder: {zarr_file}.zarr, n_Failed = {N_NOT_AVAILABLE}")
            N_NOT_AVAILABLE += 1
            # traceback_error(icase)
            continue

    xr_ds.to_netcdf(f'{DATA_PATH}/analyser-{task}-fn{ifold}.nc')

    dummy = 5
