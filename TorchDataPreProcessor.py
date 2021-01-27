#!/home/sdig/anaconda3/bin/python

import logging
import sys
import re

import numpy as np

PATH_TO_LARDA = '/home/sdig/code/larda3/larda/'
sys.path.append(PATH_TO_LARDA)
import pyLARDA.helpers as h

from generate_trainingset import VoodooXR
from libVoodoo.Utils import load_dataset_from_zarr, log_number_of_classes

logger = logging.getLogger('torchdatapreprocessor')
logger.setLevel(logging.CRITICAL)

""" /home/sdig/code/larda3/voodoo/tomls/
- `auto-trainingset-20181127-20190927.toml` —>	60/600 = 723 files
- `auto-trainingset-20181127-20190928.toml` —>	60/300 = 1469 files
- `auto-trainingset-20181128-20190927.toml` —>	60/180 = 2432 files
- `auto-trainingset-20181128-20190928.toml` —>	60/60 = 7320 files
"""
voodoo_path = '/home/sdig/code/larda3/voodoo/'
model_path = '/home/sdig/code/larda3/voodoo/HP_12chdp2.toml'
toml_file = '/home/sdig/code/larda3/voodoo/tomls/auto-trainingset-20190801-20190801.toml'
CALIBRATED_PATH = '/home/sdig/code/larda3/voodoo/data_12chdp/CLOUDNETpy94/xarray/'
CONCATINATED_PATH = '/home/sdig/code/larda3/voodoo/data_12chdp/xarray_zarr_5folds/'
TOMLS_PATH = '/home/sdig/code/larda3/voodoo/tomls/5folds_all/'


def save_torch_input(args, path, datestr=''):
    xr_ds = VoodooXR(None, None)
    # add coordinates
    xr_ds._add_coordinate({'nsamples': np.arange(args[0].shape[0])}, 'Number of training samples')
    xr_ds._add_coordinate({'nvelocity': np.arange(args[0].shape[1])}, 'Number of velocity bins')
    xr_ds._add_coordinate({'nchannels': np.arange(args[0].shape[2])}, 'Number of stacked spectra')
    xr_ds._add_coordinate({'npolarization': np.arange(args[0].shape[3])}, 'vertical(co) and horizontal(cx) polarization')
    xr_ds.add_nD_variable('features', ('nsamples', 'nvelocity', 'nchannels', 'npolarization'), args[0], **{})
    xr_ds.add_nD_variable('targets', ('nsamples'), args[1], **{})

    h.change_dir(path)
    FILE_NAME = f'{datestr}-{args[0].shape[2]}ch{args[0].shape[3]}pol.zarr'
    xr_ds.to_zarr(store=FILE_NAME, mode='w')
    logger.critical(f'save :: {FILE_NAME}')


def save_2D_validation_data(args, path, datestr=''):
    xr_ds2D = VoodooXR(args[12], args[13])
    xr_ds2D.add_nD_variable(
        'classes', ('ts', 'rg'), args[3],
        **{'colormap': 'cloudnet_target_new',
           'rg_unit': 'km',
           'var_unit': '',
           'system': 'Cloudnetpy',
           'var_lims': [0, 10]}
    )
    xr_ds2D.add_nD_variable(
        'status', ('ts', 'rg'), args[4],
        **{'colormap': 'cloudnetpy_detection_status',
           'rg_unit': 'km',
           'var_unit': '',
           'system': 'Cloudnetpy',
           'var_lims': [0, 17]}
    )
    xr_ds2D.add_nD_variable(
        'Ze', ('ts', 'rg'), args[17],
        **{'colormap': 'jet',
           'rg_unit': 'km',
           'var_unit': 'dBZ',
           'system': 'Cloudnetpy',
           'var_lims': [-50, 20]}
    )
    xr_ds2D.add_nD_variable(
        'mask', ('ts', 'rg'), args[8],
        **{'colormap': 'coolwarm',
           'rg_unit': 'km', 'var_unit': '',
           'system': 'Cloudnetpy',
           'var_lims': [0, 1]}
    )

    h.change_dir(path)
    FILE_NAME = f'{datestr}-X-{args[0].shape[2]}ch{args[0].shape[3]}pol2D.zarr'
    xr_ds2D.to_zarr(store=FILE_NAME, mode='w')
    logger.critical(f'save :: {FILE_NAME}')


if __name__ == '__main__':

    nfiles = 5
    toml_file = f'{TOMLS_PATH}/auto-trainingset-20190801-20190801.toml'
    # reprocess entire trainingset
    for ifile in range(1, nfiles):
        if nfiles > 1:
            toml_file = f'{TOMLS_PATH}/auto-trainingset-20181127-20190927-{ifile}.toml'
        print(f'\nLoad toml file: {toml_file}')

        args = load_dataset_from_zarr(
            CALIBRATED_PATH,
            toml_file,
            CLOUDNET='CLOUDNETpy94',
            RADAR='limrad94',
            TASK='train',
        )

        log_number_of_classes(args[1], text=f'\nsamples per class')

        m = re.search('\d{8}-\d{8}', toml_file).group(0)
        save_torch_input(args, CONCATINATED_PATH, datestr=f'{m}-{ifile}-allclasses-allfiles')
        try:
            save_2D_validation_data(args, CONCATINATED_PATH, datestr=m)
        except:
            print('Skip 2D data.')
