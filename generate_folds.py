#!/home/sdig/anaconda3/bin/python

import logging
import sys, os
import re

from libVoodoo.Utils import load_dataset_from_zarr, log_number_of_classes, change_dir
from libVoodoo.Loader import spectra_fold_to_zarr, validation_fold_to_zarr, logger

""" /home/sdig/code/larda3/voodoo/tomls/
- `auto-trainingset-20181127-20190927.toml` —>	60/600 = 723 files
- `auto-trainingset-20181127-20190928.toml` —>	60/300 = 1469 files
- `auto-trainingset-20181128-20190927.toml` —>	60/180 = 2432 files
- `auto-trainingset-20181128-20190928.toml` —>	60/60 = 7320 files
"""

if __name__ == '__main__':

    logger.setLevel(logging.CRITICAL)
    voodoo_path = os.getcwd()
    scenario = '10folds_all'
    ANN_INI_FILE = 'HP_12chdp2.toml'
    task = 'train'

    CALIBRATED_PATH = f'{voodoo_path}/data/{ANN_INI_FILE[:-5]}/hourly/noSL/'
    CONCATINATED_PATH = f'{voodoo_path}/data/{ANN_INI_FILE[:-5]}/{scenario}/'
    TOMLS_PATH = f'{voodoo_path}/tomls/{scenario}/'

    nfiles = 10
    change_dir(CONCATINATED_PATH)

    # reprocess entire trainingset
    for ifold in range(nfiles):
        toml_file = f'{TOMLS_PATH}/auto-trainingset-20181127-20190927-{ifold}.toml'
        m = re.search('\d{8}-\d{8}', toml_file).group(0)
        print(f'\nLoad toml file: {toml_file}')

        args = load_dataset_from_zarr(
            CALIBRATED_PATH,
            toml_file,
            CLOUDNET='CLOUDNETpy94',
            RADAR='limrad94',
            TASK=task,
        )

        log_number_of_classes(args[1], text=f'\nsamples per class')

        # ND variables
        xr_ds = spectra_fold_to_zarr(args)
        FILE_NAME = f'{m}-{ifold}-{scenario}-ND.zarr'
        xr_ds.to_zarr(store=FILE_NAME, mode='w')
        print(f'save :: {FILE_NAME}')

        if task == 'predict':
            # 2D variables
            xr_ds2D = validation_fold_to_zarr(args)
            FILE_NAME = f'{m}-{ifold}-{scenario}-2D.zarr'
            xr_ds2D.to_zarr(store=FILE_NAME, mode='w')
            print(f'save :: {FILE_NAME}')
