#!/home/sdig/anaconda3/bin/python
import logging
import sys, os
import re

from libVoodoo.Utils import log_number_of_classes, change_dir, read_cmd_line_args
from libVoodoo.Loader import dataset_from_zarr_new, validation_fold_to_zarr, logger


CONCATINATED_PATH = '/media/sdig/Voodoo_Data/'

if __name__ == '__main__':

    logger.setLevel(logging.CRITICAL)
    voodoo_path = os.getcwd()
    scenario = '10folds_all'
    ANN_INI_FILE = 'VnetSettings-1.toml'
    task = '' #'train'

    _, agrs, kwargs = read_cmd_line_args()
    ifile = kwargs['fn'] if 'fn' in kwargs else 6
    task = kwargs['task'] if 'task' in kwargs else ''

    CALIBRATED_PATH = f'{voodoo_path}/data/Vnet_6ch_noliqext/hourly/'
    TOMLS_PATH = f'{voodoo_path}/tomls/'


    # reprocess entire trainingset
    if task == 'validation':
        toml_file = f'{TOMLS_PATH}/{scenario}/validation_trainingset.toml'
        m = 'validation_testingset'
    elif task == 'train':
        toml_file = f'{TOMLS_PATH}/auto-trainingset-20190801-20190801.toml'
        m = 'debugger_trainingset'
    else:
        #toml_file = f'{TOMLS_PATH}/{scenario}/auto-trainingset-20181127-20190927-{ifile}.toml' # PA
        toml_file = f'{TOMLS_PATH}/{scenario}/auto-trainingset-20201216-20211001-{ifile}.toml' # LIM
        m = re.search('\d{8}-\d{8}', toml_file).group(0)
    print(f'\nLoad toml file: {toml_file}')

    change_dir(f'{CONCATINATED_PATH}/{scenario}/')

    ND_ds, twoD_ds = dataset_from_zarr_new(
        CALIBRATED_PATH,
        toml_file,
        CLOUDNET='CLOUDNETpy94',
        RADAR='limrad94',
        TASK=task,
        PLOT=False,
    )

    numbers = log_number_of_classes(ND_ds['targets'].values, text=f'\nsamples per class')
    print(numbers)

    # ND variables
    FILE_NAME = f'{m}-{ifile}-{scenario}-ND.zarr'
    ND_ds.to_zarr(store=FILE_NAME, mode='w')
    print(f'save :: {FILE_NAME}')
#    if task != 'train':
#        # 2D variables
#        FILE_NAME = f'{m}-{ifile}-{scenario}-2D.zarr'
#        twoD_ds.to_zarr(store=FILE_NAME, mode='w')
#        print(f'save :: {FILE_NAME}')
