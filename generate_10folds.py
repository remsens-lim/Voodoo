#!/home/sdig/anaconda3/bin/python
import glob
import logging
import os
import torch
import toml

from tqdm.auto import tqdm
from libVoodoo.Loader import tensor_from_hourly_zarr, logger
from libVoodoo.Utils import change_dir, read_cmd_line_args

if __name__ == '__main__':

    logger.setLevel(logging.CRITICAL)
    voodoo_path = os.getcwd()

    paths = toml.load('VnetSettings-1.toml')['paths']
    hourly_path = paths['hourly_path']
    save_path = paths['folds_path']

    _, agrs, kwargs = read_cmd_line_args()

    if 'fn' in kwargs:
        # create 10 folds
        ifile = kwargs['fn'] if 'fn' in kwargs else 0

        hourly_files = sorted(glob.glob(hourly_path + '*ND.zarr'))[ifile:]
        hourly_files = hourly_files[::10]
        X, y = tensor_from_hourly_zarr(hourly_files, task='train')

        # ND variables
        change_dir(save_path)
        FILE_NAME = f'10foldsall-fn{ifile}-voodoo-ND2.pt'
        torch.save({'X': X, 'y': y}, FILE_NAME)
        print(f'save :: {FILE_NAME}')


    else:
        # concatenate all folds together
        X, y = [], []
        for ifile in tqdm(range(10)):
            FILE_NAME = save_path + f'10foldsall-fn{ifile}-voodoo-ND2.pt'
            m = torch.load(FILE_NAME)
            X.append(m['X'])
            y.append(m['y'])

        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)

        FILE_NAME = save_path + f'10foldsall-fnX-voodoo-ND2.pt'
        torch.save({'X': X, 'y': y}, FILE_NAME)
        print(f'save :: {FILE_NAME}')
