#!/home/sdig/anaconda3/bin/python

import logging
import os

import matplotlib.pyplot   as plt
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from libVoodoo.Loader import logger
from libVoodoo.Utils import cloudnetpy_classes_n

""" /home/sdig/code/larda3/voodoo/tomls/
- `auto-trainingset-20181127-20190927.toml` —>	60/600 = 723 files
- `auto-trainingset-20181127-20190928.toml` —>	60/300 = 1469 files
- `auto-trainingset-20181128-20190927.toml` —>	60/180 = 2432 files
- `auto-trainingset-20181128-20190928.toml` —>	60/60 = 7320 files
"""

if __name__ == '__main__':

    logger.setLevel(logging.CRITICAL)
    voodoo_path = os.getcwd()
    mode = '10folds_all'
    ANN_INI_FILE = 'HP_12chdp2.toml'
    task = 'train'

    CONCATINATED_PATH = f'{voodoo_path}/data/Vnet_6ch_noliqext/'
    TOMLS_PATH = f'{voodoo_path}/tomls/{mode}/'

    nfiles = 10
    n_cl = 14
    rot = 45


    x_lims = np.zeros((nfiles, n_cl))
    # reprocess entire trainingset
    fig, ax = plt.subplots(nrows=nfiles, ncols=n_cl, figsize=(18, 16))
    for ifold in tqdm(range(nfiles), ncols=100):
        N_NOT_AVAILABLE = 0
        nc_file = f'{CONCATINATED_PATH}/{mode}/analyser-train-fn{ifold}.nc'

        xr_ds = xr.open_mfdataset(nc_file)
        dist = xr_ds['dist'].values.copy()

        for iclass in range(n_cl):
            sum = np.sum(dist[:, :, iclass], axis=0)
            ax[ifold, iclass].barh(xr_ds['rg'].values, sum, height=1.5)
            x_lims[ifold, iclass] = np.max(sum)
            ax[ifold, iclass].text(0.95 * x_lims[ifold, iclass], 250.0,
                                   f'{np.sum(sum, dtype=int):_}', color='black', fontsize=8, ha='right',
                                   bbox={'facecolor': 'white', 'alpha': 0.75, 'pad': 2})

            if ifold < nfiles - 1 and iclass > 0:
                ax[ifold, iclass].set(xticklabels=[], yticklabels=[], xlabel='', ylabel='')

        sums = np.sum(dist[:, :, -2:], axis=(0, 1))
        ratio = np.sum(sums[0])/np.sum(sums[1])
        print(f'valid samples = {np.sum(sums, dtype=int)} :: {np.sum(sums[0])}/{np.sum(sums[1])} = {ratio:.4f} :: {1/ratio:.4f}')


    for ifold in range(nfiles):
        ax[ifold, 0].set_ylabel(f'fn{ifold}')

    # titles for Cloudnet classes
    for iclass in range(n_cl):
        if iclass < 11:
            ax[0, iclass].set_title(cloudnetpy_classes_n[iclass], rotation=rot)
        if iclass > 0:
            ax[-1, iclass].set(yticklabels=[], ylabel='')

        for ifold in range(nfiles):
            ax[ifold, iclass].set(xlim=(0, np.max(x_lims[:, iclass])), ylim=(-5, 300))

    # titles for droplet/non-droplet pltos
    ax[0, -4].set_title('CD-group\n(all)', rotation=rot)
    ax[0, -3].set_title('non-CD-group\n(all)', rotation=rot)
    ax[0, -2].set_title('CD-group\n(valid)', rotation=rot)
    ax[0, -1].set_title('non-CD-group\n(valid)', rotation=rot)

    fig.subplots_adjust(bottom=0.05, right=0.95, top=0.90, left=0.05)
    fig.savefig(f'{CONCATINATED_PATH}analyser.png', dpi=200)
    print(f'saved to: {CONCATINATED_PATH}analyser.png')

    dummy = 5
