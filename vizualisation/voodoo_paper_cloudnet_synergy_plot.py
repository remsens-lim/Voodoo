#!/home/sdig/anaconda3/bin/python
import sys
import os
import csv
sys.path.append('/home/sdig/code/larda3/larda/')
sys.path.append('/home/sdig/code/Voodoo/')
import pyLARDA
import numpy as np
import pyLARDA.helpers as h
import datetime
# optionally configure the logging
# StreamHandler will print to console
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

import libVoodoo.Utils as UT

log = logging.getLogger('pyLARDA')
log.setLevel(logging.CRITICAL)
log.addHandler(logging.StreamHandler())

VOODOO_PATH = os.getcwd()

QUICKLOOK_PATH = f'{VOODOO_PATH}/plots/sniffer_cases/'

def toC(datalist):
    return datalist[0]['var'] - 273.15, datalist[0]['mask']


def load_data(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        for row in reader:
            # data.append(row)
            comp = row[2].replace('_', '')
            #if row[1][:14] == comp:
            data.append({k: v for k, v in zip(header, row)})
            #else:
            #    print('corrupted row ', row)
            #    break

    return data


if __name__ == '__main__':
    # init larda
    larda = pyLARDA.LARDA().connect('leipzig_gpu', build_lists=True) #lacros_dacapo_gpu
    #larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=False) #lacros_dacapo_gpu

    _, agrs, kwargs = UT.read_cmd_line_args()

    begin_dt = datetime.datetime.strptime(str(kwargs['time'])+'0000', '%Y%m%d%H%M')
    end_dt = datetime.datetime.strptime(str(kwargs['time'])+'1500', '%Y%m%d%H%M')
    #end_dt = begin_dt + datetime.timedelta(seconds=int(24*60*60-1))
    range_interval = np.array([0, 8000])

    var_list = ['Z', 'VEL', 'width', 'beta', 'LWP']
    #var_list = ['Z', 'VEL', 'width', 'beta', 'LWP']
    var_dict = {var: larda.read("CLOUDNETpy94", var, [begin_dt, end_dt], range_interval) for var in var_list}


    T = larda.read("CLOUDNETpy94", "T", [begin_dt-datetime.timedelta(seconds=3600), end_dt], range_interval)
    contour_T = {
        'data': pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"}),
        'levels': np.array([-40, -25, -15, 0])
    }
    dt_list = [h.ts_to_dt(ts) for ts in var_dict['Z']['ts']]

    xarr = UT.container_to_xarray(T, QUICKLOOK_PATH+ 'test.nc')

    h.change_dir(QUICKLOOK_PATH)

    fig, ax = plt.subplots(nrows=len(var_list), figsize=(12, len(var_list)*4))
    for i, (key, var) in enumerate(var_dict.items()):
        var['var'] = np.ma.masked_greater_equal(var['var'], 9.0e16)
        if key != 'LWP':
            fig, ax[i] = pyLARDA.Transformations.plot_timeheight2(
                var, fig=fig, ax=ax[i],
                range_interval=range_interval/1000.,
                rg_converter=True,
                contour=contour_T,
                title=f"CloudnetPy {key} [{var['var_unit']}]",
                var_converter='log' if key == 'beta' else None,
                label=' '
            )
        else:
            fig, ax[i] = pyLARDA.Transformations.plot_timeseries2(
                var, fig=fig, ax=ax[i],
                title=f"CloudnetPy {key} [{var['var_unit']}]",
            )
            ax[i].bar(dt_list, var['var'], width=0.001)
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="16.25%", pad=0.2)
            cax.axis('off')
            fig.add_axes(cax)
    for i in range(len(var_list) - 1):
        ax[i].set(xlabel='')


    fig.subplots_adjust(bottom=0.03, right=1.05, top=0.975, left=0.075, hspace=0.175)
    fig_name = f'{begin_dt:%Y%m%d-%H%M}-Cloudnet-Categorize.png'
    fig.savefig(fig_name, dpi=200)
    print(f' saved  {begin_dt:%Y%m%d-%H%M}-Cloudnet-Categorize.png')
    dummy = 5