#!/home/sdig/anaconda3/bin/python
import sys
import os
import csv
sys.path.append('/home/sdig/code/larda3/larda/')
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
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu', build_lists=False) #lacros_dacapo_gpu
    #larda = pyLARDA.LARDA().connect('leipzig_gpu', build_lists=False) #lacros_dacapo_gpu


    #begin_dt = datetime.datetime(2019, 1, 4, 0, 0)
    #end_dt = begin_dt + datetime.timedelta(seconds=int(24*60*60-1))
    range_interval = np.array([0, 12000])
#    begin_dt = datetime.datetime(2019, 8, 1, 0, 0)
#    end_dt = datetime.datetime(2019, 8, 1, 9, 0)
#    range_interval = np.array([0, 6000])
#
#    begin_dt = datetime.datetime(2019, 3, 9, 0, 0)
#    end_dt = datetime.datetime(2019, 3, 9, 23, 59)
#    range_interval = np.array([0, 10000])
    #begin_dt = datetime.datetime(2021, 2, 16, 0, 0)
    #end_dt = datetime.datetime(2021, 2, 16, 23, 59)
    #range_interval = np.array([0, 10000])
    #begin_dt = datetime.datetime(2019, 3, 13, 3, 0)
    #end_dt = datetime.datetime(2019, 3, 13, 23, 59)
    #ange_interval = np.array([0, 6000])

    import csv

    filename = {
        #"Pun_larda3_old": '../cloud_collections/cloud_collection_lacros_dacapo_all_w_dl.csv.bak-20201013',
        "Pun_larda3": f'{VOODOO_PATH}/data/cloud_collection_lacros_dacapo_LIM.csv',
        #"Lim_larda3": '../cloud_collections/cloud_collection_lacros_cycare_all_w_dl.csv',
        #"Lei_larda3": '../cloud_collections/cloud_collection_lacros_leipzig_all.csv',
    }

    clouds_new = load_data(filename['Pun_larda3'])

    from numpy import genfromtxt

    with open(filename['Pun_larda3'], 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)

    my_data = genfromtxt(filename['Pun_larda3'], delimiter=';')[1:, :]
    n_cases, n_variables = my_data.shape
    var_list = ['Z', 'VEL', 'width', 'beta', 'LWP']

    for icase in range(7379, 7414):
        try:
            begin_dt = datetime.datetime.utcfromtimestamp(my_data[icase, 2])
            end_dt = datetime.datetime.utcfromtimestamp(my_data[icase, 25])
            range_interval = np.array([my_data[icase, 3], my_data[icase, 6]])

            var_dict = {var: larda.read("CLOUDNETpy94", var, [begin_dt, end_dt], range_interval) for var in var_list}
            T = larda.read("CLOUDNETpy94", "T", [begin_dt-datetime.timedelta(seconds=3600), end_dt], range_interval)


            contour_T = {
                'data': pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"}),
                'levels': np.array([-40, -25, -15, 0])
            }
            dt_list = [h.ts_to_dt(ts) for ts in var_dict['Z']['ts']]

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
                ax[i].set(xticklabels=[], xlabel='')

            fig.subplots_adjust(bottom=0.03, right=1.05, top=0.975, left=0.05, hspace=0.175)
            fig_name = f'{begin_dt:%Y%m%d-%H%M}-Cloudnet-Categorize.png'
            h.change_dir(f'{QUICKLOOK_PATH}/')
            fig.savefig(fig_name, dpi=200)

            print(f' saved  {begin_dt:%Y%m%d-%H%M}-Cloudnet-Categorize.png')
            dummy = 5
        except:
            print(f'skip {begin_dt}')
