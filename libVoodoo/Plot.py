"""
This module contains additional plotting routines used for displaying quicklooks of the ANN input and output, also histories and histograms.
"""
import datetime
import logging
import sys
import time
from itertools import product

import matplotlib
import matplotlib.pyplot   as plt

plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np
from matplotlib import ticker

from .Loader import preproc_ini

sys.path.append(preproc_ini['larda']['path'])
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.StreamHandler())

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2021, The Voodoo Project"
__credits__ = ["Willi Schimmel"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

_FIG_SIZE = [12, 7]
_DPI = 450
_FONT_SIZE = 14
_FONT_WEIGHT = 'normal'


def create_quicklook(da):
    f, ax = plt.subplots(nrows=1, figsize=(14, 5.7))
    f, ax = tr.plot_timeheight2(da, fig=f, ax=ax)
    return f, ax


def create_quicklook_ts(da):
    f, ax = plt.subplots(nrows=1, figsize=(14, 5.7))
    f, ax = tr.plot_timeseries(da, fig=f, ax=ax)
    return f, ax


def plot_single_spectrogram(ZSpec, ts, rg, **font_settings):
    import matplotlib.ticker as plticker
    Z = ZSpec.sel(ts=ts, rg=rg)

    Z = np.squeeze(Z.values)
    X, Y = np.meshgrid(
        np.linspace(-Z.shape[1] // 2, Z.shape[1] // 2, num=Z.shape[1]),
        np.linspace(0, Z.shape[0], num=Z.shape[0])
    )

    if 'fig' in font_settings and 'ax' in font_settings:
        fig, ax = font_settings['fig'], font_settings['ax']
    else:
        fig, ax = plt.subplots(1, figsize=(7, 7))

    surf = ax.contourf(X, Y, Z, cmap='coolwarm', linewidth=0.5, linestyle=':', antialiased=False)

    ax.set(xlabel=r"Doppler velocity bins [-]", ylabel="signal normalized")
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=32))
    ax.grid(which='both')

    return fig, ax


def create_acc_loss_graph(stats):
    fig = plt.figure(figsize=(15, 12))
    names = [
        'TP',
        'TN',
        'FP',
        'FN',
        'precision',
        # 'npv',
        'recall',
        # 'specificity',
        'accuracy',
        'F1-score',
        # 'Jaccard-index',
    ]
    ax1 = plt.subplot2grid((3, 1), (0, 0))
    ax2 = plt.subplot2grid((3, 1), (1, 0), sharex=ax1)
    ax3 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)
    train_stats, val_stats = [], []
    train_loss, val_loss = [], []
    for istat in stats:
        train_stats.append(istat[0][0]['array'][4:])
        train_loss.append(istat[0][1])
        val_stats.append(istat[1][4:])
        val_loss.append(istat[1][-1])
    train_stats = np.array(train_stats)
    val_stats = np.array(val_stats)
    for i, iline in enumerate(names[4:]):
        ax1.plot(train_stats[:, i], label=f"{iline}")
        ax2.plot(val_stats[:, i], label=f"val_{iline}")
    ax3.plot(train_loss, label=f"loss")
    ax3.plot(val_loss, label=f"val_loss")
    ax1.legend(loc=2, bbox_to_anchor=(-.21, 1))
    ax2.legend(loc=2, bbox_to_anchor=(-.21, 1))
    ax3.legend(loc=2, bbox_to_anchor=(-.21, 1))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set(ylim=[0.0, 1.0])
    ax2.set(ylim=[0.0, 1.0])
    ax3.set(ylim=[0.0, 1.6])
    load_xy_style(ax1, xlabel='training epochs', ylabel='accuracy [1]', fs=10)
    load_xy_style(ax2, xlabel='validation epochs', ylabel='accuracy [1]', fs=10)
    load_xy_style(ax3, xlabel='validation epochs', ylabel='loss [-]', fs=10)
    fig.subplots_adjust(left=0.175, right=0.95, top=0.9, bottom=0.1)

    return fig, np.array([ax1, ax2, ax3], dtype=object)


# Some adjustments to the axis labels, ticks and fonts
def load_xy_style(axis, xlabel='Time [UTC]', ylabel='Height [m]', fs=10):
    """
    Method that alters the apperance of labels on the x and y axis in place.

    Note:
        If xlabel == 'Time [UTC]', the x axis set to major
        ticks every 3 hours and minor ticks every 30 minutes.

    Args:
        ax (matplotlib.axis) :: axis that gets adjusted
        **xlabel (string) :: name of the x axis label
        **ylabel (string) :: name of the y axis label

    """

    axis.set_xlabel(xlabel, fontweight='normal', fontsize=fs)
    axis.set_ylabel(ylabel, fontweight='normal', fontsize=fs)
    if xlabel == 'Time [UTC]':
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        axis.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=2))
        axis.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=[0, 15, 30, 45]))
        axis.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(500))
    axis.tick_params(axis='both', which='major', top=True, right=True, labelsize=fs, width=3, length=4)
    axis.tick_params(axis='both', which='minor', top=True, right=True, width=2, length=3)
    return axis


def load_cbar_style(cbar, cbar_label=''):
    """
    Method that alters the apperance of labels on the color bar axis in place.

    Args:
        ax (matplotlib.axis) :: axis that gets adjusted
        **cbar_label (string) :: name of the cbar axis label, Defaults to empty string.

    """
    cbar.ax.set_ylabel(cbar_label, fontweight='normal', fontsize=_FONT_SIZE)
    cbar.ax.tick_params(axis='both', which='major', labelsize=_FONT_SIZE, width=2, length=4)


def featureql(xr_ds, xr_ds2D, indices, **kwargs):
    """

    Args:
        xr_ds: contains xarray DataArrays ZSpec, probabilities,
                with dimension (time, range, velocity) and (time, range, classes)

    Returns:

    """

    N = 4
    if 'fig' in kwargs and 'ax' in kwargs:
        fig, ax = kwargs['fig'], kwargs['ax']
    else:
        fig, ax = plt.subplots(nrows=N, ncols=N, figsize=(16, 16))  # , subplot_kw={'projection': '3d'})

    font_settings = {'fig': fig}

    icnt = 0
    for i, j in product(range(N), range(N)):
        its, irg = indices[icnt, 0], indices[icnt, 1]
        ts = xr_ds.ZSpec.ts.values[its]
        dt2 = datetime.datetime.utcfromtimestamp(ts)
        rg = xr_ds.ZSpec.rg.values[irg]

        fig, ax[i, j] = plot_single_spectrogram(xr_ds.ZSpec, ts, rg, ax=ax[i, j], **font_settings)

        load_xy_style(ax[i, j], xlabel='Doppler velocity bins [-]', ylabel='Number of time steps')
        ax[i, j].set_title(f'({icnt}) {dt2:%H:%M:%S} at {rg:6.3f} [m]', fontsize=12)
        if j > 0: ax[i, j].set_ylabel('')
        if i < N - 1: ax[i, j].set_xlabel('')

        pICE = xr_ds.PROBDIST[its, irg, 4].values
        pPRECIP = xr_ds.PROBDIST[its, irg, 2].values
        pMIXED = xr_ds.PROBDIST[its, irg, 5].values
        pDROPS = xr_ds.PROBDIST[its, irg, 1].values
        CN_class = xr_ds["CLOUDNET_CLASS"][its, irg].values
        VD_class = xr_ds["CLASS"][its, irg].values

        text_dict = {'backgroundcolor': 'white', 'fontsize': 8}
        ax[i, j].text(-120, 5.5, s=f'MXD={pMIXED:.3f}', **text_dict)
        ax[i, j].text(-120, 4.75, s=f'ICE={pICE:.3f}', **text_dict)
        ax[i, j].text(-120, 4.0, s=f'PRC={pPRECIP:.3f}', **text_dict)
        ax[i, j].text(-120, 3.25, s=f'DRP={pDROPS:.3f}', **text_dict)
        ax[i, j].text(70, 5.5, f'cCLASS={CN_class}', **text_dict)
        ax[i, j].text(70, 4.75, f'vCLASS={VD_class}', **text_dict)
        icnt += 1

    # print(f'PUNTA AREAS {dt_list[0]:%A %d. %B %Y}')

    return fig, ax


def grid_plt(xr_ds, xr_ds2D, indices):
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 16))  # , constrained_layout=True)
    ncols = 4

    gs = GridSpec(ncols + 2, ncols, left=0.05, right=0.98, hspace=0.25, wspace=0.1)
    ax_class = fig.add_subplot(gs[:2, :])

    N = 4
    dots_ts = xr_ds.dt.values[indices[:, 0]]
    dots_rg = xr_ds.rg.values[indices[:, 1]] / 1000.

    ax_class.scatter(dots_ts, dots_rg, s=42, c='r', edgecolors='white')
    fig, ax_class = tr.plot_timeheight2(xr_ds['CLASS'], fig=fig, ax=ax_class, label=False, rg_converter=True)
    for i in range(N * N):
        x, y = dots_ts[i], xr_ds.rg.values[indices[i, 1]] / 1000.
        ax_class.text(x, y + 0.1, f'{i}', fontsize=12)

    ax_class.set_xlabel('')

    ax_feat = np.zeros((ncols, ncols), dtype=object)
    for i, j in product(range(ncols), range(ncols)):
        ax_feat[i, j] = fig.add_subplot(gs[i + 2, j])

    fig, ax_feat = featureql(xr_ds, xr_ds2D, indices, fig=fig, ax=ax_feat)

    return fig, [ax_class, ax_feat]


def print_elapsed_time(t0, string='time = '):
    logger.info(f'{string}{datetime.timedelta(seconds=int(time.time() - t0))} [min:sec]')


def save_figure(fig, **kwargs):
    """
    Creates at folder and saves a matplotlib figure.

    Args:
        fig (matplotlib figure): figure to save as png

    Keyword Args:
        dpi (int): dots per inch
        name (string): name of the png
        path (string): path where the png is stored

    Returns:    0

    """
    dotsperinch = kwargs['dpi'] if 'dpi' in kwargs else 200
    name = kwargs['name'] if 'name' in kwargs else 'no-name.png'
    path = kwargs['path'] if 'path' in kwargs else ''
    if len(path) > 0: h.change_dir(path)
    fig.savefig(name, dpi=dotsperinch)
    logger.info(f'Save figure :: {name}')
    return 0


def plot_bar_data(fig, ax, time, data, mask_value=0.0, font_size=_FONT_SIZE):
    """Plots 1D variable as bar plot.
    Args:
        ax (obj): Axes object.
        data (ndarray): 1D data array.
        time (ndarray): 1D time array.
    """

    plt.rcParams.update({'font.size': font_size})
    data = np.ma.masked_less_equal(data, mask_value)
    pos0 = ax.get_position()
    ax_new = fig.add_axes([0., 0., 1., 4.5])
    ax_new.bar(time, data.filled(0), width=1 / 1200, align='center', alpha=0.5, color='royalblue')
    ax_new.set(
        ylim=[-10, 200], ylabel='mwr-lwp [g m-2]',
        xlim=[time[0], time[-1]],
        position=[pos0.x0, pos0.height + pos0.height * 0.125, pos0.width, pos0.height / 2],
        # fontweight='semibold'
    )
    ax_new.tick_params(labelbottom=False, labeltop=True)
    ax_new.grid(True)

    time_extend = datetime.timedelta(seconds=(time[-1].values - time[0].values).astype(np.float64) / 10 ** 9)
    ax_new = tr.set_xticks_and_xlabels(ax_new, time_extend)
    ax_new.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax_new.tick_params(axis='both', which='both', right=True, top=True)
    ax_new.tick_params(axis='both', which='major', labelsize=_FONT_SIZE, width=3, length=5.5)
    ax_new.tick_params(axis='both', which='minor', width=2, length=3)
    return ax_new


def plot_ll_thichkness(ax, t, l1, l2, font_size=_FONT_SIZE):
    y_lim = [-0.12, 2.5]

    plt.rcParams.update({'font.size': font_size})
    ax1 = ax.twinx()
    ax1.plot(t, l1 / 1000., color='#E64A23', alpha=0.75, label='VOODOO (V)')
    ax1.set_ylim(y_lim)
    ax1.plot(t, l2 / 1000., color='navy', alpha=0.75, label='Cloudnet (C)')
    # ax1.plot(dt_list, sum_ll_thickness[nn_varname], color='red', linestyle='-', alpha=0.75, label=nn_varname)

    ax1.set(ylim=y_lim, ylabel='liquid layer\nthickness [km]')
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1.tick_params(axis='both', which='both', right=True)
    ax1.tick_params(axis='both', which='major', labelsize=font_size, width=3, length=5.5)
    ax1.tick_params(axis='both', which='minor', width=2, length=3)
    time_extend = datetime.timedelta(seconds=(t[-1].values - t[0].values).astype(np.float64) / 10 ** 9)
    ax1 = tr.set_xticks_and_xlabels(ax1, time_extend)
    ax1.legend(loc='upper left')

    return ax1


def add_lwp_to_classification(fig, ax, prediction, lwp, llt_v, llt_c, font_size=_FONT_SIZE):
    # add the lwp ontop
    dt_lwp = [h.ts_to_dt(ts) for ts in lwp['ts']]

    ax.set_xlim([h.ts_to_dt(lwp['ts'][0]), h.ts_to_dt(lwp['ts'][-1])])
    lwp_ax = plot_bar_data(
        fig, ax,
        dt_lwp,
        lwp['var'],
        font_size=_FONT_SIZE
    )

    plot_ll_thichkness(
        lwp_ax,
        [h.ts_to_dt(ts) for ts in prediction['ts']],
        llt_v,
        llt_c,
        font_size=font_size
    )

    return fig, ax


def add_lwp_to_classification2(fig, ax, lwp, lwp_ad, lwp_ad2, font_size=_FONT_SIZE):
    # add the lwp ontop
    dt_lwp = [h.ts_to_dt(ts) for ts in lwp['ts']]

    ax.set_xlim([h.ts_to_dt(lwp['ts'][0]), h.ts_to_dt(lwp['ts'][-1])])
    lwp_ax = plot_bar_data(
        fig, ax,
        dt_lwp,
        lwp['var'],
        font_size=_FONT_SIZE
    )

    lwp_ax.plot(dt_lwp, lwp_ad, label='adiabatic lwp')
    lwp_ax.plot(dt_lwp, lwp_ad2, label='adiabatic lwp cn')
    lwp_ax.set_ylim([-10, 250])

    ax.legend(loc='best')

    return fig, ax
