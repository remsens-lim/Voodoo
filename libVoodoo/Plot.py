"""
This module contains additional plotting routines used for displaying quicklooks of the ANN input and output, also histories and histograms.
"""
import sys

import pywt
import time
import datetime
import numpy as np
import logging

from scipy.fftpack import fft

import matplotlib
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

import itertools

import pyLARDA
import pyLARDA.helpers as h
import pyLARDA.Transformations as tr
import pyLARDA.VIS_Colormaps as VIS_Colormaps

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.StreamHandler())

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"

FIG_SIZE_ = [12, 7]
DPI_ = 200
_FONT_SIZE = 12
_FONT_WEIGHT = 'normal'

def History(history):
    """This routine generates a history quicklook for a trained Tensoflow ANN model. The figure contains the in-sample-loss/accuracy and
    out-of-sample-loss/accuracy.

    Args:
        - history (tensorflow object) : contains the training history

    Returns:
        - fig (matplotlib figure) : figure
        - ax (matplotlib axis) : axis
    """
    import pandas as pd
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    figure, axis = plt.subplots(nrows=1, ncols=1)
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Loss/Accuracy')
    for loss in history.history:
        axis.plot(hist['epoch'], hist[loss], label=f'{loss}')
    axis.legend()

    return figure, axis

def LearningRate(hist):
    nb_epoch = len(hist.history['loss'])
    learning_rate = hist.history['lr']
    xc = range(nb_epoch)
    figure, axis = plt.figure(3, figsize=(7, 5))
    axis.plot(xc, learning_rate)
    axis.xlabel('num of Epochs')
    axis.ylabel('learning rate')
    axis.title('Learning rate')
    axis.grid(True)
    axis.style.use(['seaborn-ticks'])
    return figure, axis

def Quicklooks(RPG_moments, polly_var, begin_dt, end_dt, **kwargs):
    """This routine generates all quicklooks for a given training period.

    Args:
        - RPG_moments (dict) : contains the radar data set
        - polly_var (dict) : contains the lidar data set

    **Kwargs:
        - fig_size (list) : size of the pyplot figures (in inches), default: [12, 7]
        - plot_range (list) : range interval to be plotted, default: [0, 12000] meter
        - rg_converter (bool) : if True, convert range from "m" to "km", default: True

    Returns:
        - fig (matplotlib figure) : figure
        - ax (matplotlib axis) : axis
    """
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [12, 7]
    plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else [0, 12000]
    range2km = kwargs['rg_converter'] if 'rg_converter' in kwargs else True

    # LIMRAD
    if 'Ze' in RPG_moments:
        RPG_moments['Ze']['var_unit'] = 'dBZ'
        RPG_moments['Ze']['var_lims'] = [-60, 20]
        fig_name = f'LIMRAD94_Ze_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['Ze'], fig_size=fig_size,
                                                         range_interval=plot_range, z_converter='lin2z',
                                                         rg_converter=range2km, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'VEL' in RPG_moments:
        RPG_moments['VEL']['var_lims'] = [-4, 2]
        fig_name = f'LIMRAD94_vel_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['VEL'], fig_size=fig_size,
                                                         range_interval=plot_range, rg_converter=range2km,
                                                         title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'sw' in RPG_moments:
        RPG_moments['sw']['var_lims'] = [0, 1]
        fig_name = f'LIMRAD94_sw_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['sw'], fig_size=fig_size,
                                                         range_interval=plot_range, rg_converter=range2km,
                                                         title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    # POLLYxt
    if 'attbsc1064' in polly_var:
        fig_name = f'POLLYxt_attbsc1064_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['attbsc1064'], fig_size=fig_size,
                                                         range_interval=plot_range,
                                                         z_converter="log", rg_converter=range2km, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'depol' in polly_var:
        fig_name = f'POLLYxt_depol_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['depol'], fig_size=fig_size, range_interval=plot_range,
                                                         rg_converter=range2km, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    # POLLYxt interpolated
    orig_masks = {'attbsc1064_ip': polly_var['attbsc1064_ip']['mask'],
                  # 'voldepol532_ip': polly_var['voldepol532_ip']['mask'],
                  'Ze': RPG_moments['Ze']['mask'],
                  'VEL': RPG_moments['VEL']['mask'],
                  'sw': RPG_moments['sw']['mask']}
    training_mask = np.logical_or(RPG_moments['Ze']['mask'], polly_var['attbsc1064_ip']['mask'])
    RPG_moments['Ze']['mask'] = training_mask
    RPG_moments['VEL']['mask'] = training_mask
    RPG_moments['sw']['mask'] = training_mask

    if 'attbsc1064_ip' in polly_var:
        polly_var['attbsc1064_ip']['mask'] = training_mask
        fig_name = f'traing_label_POLLYxt_attbsc1064_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['attbsc1064_ip'], fig_size=fig_size,
                                                         range_interval=plot_range,
                                                         z_converter='log', rg_converter=range2km, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'depol_ip' in polly_var:
        polly_var['depol_ip']['mask'] = training_mask
        fig_name = f'traing_label_POLLYxt_depol_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['depol_ip'], fig_size=fig_size,
                                                         range_interval=plot_range,
                                                         rg_converter=range2km, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'Ze' in RPG_moments:
        fig_name = f'training_set_LIMRAD94_Ze_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['Ze'], fig_size=fig_size,
                                                         range_interval=plot_range, z_converter='lin2z',
                                                         rg_converter=range2km, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'VEL' in RPG_moments:
        fig_name = f'training_set_LIMRAD94_vel_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['VEL'], fig_size=fig_size,
                                                         range_interval=plot_range, rg_converter=range2km,
                                                         title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'sw' in RPG_moments:
        fig_name = f'training_set_LIMRAD94_sw_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['sw'], fig_size=fig_size,
                                                         range_interval=plot_range, rg_converter=range2km,
                                                         title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        save_figure(fig, name=fig_name, dpi=300)

    if 'attbsc1064_ip' in polly_var:
        polly_var['attbsc1064_ip']['mask'] = orig_masks['attbsc1064_ip']
    if 'voldepol532_ip' in polly_var:
        polly_var['voldepol532_ip']['mask'] = orig_masks['voldepol532_ip']
    RPG_moments['Ze']['mask'] = orig_masks['Ze']
    RPG_moments['VEL']['mask'] = orig_masks['VEL']
    RPG_moments['sw']['mask'] = orig_masks['sw']

def spectra_wavelettransform(vel, spectrum, cwt_matrix, **kwargs):
    scales = kwargs['scales'] if 'scales' in kwargs else ValueError('Scales not given! Check call to spectra_wavelettransform')
    ts = kwargs['ts'] if 'ts' in kwargs else 0
    rg = kwargs['rg'] if 'rg' in kwargs else 0
    v_lims = kwargs['v_lims'] if 'v_lims' in kwargs else [0.0, 1.0]
    x_lims = kwargs['x_lims'] if 'x_lims' in kwargs else [-10, 10]
    y_lims = kwargs['y_lims'] if 'y_lims' in kwargs else [-60, 20]
    colormap = kwargs['colormap'] if 'colormap' in kwargs else 'cloudnet_jet'
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.625]
    font_size = kwargs['font_size'] if 'font_size' in kwargs else 10
    font_weight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'
    hydroclass = kwargs['hydroclass'] if 'hydroclass' in kwargs else 'non-typed'

    n_bins_signal = spectrum.size - np.ma.count_masked(spectrum)

    logger.debug("custom colormaps {}".format(VIS_Colormaps.custom_colormaps.keys()))
    if colormap in VIS_Colormaps.custom_colormaps.keys():
        colormap = VIS_Colormaps.custom_colormaps[colormap]

    # plot spectra
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=fig_size)

    ax[0].set_title(f'Doppler spectra, normalized and wavlet transformation\nheight: {str(round(rg, 2))} [m]; ' +
                    f'time: {h.ts_to_dt(ts):%Y-%m-%d %H:%M:%S} [UTC]', fontweight=font_weight, fontsize=font_size)

    nds = ax[0].plot(vel, spectrum, linestyle='-', color='royalblue')
    ax[0].set_xlim(left=x_lims[0], right=x_lims[1])
    ax[0].set_ylim(bottom=0, top=1)
    ax[0].set_ylabel('normalized\nspectrum [1]', fontweight=font_weight, fontsize=font_size)
    ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax[0].grid(linestyle='--')
    ax[0].text(x_lims[1] - 5.0, 10, f'nnz = {n_bins_signal}', fontweight=font_weight, fontsize=font_size)
    ax[0].tick_params(axis='both', which='both', right=False, left=True, top=True)
    ax[0].tick_params(axis='both', which='major', labelsize=font_size, width=2, length=5.5)
    ax[0].text(2.2, 0.875, f'class: {hydroclass}',
               {
                   'color': 'black', 'ha': 'left', 'va': 'center',
                   'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)
               },
               fontsize=font_size, fontweight=font_weight)

    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="2.5%", pad=0.05)
    cax0.axis('off')

    img = ax[1].imshow(cwt_matrix, extent=[x_lims[0], x_lims[1], scales[-1], scales[0]], cmap=colormap, aspect='auto', vmin=v_lims[0], vmax=v_lims[1])

    ax[1].set_ylabel('wavelet scale', fontweight=font_weight, fontsize=font_size)
    ax[1].set_xlabel('Doppler Velocity [m s-1]', fontweight=font_weight, fontsize=font_size)
    ax[1].set_xlim(left=x_lims[0], right=x_lims[1])
    ax[1].set_ylim(bottom=scales[0], top=scales[-1])
    ax[1].set_yticks(np.linspace(scales[0], scales[-1], 4))
    ax[1].grid(linestyle='--')
    ax[1].xaxis.set_ticks_position('both')
    ax[1].invert_yaxis()

    # add the colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="2.5%", pad=0.05)
    fig.add_axes(cax)
    cbar = fig.colorbar(img, cax=cax, orientation="vertical")
    cbar.set_label('||Magnitude||', fontsize=font_size, fontweight=font_weight)
    ax[1].tick_params(axis='both', which='both', right=False, left=True, top=True)
    ax[1].tick_params(axis='both', which='major', labelsize=font_size, width=2, length=5.5)

    fig.subplots_adjust(hspace=0.25)

    return fig, ax

def lidar_profile_range_spectra(lidar, spec, **kwargs):
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [13, 8]
    font_size = kwargs['font_size'] if 'font_size' in kwargs else 14
    font_weight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'
    path = kwargs['path'] if 'path' in kwargs else ''
    cmap = kwargs['colormap'] if 'colormap' in kwargs else spec['colormap']
    plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else [lidar['attbsc1064_ip']['rg'][0], lidar['attbsc1064_ip']['rg'][0]['rg'][-1]]
    bsc = lidar['attbsc1064']
    bsc_interp = lidar['attbsc1064_ip']
    dpl = lidar['depol']
    dpl_interp = lidar['depol_ip']
    ts_list = spec['ts']
    dt_list = [h.ts_to_dt(ts) for ts in dpl['ts']]
    dt_begin = dt_list[0]
    vlims_bsc = np.array(bsc['var_lims'])
    vlims_dpl = np.array(dpl['var_lims'])
    # plot it

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cnt = 0
    for iT, ts in enumerate(ts_list):

        fig_name = path + f'limradVSpolly_{str(cnt).zfill(5)}_{dt_begin:%Y%m%d-%H%M%S}_range_spectrogram.png'
        intervall = {'time': [ts], 'range': plot_range}
        spectrogram_slice = pyLARDA.Transformations.slice_container(spec, value=intervall)

        spectrogram_slice['colormap'] = 'jet'
        spectrogram_slice['var_lims'] = [-60, 20]
        spectrogram_slice['rg_unit'] = 'km'
        spectrogram_slice['rg'] = spectrogram_slice['rg'] / 1000.
        spectrogram_slice['colormap'] = cmap
        bsc['var_lims'] = [5.e-8, 5.e-3]
        dpl['var_lims'] = [-0.05, 0.35]
        dpl['var'][dpl['var'] > 0.3] = 0.3

        iT_lidar = h.argnearest(dpl['ts'], ts_list[iT])

        fig, (axspec, pcmesh) = pyLARDA.Transformations.plot_spectrogram(spectrogram_slice, z_converter='lin2z',
                                                                         fig_size=fig_size, v_lims=[-4, 2], grid='both', cbar=False)
        # additional spectrogram settings
        axspec.patch.set_facecolor('#E0E0E0')
        axspec.patch.set_alpha(0.7)
        axspec.set_ylim(np.array(plot_range) / 1000.)
        axspec.grid(b=True, which='major', color='white', linestyle='--')
        axspec.grid(b=True, which='minor', color='white', linestyle=':')
        axspec.set_ylabel('Height [km]', fontsize=font_size, fontweight=font_weight)
        axspec.grid(linestyle=':', color='white')
        axspec.tick_params(axis='y', which='both', right=False, top=True)
        divider = make_axes_locatable(axspec)
        # range spectrogram colorbar settings
        axcbar = divider.append_axes('top', size=0.2, pad=0.1)
        cbar = fig.colorbar(pcmesh, cax=axcbar, orientation="horizontal")
        cbar.ax.tick_params(axis='both', which='major', labelsize=font_size, width=2, length=4)
        cbar.ax.tick_params(axis='both', which='minor', width=0.5, length=3)
        cbar.ax.minorticks_on()
        axcbar.xaxis.set_label_position('top')
        axcbar.xaxis.set_ticks_position('top')
        axcbar.set_xlabel('Radar Reflectivity [dBZ m$\\mathregular{^{-1}}$ s$\\mathregular{^{-1}}$]', fontweight=font_weight, fontsize=font_size)
        # backscatter plot settings
        axbsc = divider.append_axes("right", size=2.5, pad=0.0)
        axbsc.grid(b=True, which='major', color='white', linestyle='--')
        axbsc.grid(b=True, which='minor', axis='y', color='white', linestyle=':')
        axbsc.set_title(f'{h.ts_to_dt(ts):%b %d, %Y\n%H:%M:%S [UTC]}', fontsize=font_size, fontweight=font_weight)
        settings = [['red', 'attenuated'], ['orange', 'non-liquid'], ['royalblue', 'liquid']]
        for i, iset in enumerate(settings):
            axbsc.scatter(bsc['var'][iT_lidar, bsc['flags'][iT_lidar, :] == i], bsc['rg'][np.where(bsc['flags'][iT_lidar, :] == i)] / 1000.,
                          color=iset[0], alpha=0.9, label=iset[1])
        axbsc.plot(bsc_interp['var'][iT, :], bsc_interp['rg'][:] / 1000., linewidth=2,
                   color='black', alpha=0.5, label=r'interpolation')
        axbsc.set_xlabel('att. bsc. [sr$\\mathregular{^{-1}}$ m$\\mathregular{^{-1}}$]', fontsize=font_size, fontweight=font_weight)
        axbsc.set_xscale('log')
        axbsc.set_xlim(bsc['var_lims'])
        axbsc.set_ylim(np.array(plot_range) / 1000.)
        axbsc.patch.set_facecolor('#E0E0E0')
        axbsc.patch.set_alpha(0.7)
        axbsc.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=6))
        axbsc.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axbsc.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, numticks=6))
        axbsc.tick_params(axis='both', which='both', labelleft=False, left=False, right=False, top=True)
        axbsc.tick_params(axis='both', which='major', labelsize=font_size, width=3, length=5.5)
        axbsc.tick_params(axis='y', which='minor', width=2, length=3)
        axbsc.tick_params(axis='x', which='minor', width=0.75, length=1.5)
        # depol plot settings
        axdpl = divider.append_axes("right", size=2.5, pad=0.0)
        axdpl.grid(b=True, which='major', color='white', linestyle='--')
        axdpl.grid(b=True, which='minor', axis='y', color='white', linestyle=':')
        for i, iset in enumerate(settings):
            axdpl.scatter(dpl['var'][iT_lidar, dpl['flags'][iT_lidar, :] == i], dpl['rg'][np.where(dpl['flags'][iT_lidar, :] == i)] / 1000.,
                          color=iset[0], alpha=0.9, label=iset[1])
        axdpl.plot(dpl_interp['var'][iT, :], dpl_interp['rg'][:] / 1000., linewidth=2,
                   color='black', alpha=0.5, label=r'interpolation')
        axdpl.yaxis.set_label_position("right")
        axdpl.yaxis.tick_right()
        axdpl.set_ylabel('Height [km]', fontsize=font_size, fontweight=font_weight)
        axdpl.set_xlabel('lin. vol. depol [1]', fontsize=font_size, fontweight=font_weight)
        axdpl.set_xlim(dpl['var_lims'])
        axdpl.set_ylim(np.array(plot_range) / 1000.)
        axdpl.patch.set_facecolor('#E0E0E0')
        axdpl.patch.set_alpha(0.7)
        axdpl.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        axdpl.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axdpl.tick_params(axis='both', which='both', right=True, top=True)
        axdpl.tick_params(axis='both', which='major', labelsize=font_size, width=3, length=5.5)
        axdpl.tick_params(axis='y', which='minor', width=2, length=3)
        axdpl.tick_params(axis='x', which='minor', width=1.5, length=2.)
        axdpl.legend(loc=1, fontsize=font_size)
        save_figure(fig, name=fig_name)
        cnt += 1

    bsc['var_lims'] = vlims_bsc
    dpl['var_lims'] = vlims_dpl

    return fig, [axspec, axbsc]

def lidar_profiles(bsc, depol, fltr, **kwargs):
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [12, 9]
    font_size = kwargs['font_size'] if 'font_size' in kwargs else 12
    font_weight = kwargs['font_weight'] if 'font_weight' in kwargs else 'bold'
    plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else [bsc['rg'][0], bsc['rg'][-1]]

    iT = fltr['iT']
    vT = h.ts_to_dt(depol['ts'][iT])
    depol_filter = fltr['depol_filter']
    idx_scl_end = fltr['idx_scl_end']
    idx_scl_start = fltr['idx_scl_start']
    d_bsc_depol = fltr['d_bsc_depol']
    thresh_fac_fcn = fltr['thresh_fac_fcn']
    factor_thresh = thresh_fac_fcn[0]
    idx_bsc_max = fltr['idx_bsc_max']
    min_bsc_thresh = fltr['min_bsc_thresh']
    bsc_diff = fltr['bsc_diff']
    n_bins_250m = fltr['n_bins_250m']

    fig, ax = plt.subplots(2, 1, figsize=fig_size)

    bsc_var_lims = [1e-7, 1e-3]
    depol_var_lims = [0, 0.5]

    ln1 = ax[0].plot(bsc['rg'], bsc['var'][iT, :], color='royalblue', label=r'$\beta_{1064}$')
    ax[0].set_xlabel(f'range [{bsc["rg_unit"]}]', fontsize=font_size, fontweight=font_weight)
    ax[0].set_ylabel(f'att. bsc. [{bsc["var_unit"]}]', color='royalblue', fontsize=font_size, fontweight=font_weight)
    ax[0].set_yscale('log')
    ax[0].set_ylim(bsc_var_lims)
    ax[0].set_xlim(plot_range)

    ax0_right = ax[0].twinx()
    ax1_right = ax[1].twinx()
    ax0_right.set_ylabel('depol', color='grey', fontsize=font_size, fontweight=font_weight)
    ln3 = ax0_right.plot(depol['rg'], depol_filter, 'red', label=r'$\delta_{532}$ corrected')
    ln2 = ax0_right.plot(depol['rg'], depol['var'][iT, :], 'grey', label=r'$\delta_{532}$')
    ax0_right.set_ylim(depol_var_lims)

    width = bsc['rg'][idx_scl_end] - bsc['rg'][idx_scl_start]

    for idx0, idx1, wdth in zip(idx_scl_start, idx_scl_end, width):
        ax[0].axvline(bsc['rg'][idx0], linestyle='--', color='black')
        ax[0].axvline(bsc['rg'][idx1], linestyle='--', color='black')
        ax[1].axvline(bsc['rg'][idx0], linestyle='--', color='black')
        ax[1].axvline(bsc['rg'][idx1], linestyle='--', color='black')

        ax[1].scatter(bsc['rg'][idx0], d_bsc_depol[np.array(idx0)], marker='o', s=50, label=r'points')
        ax1_right.scatter(bsc['rg'][idx1], thresh_fac_fcn[np.array(idx1)], marker='o', s=50, label=r'points')

        ax0_right.annotate(s='', xy=(bsc['rg'][idx1], 0.4), xytext=(bsc['rg'][idx1], 0.4),
                           arrowprops=dict(arrowstyle='<->'))
        ax0_right.text(bsc['rg'][idx0 + 12], 0.375, f'{int(wdth):}',
                       {'color': 'black',
                        'ha': 'center',
                        'va': 'center',
                        'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)
                        }, fontsize=font_size, fontweight=font_weight)

    width0 = bsc['rg'][-1] - bsc['rg'][idx_bsc_max]
    rect = Rectangle((bsc['rg'][idx_bsc_max], 0.0), width0, 0.5, label='excluded data')
    collection = PatchCollection([rect], cmap='jet', alpha=0.5)
    ax0_right.add_collection(collection)

    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    labs.append(rect.get_label())
    ax[0].legend(lns, labs, ncol=1, labelspacing=0.05, fontsize=font_size, loc='upper left')
    ax[0].set_title(f'{vT:%Y-%m-%d - %H:%M:%S [UTC]} -- removed all bsc values < {min_bsc_thresh}',
                    fontsize=font_size, fontweight=font_weight)

    ln0 = ax[1].plot(bsc['rg'], d_bsc_depol, linestyle='-', color='royalblue',
                     label=r'$||\beta_{1064}||-||\delta_{532}||$')
    ax[1].set_xlabel(f'range [{bsc["rg_unit"]}]', fontsize=font_size, fontweight=font_weight)
    ax[1].set_ylabel(f'||backcatter||-||depolarization||', color='royalblue', fontsize=font_size,
                     fontweight=font_weight)  # we already handled the x-label with ax1
    ax[1].set_ylim([-1, 1])
    ax[1].set_xlim(plot_range)

    ax1_right.set_ylabel('signal increase/decrease', color='red', fontsize=font_size, fontweight=font_weight)
    ln1 = ax1_right.plot(bsc['rg'][n_bins_250m:], bsc_diff, linestyle='-', color='red',
                         label=r'$\beta_{1064}(h+250m)*\beta_{1064}^{-1}(h)$')
    ln3 = ax1_right.plot([bsc['rg'][0], bsc['rg'][-1]], [factor_thresh, factor_thresh], linestyle='--', color='k',
                         label=r'thresh')

    ax1_right.set_ylim([1.e-3, 1.e3])
    ax1_right.set_yscale('log')

    lns = ln0 + ln1 + ln3
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, ncol=1, labelspacing=0.05, fontsize=font_size, loc='upper left')

    ax0_right.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax0_right.tick_params(axis='both', which='both', right=True, left=False, top=True)
    ax0_right.tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
    ax0_right.tick_params(axis='both', which='minor', width=2, length=3)

    ax1_right.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1_right.tick_params(axis='both', which='both', right=True, left=False, top=True)
    ax1_right.tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
    ax1_right.tick_params(axis='both', which='minor', width=2, length=3)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=7)
    ax1_right.yaxis.set_minor_locator(locmin)
    ax1_right.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    for i in range(2):
        ax[i].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[i].tick_params(axis='both', which='both', right=False, top=True)
        ax[i].tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
        ax[i].tick_params(axis='both', which='minor', width=2, length=3)

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=5)
    ax[0].yaxis.set_minor_locator(locmin)
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax0_right.tick_params(axis='both', which='minor', width=2, length=3)

    return fig, ax

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

def Histogram(data, **kwargs):
    from copy import copy
    var_info = kwargs['var_info'] if 'var_info' in kwargs else sys.exit(-1)
    n_bins = kwargs['n_bins'] if 'n_bins' in kwargs else 256
    n_Dbins = kwargs['n_Dbins'] if 'n_Dbins' in kwargs else 256
    kind = kwargs['kind'] if 'kind' in kwargs else ''
    y_val = kwargs['y_val'] if 'y_val' in kwargs else np.linspace(-9, 9, 256)
    x_lim = kwargs['x_lim'] if 'x_lim' in kwargs else [0, 1]
    title = kwargs['title'] if 'title' in kwargs else 'Feature/Target space viewer'

    var = data.copy()

    i_moments = 0
    n_variables = 0
    font_size = 15
    font_weight = 'bold'
    list_moments = []
    var_lims = {}
    var[var <= 0.0] = 1.e-6

    if 'Ze_lims' in var_info:
        var_lims.update({'Ze': var_info['Ze_lims']})
        logger.info(f'min/max      Ze = {var[:, i_moments].min():.4f}/{var[:, i_moments].max():.4f}')
        logger.info(f'boundaries   Ze = {var_lims["Ze"][0]:.4f}/{var_lims["Ze"][1]:.4f}')
        i_moments += 1
        n_variables += 1
        list_moments.append('Ze')

    if 'VEL_lims' in var_info:
        var_lims.update({'VEL': var_info['VEL_lims']})
        logger.info(f'min/max     VEL = {var[:, i_moments].min():.4f}/{var[:, i_moments].max():.4f}')
        logger.info(f'boundaries  VEL = {var_lims["VEL"][0]:.4f}/{var_lims["VEL"][1]:.4f}')
        i_moments += 1
        n_variables += 1
        list_moments.append('VEL')

    if 'sw_lims' in var_info:
        var_lims.update({'sw': var_info['sw_lims']})
        logger.info(f'min/max      sw = {var[:, i_moments].min():.4f}/{var[:, i_moments].max():.4f}')
        logger.info(f'boundaries   sw = {var_lims["sw"][0]:.4f}/{var_lims["sw"][1]:.4f}')
        i_moments += 1
        n_variables += 1
        list_moments.append('sw')

    if 'spec_lims' in var_info:
        var_lims.update({'spec': var_info['spec_lims']})
        logger.info(f'min/max    spec = {var[:, i_moments:].min():.4f}/{var[:, i_moments:].max():.4f}')
        logger.info(f'boundaries spec = {0:.4f}/{1:.4f}')
        n_variables += 1

    if 'bsc_lims' in var_info:
        var_lims.update({'bsc': var_info['bsc_lims']})
        logger.info(f'min/max     bsc = {var[:, 0].min():.4f}/{var[:, 0].max():.4f}')
        logger.info(f'boundaries  bsc = {var_lims["bsc"][0]:.4f}/{var_lims["bsc"][1]:.4f}')
        n_variables += 1

    if 'dpl_lims' in var_info:
        var_lims.update({'dpl': var_info['dpl_lims']})
        logger.info(f'min/max     dpl = {var[:, 1].min():.4f}/{var[:, 1].max():.4f}')
        logger.info(f'boundaries  dpl = {var_lims["dpl"][0]:.4f}/{var_lims["dpl"][1]:.4f}')
        n_variables += 1

    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [8, n_variables * 5]
    fig, ax = plt.subplots(n_variables, 1, figsize=fig_size)
    if n_variables == 1:
        ax = [ax]
    ax[0].set_title(title, fontsize=20)

    var[np.isnan(var)] = -1.0

    # make histograms
    for i, ivar in enumerate(list_moments):
        ax[i].hist(var[:, i], bins=np.linspace(var_lims[ivar][0], var_lims[ivar][1], n_bins),
                   density=False, facecolor='royalblue', alpha=0.95)
        ax[i].set_xlim(var_lims[ivar])
        # ax[i].set_ylim([0, 20])
        ax[i].set_yscale('log')
        ax[i].set_ylabel(f'FoO of {list_moments[i]}', fontsize=font_size, fontweight=font_weight)
        ax[i].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[i].tick_params(axis='both', which='both', right=False, top=True)
        ax[i].tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
        ax[i].tick_params(axis='both', which='minor', width=2, length=3)
        ax[i].grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax[i].grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)
        # n, bins, patches

    if kind == 'trainingset':
        H_spec = np.zeros((n_bins, n_Dbins))
        ivar = 'spec'
        for i in range(n_Dbins):
            H_spec[:, i], _ = np.histogram(var[:, i_moments + i],
                                           bins=np.linspace(var_lims[ivar][0], var_lims[ivar][1], n_bins + 1),
                                           density=False)

        import matplotlib.colors as colors
        i = n_variables - 1
        # create figure containing the frequency of occurrence of reflectivity over height and the sensitivity limit
        cmap = copy(plt.get_cmap('viridis'))
        cmap.set_under('white', 1.0)

        pcol = ax[i].pcolormesh(np.linspace(x_lim[0], x_lim[1], n_bins), y_val, H_spec.T,
                                cmap=cmap, label='histogram', norm=colors.LogNorm(ymin=1, ymax=H_spec.max()))

        cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01,
                            format='%2d', orientation='horizontal', fraction=0.1)
        cbar.set_label(label="Frequencies of occurrence of spec values ", fontsize=font_size, fontweight=font_weight)
        cbar.aspect = 100

        ax[i].set_xlim(var_lims['spec'])
        ax[i].set_ylim([y_val[0], y_val[-1]])
        ax[i].set_ylabel('Doppler velocity [m s-1]', fontsize=font_size, fontweight=font_weight)
        ax[i].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[i].tick_params(axis='both', which='both', right=False, top=True)
        ax[i].tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
        ax[i].tick_params(axis='both', which='minor', width=2, length=3)
        ax[i].grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax[i].grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)

    if kind == 'cwt2d':
        n_scales = kwargs['n_scales'] if 'n_scales' in kwargs else var.shape[1]
        H_spec = np.zeros((n_scales, n_Dbins))
        ivar = 'spec'
        for iDbin in range(n_Dbins):
            H_spec[:, iDbin], _ = np.histogram(np.max(var[:, :, iDbin, 1], axis=1),
                                               bins=np.linspace(var_lims[ivar][0], var_lims[ivar][1], n_bins + 1),
                                               density=False)

        import matplotlib.colors as colors
        i = n_variables - 1
        # create figure containing the frequency of occurrence of reflectivity over height and the sensitivity limit
        cmap = copy(plt.get_cmap('viridis'))
        cmap.set_under('white', 1.0)

        pcol = ax[i].pcolormesh(np.linspace(x_lim[0], x_lim[1], n_bins), y_val, H_spec.T,
                                cmap=cmap, label='histogram', norm=colors.LogNorm(ymin=1, ymax=H_spec.max()))

        cbar = fig.colorbar(pcol, use_gridspec=True, extend='min', extendrect=True, extendfrac=0.01,
                            format='%2d', orientation='horizontal', fraction=0.1)
        cbar.set_label(label="Frequencies of occurrence of spec values ", fontsize=font_size, fontweight=font_weight)
        cbar.aspect = 100

        ax[i].set_xlim(var_lims['spec'])
        ax[i].set_ylim([y_val[0], y_val[-1]])
        ax[i].set_ylabel('Doppler velocity [m s-1]', fontsize=font_size, fontweight=font_weight)
        ax[i].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[i].tick_params(axis='both', which='both', right=False, top=True)
        ax[i].tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
        ax[i].tick_params(axis='both', which='minor', width=2, length=3)
        ax[i].grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax[i].grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)

    if 'z_converter' in kwargs and kwargs['z_converter'] == 'log':
        ax.set_xscale('log')
        ax.set_yscale('log')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, ax

def get_ave_values(xvalues, yvalues, n=5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length // n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length // n, n)
    yarr.resize(signal_length // n, n)
    xarr_reshaped = xarr.reshape((-1, n))
    yarr_reshaped = yarr.reshape((-1, n))
    x_ave = xarr_reshaped[:, 0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def plot_signal_plus_average(ax, time, signal, average_over=5):
    # time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    # ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('amplitude', fontsize=16)
    ax.set_title('signal', fontsize=16)
    ax.legend(loc='upper right')

def get_fft_values(y_values, T, N, f_s):
    N2 = 2 ** (int(np.log2(N)) + 1)  # round up to next highest power of 2
    f_values = np.linspace(0.0, 1.0 / (2.0 * T), N2 // 2)
    fft_values_ = fft(y_values)
    fft_values = 2.0 / N2 * np.abs(fft_values_[0:N2 // 2])
    return f_values, fft_values

def plot_fft_plus_power(ax, time, signal, plot_direction='horizontal', yticks=None, ylim=None):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1 / dt

    variance = np.std(signal) ** 2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2
    if plot_direction == 'horizontal':
        ax.plot(f_values, fft_values, 'r-', label='Fourier transform')
        ax.plot(f_values, fft_power, 'k--', linewidth=1, label='FFT power spectrum')
    elif plot_direction == 'vertical':
        scales = 1. / f_values
        scales_log = np.log2(scales)
        ax.plot(fft_values, scales_log, 'r-', label='Fourier transform')
        ax.plot(fft_power, scales_log, 'k--', linewidth=1, label='FFT power spectrum')
        ax.set_yticks(np.log2(yticks))
        ax.set_yticklabels(yticks)
        ax.invert_yaxis()
        ax.set_ylim(ylim[0], -1)
    ax.legend()

def plot_wavelet(ax, time, signal, scales, waveletname='mexh',
                 cmap=plt.cm.seismic, title='', ylabel='', xlabel=''):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
    contourlevels = np.log2(levels)

    im = ax.contourf(time, scales, np.log2(power), contourlevels, extend='both', cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(scales[-1], 1)
    return yticks, ylim

def spectra_wavelettransform2(time, signal, scales):
    fig = plt.figure(figsize=(9, 9))
    spec = gridspec.GridSpec(ncols=6, nrows=6)
    top_ax = fig.add_subplot(spec[:2, 0:4])
    bottom_left_ax = fig.add_subplot(spec[2:, :4])
    bottom_right_ax = fig.add_subplot(spec[2:, 4:])

    plot_signal_plus_average(top_ax, time, signal, average_over=3)
    yticks, ylim = plot_wavelet(bottom_left_ax, time, signal, scales, xlabel='Doppler velocity [m s-1]', ylabel='normalized spectrum [1]', title='')

    plot_fft_plus_power(bottom_right_ax, time, signal, plot_direction='vertical', yticks=yticks, ylim=ylim)
    bottom_right_ax.set_ylabel('??? [???]', fontsize=14)
    plt.tight_layout()

    return fig, (top_ax, bottom_left_ax, bottom_right_ax)

def spectra_3by3(data, *args, **kwargs):
    """Finds the closest match to a given point in time and height and plot Doppler spectra.

        Notes:
            The user is able to provide sliced containers, e.g.

            - one spectrum: ``data['dimlabel'] = ['vel']``
            - range spectrogram: ``data['dimlabel'] = ['range', 'vel']``
            - time spectrogram: ``data['dimlabel'] = ['time, 'vel']``
            - time-range spectrogram: ``data['dimlabel'] = ['time, 'range', 'vel']``

        Args:
            data (dict): data container
            *data2 (dict or numpy.ndarray): data container of a second device
            **z_converter (string): convert var before plotting use eg 'lin2z'
            **var_converter (string): alternate name for the z_converter
            **xmin (float): minimum x axis value
            **xmax (float): maximum x axis value
            **ymin (float): minimum y axis value
            **ymax (float): maximum y axis value
            **save (string): location where to save the pngs
            **fig_size (list): size of png, default is [10, 5.7]
            **mean (float): numpy array dimensions (time, height, 2) containing mean noise level for each spectra
                            in linear units [mm6/m3]
            **thresh (float): numpy array dimensions (time, height, 2) containing noise threshold for each spectra
                              in linear units [mm6/m3]
            **title (str or bool)
            **alpha (float): triggers transparency of the line plot (not the bar plot), 0 <= alpha <= 1

        Returns:  
            tuple with

            - fig (pyplot figure): contains the figure of the plot 
              (for multiple spectra, the last fig is returned)
            - ax (pyplot axis): contains the axis of the plot 
              (for multiple spectra, the last ax is returned)
        """

    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [10, 5.7]
    fsz = kwargs['font_size'] if 'font_size' in kwargs else 17
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0
    name = kwargs['save'] if 'save' in kwargs else ''

    # reschape the spectrum data to (n_time, n_height, n_Dbin)
    time_ts, height, var, mask = h.reshape_spectra(data)
    n_time, n_height, n_Dbin = data['var'].shape

    ncol = 1  # number of columns for legend
    # check if a second data container was given
    if len(args) > 0 and type(args[0]) is dict:
        data2 = args[0]
        time2_ts, height2, var2, mask2 = h.reshape_spectra(data2)
        if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
            var2 = h.get_converter_array(kwargs['z_converter'])[0](var2)
        second_data_set = True
        ncol += 1
    else:
        second_data_set = False

    if 'mean' in kwargs:  ncol += 1
    if 'thresh' in kwargs:  ncol += 1

    # set x-axsis and y-axis limits
    xmin = kwargs['xmin'] if 'xmin' in kwargs else max(min(data['vel']), -8.0)
    xmax = kwargs['xmax'] if 'xmax' in kwargs else min(max(data['vel']), 8.0)
    ymin = kwargs['ymin'] if 'ymin' in kwargs else data['var_lims'][0]
    ymax = kwargs['ymax'] if 'ymax' in kwargs else data['var_lims'][1]

    logger.debug(f'x-axis varlims {xmin} {xmax}')
    logger.debug(f'y-axis varlims {ymin} {ymax}')

    if 'var_converter' in kwargs:
        kwargs['z_converter'] = kwargs['var_converter']
    if 'z_converter' in kwargs and kwargs['z_converter'] == 'lin2z':
        var = h.get_converter_array(kwargs['z_converter'])[0](var)

    # plot spectra
    ifig = 1
    n_figs = (n_time - 2) * (n_height - 2)

    assert n_time > 2 and n_height > 2, 'Time-Height slice too small. Need at least 3 time steps and 3 range gates!'

    for iT in range(1, n_time - 1):
        for iH in range(1, n_height - 1):

            fig = plt.figure(figsize=fig_size)
            gs1 = gridspec.GridSpec(3, 3)
            gs1.update(wspace=0, hspace=0)  # set the spacing between axes.

            dt_center, rg_center = h.ts_to_dt(time_ts[iT]), height[iH]
            i = -1
            for irow, icol in itertools.product(range(-1, 2), range(-1, 2)):
                ax = plt.subplot(gs1[irow + 1, icol + 1])

                # plot the spectrum
                ax.plot(data['vel'], var[iT + icol, iH + irow, :], color='royalblue', linestyle='-',
                        linewidth=2, alpha=alpha, label=data['system'] + ' ' + data['name'])

                # if a 2nd dict is given, plot the spectrum of the 2nd device using the nearest
                # spectrum point in time and height with respect to device 1
                if second_data_set:
                    # find the closest spectra to the first device
                    iT2 = h.argnearest(time2_ts, time_ts[iT + icol])
                    iH2 = h.argnearest(height2, height[iH + irow])
                    ax.plot(data2['vel'], var2[iT2 + icol, iH2 + irow, :], color='darkred', linestyle='-',
                            linewidth=2, alpha=alpha, label=data2['system'] + ' ' + data2['name'])

                    diff_t = np.abs(time_ts[iT + icol] - time2_ts[iT2])
                    diff_h = np.abs(height[iH + irow] - height2[iH2])
                    ax.text(-6, 12, r'$\bigtriangleup t = $' + f'{diff_t:.1f} [s];', fontsize=12)
                    ax.text(.5, 12, r'$\bigtriangleup h = $' + f'{diff_h:.1f} [m]', fontsize=12)

                # if mean noise level is given add it to plot
                if 'mean' in kwargs and kwargs['mean'][iT, iH] > 0.0:
                    mean = h.lin2z(kwargs['mean'][iT, iH]) if kwargs['mean'].shape != () \
                        else h.lin2z(kwargs['mean'])
                    legendtxt_mean = f'mean noise floor =  {mean:.2f} '
                    ax.plot([data['vel'][0], data['vel'][-1]], [mean, mean], color='k', linestyle='--', linewidth=1, label=legendtxt_mean)

                # if thresh noise level is given add it to plot
                if 'thresh' in kwargs and kwargs['thresh'][iT, iH] > 0.0:
                    thresh = h.lin2z(kwargs['thresh'][iT, iH]) if kwargs['thresh'].shape != () \
                        else h.lin2z(kwargs['thresh'])
                    legendtxt_thresh = f'noise floor threshold =  {thresh:.2f} '
                    ax.plot([data['vel'][0], data['vel'][-1]], [thresh, thresh], color='k', linestyle='-', linewidth=1, label=legendtxt_thresh)

                ax.set_xlim(left=xmin, right=xmax)
                ax.set_ylim(bottom=ymin, top=ymax)

                ax.grid(which='major', linestyle='--', linewidth=1)
                ax.grid(which='minor', linestyle=':', linewidth=0.75)

                ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
                ax.tick_params(axis='both', which='both', right=True, top=True)
                ax.tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
                ax.tick_params(axis='both', which='minor', width=2, length=3)
                ax.set_yticks(np.linspace(ymin + 10, ymax - 10, 5))

                # remove x and y labels from the upper right 4 subplots
                if irow != 1:  ax.set_xticklabels([])
                if icol != -1: ax.set_yticklabels([])

            fig.text(0.5, 0.04, 'Doppler velocity [m s-1]', ha='center', fontsize=fsz)
            fig.text(0.04, 0.5, 'spectrum power [dBZ]', va='center', rotation='vertical', fontsize=fsz)

            if 'title' in kwargs and type(kwargs['title']) == str:
                fig.suptitle(kwargs['title'], fontsize=20)
            elif 'title' in kwargs and type(kwargs['title']) == bool:
                if kwargs['title']:
                    fig.suptitle(f'{data["paraminfo"]["location"]},  '
                                 f'{dt_center:%Y-%m-%d %H:%M:%S} [UTC],  '
                                 f'{rg_center:.0f} [m]',
                                 fontsize=20)

            # gather plot labels and generate legend
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=ncol)
            fig.tight_layout(rect=[0.05, 0.05, 1, 0.96])

            figure_name = name + f'{dt_center:%Y%m%d_%H%M%S_UTC}_{str(ifig).zfill(4)}_{height[iH]:5.0f}m.png'
            fig.savefig(figure_name, dpi=150)
            logger.info(f'Saved {ifig} of {n_figs} png to {figure_name}')

            ifig += 1
            if ifig != n_figs + 1: plt.close(fig)

    return fig, ax

def _plot_bar_data(fig, ax, data, time, mask_value=0.0):
    """Plots 1D variable as bar plot.
    Args:
        ax (obj): Axes object.
        data (ndarray): 1D data array.
        time (ndarray): 1D time array.
    """
    data = np.ma.masked_less_equal(data, mask_value)
    pos0 = ax.get_position()
    ax_new = fig.add_axes([0., 0., 1., 1.])
    ax_new.bar(time, data.filled(0), width=1 / 1200, align='center', alpha=0.5, color='royalblue')
    ax_new.set(
        ylim=[-10, 200], ylabel='mwr-lwp [g m-2]',
        xlim=[time[0], time[-1]],
        position=[pos0.x0, pos0.height + pos0.height * 0.125, pos0.width, pos0.height / 2],
        # fontweight='semibold'
    )
    ax_new.tick_params(labelbottom=False, labeltop=True)
    ax_new.grid(True)

    time_extend = time[-1] - time[0]
    ax_new = tr.set_xticks_and_xlabels(ax_new, time_extend)
    ax_new.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax_new.tick_params(axis='both', which='both', right=True, top=True)
    ax_new.tick_params(axis='both', which='major', labelsize=_FONT_SIZE, width=3, length=5.5)
    ax_new.tick_params(axis='both', which='minor', width=2, length=3)
    return ax_new

def plot_ll_thichkness(ax, t, l1, l2):
    y_lim = [-0.12, 2.5]

    ax1 = ax.twinx()
    ax1.plot(t, l1 / 1000., color='#E64A23', alpha=0.75, label='neural network (nn)')
    ax1.set_ylim(y_lim)
    ax1.plot(t, l2 / 1000., color='navy', alpha=0.75, label='cloudnet (cn)')
    # ax1.plot(dt_list, sum_ll_thickness[nn_varname], color='red', linestyle='-', alpha=0.75, label=nn_varname)

    ax1.set(ylim=y_lim, ylabel='liquid layer thickness [km]')
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax1.tick_params(axis='both', which='both', right=True)
    ax1.tick_params(axis='both', which='major', labelsize=_FONT_SIZE, width=3, length=5.5)
    ax1.tick_params(axis='both', which='minor', width=2, length=3)
    ax1 = tr.set_xticks_and_xlabels(ax1, t[-1] - t[0])
    ax1.legend(loc='best')

    return ax1
