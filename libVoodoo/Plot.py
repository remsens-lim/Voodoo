"""
This module contains additional plotting routines used for displaying quicklooks of the ANN input and output, also histories and histograms.
"""
import sys

#sys.path.append('../../larda/')
#sys.path.append('.')

import time
import datetime
import numpy as np
import pandas as pd
import pyLARDA
import pyLARDA.helpers as h

import libVoodoo.Plot as Plot

import matplotlib
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib import ticker
from matplotlib import gridspec


__author__      = "Willi Schimmel"
__copyright__   = "Copyright 2019, The Voodoo Project"
__credits__     = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__     = "MIT"
__version__     = "0.0.1"
__maintainer__  = "Willi Schimmel"
__email__       = "willi.schimmel@uni-leipzig.de"
__status__      = "Prototype"

def History(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    figure, axis = plt.subplots(nrows=1, ncols=1)
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Mean Abs Error')
    for loss in history.history:
        axis.plot(hist['epoch'], hist[loss], label=f'Train Error {loss}')
    #axis.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Val Error')
    axis.legend()

    return figure, axis


def Quicklooks(RPG_moments, polly_var, radar_list, lidar_list, begin_dt, end_dt, **kwargs):
    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [12, 7]
    plot_range = kwargs['plot_range'] if 'plot_range' in kwargs else [RPG_moments['Ze']['rg'][0],
                                                                      RPG_moments['Ze']['rg'][-1]]

    # LIMRAD
    RPG_moments['Ze']['var_unit'] = 'dBZ'
    RPG_moments['Ze']['var_lims'] = [-60, 20]
    fig_name = f'LIMRAD94_Ze_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
    fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['Ze'], fig_size=fig_size,
                                                     range_interval=plot_range, z_converter='lin2z',
                                                     rg_converter=True, title=fig_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_name, dpi=300)
    print(f' Save figure :: {fig_name}')

    RPG_moments['VEL']['var_lims'] = [-4, 2]
    fig_name = f'LIMRAD94_vel_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
    fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['VEL'], fig_size=fig_size,
                                                     range_interval=plot_range, rg_converter=True,
                                                     title=fig_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_name, dpi=300)
    print(f' Save figure :: {fig_name}')


    RPG_moments['sw']['var_lims'] = [0, 1]
    fig_name = f'LIMRAD94_sw_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
    fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['sw'], fig_size=fig_size,
                                                     range_interval=plot_range, rg_converter=True,
                                                     title=fig_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_name, dpi=300)
    print(f' Save figure :: {fig_name}')

    # POLLYxt
    if 'attbsc1064' in polly_var:
        fig_name = f'POLLYxt_attbsc1064_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['attbsc1064'], fig_size=fig_size,
                                                         range_interval=plot_range,
                                                         z_converter="log", rg_converter=True, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

    if 'voldepol532' in polly_var:
        fig_name = f'POLLYxt_voldepol532_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['voldepol532'], fig_size=fig_size, range_interval=plot_range,
                                                         rg_converter=True, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

    # POLLYxt interpolated
    orig_masks = {'attbsc1064_ip':  polly_var['attbsc1064_ip']['mask'],
                  #'voldepol532_ip': polly_var['voldepol532_ip']['mask'],
                  'Ze':  RPG_moments['Ze']['mask'],
                  'VEL': RPG_moments['VEL']['mask'],
                  'sw':  RPG_moments['sw']['mask']}
    training_mask = np.logical_or(RPG_moments['Ze']['mask'], polly_var['attbsc1064_ip']['mask'])
    RPG_moments['Ze']['mask'] = training_mask
    RPG_moments['VEL']['mask'] = training_mask
    RPG_moments['sw']['mask'] = training_mask

    if 'attbsc1064_ip' in polly_var:
        polly_var['attbsc1064_ip']['mask'] = training_mask
        fig_name = f'traing_label_POLLYxt_attbsc1064_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['attbsc1064_ip'], fig_size=fig_size,
                                                         range_interval=plot_range,
                                                         z_converter='log', rg_converter=True, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

    if 'voldepol532_ip' in polly_var:
        polly_var['voldepol532_ip']['mask'] = training_mask
        fig_name = f'traing_label_POLLYxt_voldepol532_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
        fig, _ = pyLARDA.Transformations.plot_timeheight(polly_var['voldepol532_ip'], fig_size=fig_size,
                                                         range_interval=plot_range,
                                                         rg_converter=True, title=fig_name)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_name, dpi=300)
        print(f' Save figure :: {fig_name}')

    fig_name = f'training_set_LIMRAD94_Ze_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
    fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['Ze'], fig_size=fig_size,
                                                     range_interval=plot_range, z_converter='lin2z',
                                                     rg_converter=True, title=fig_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_name, dpi=300)
    print(f' Save figure :: {fig_name}')

    fig_name = f'training_set_LIMRAD94_vel_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
    fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['VEL'], fig_size=fig_size,
                                                     range_interval=plot_range, rg_converter=True,
                                                     title=fig_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_name, dpi=300)
    print(f' Save figure :: {fig_name}')


    fig_name = f'training_set_LIMRAD94_sw_{begin_dt:%Y%m%d_%H%M%S}_{end_dt:%H%M%S}.png'
    fig, _ = pyLARDA.Transformations.plot_timeheight(RPG_moments['sw'], fig_size=fig_size,
                                                     range_interval=plot_range, rg_converter=True,
                                                     title=fig_name)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_name, dpi=300)
    print(f' Save figure :: {fig_name}')

    if 'attbsc1064_ip' in polly_var:
        polly_var['attbsc1064_ip']['mask'] = orig_masks['attbsc1064_ip']
    if 'voldepol532_ip' in polly_var:
        polly_var['voldepol532_ip']['mask'] = orig_masks['voldepol532_ip']
    RPG_moments['Ze']['mask'] = orig_masks['Ze']
    RPG_moments['VEL']['mask'] = orig_masks['VEL']
    RPG_moments['sw']['mask'] = orig_masks['sw']


def lidar_profile_range_spectra(lidar, spec, **kwargs):
    fig_size    = kwargs['fig_size']    if 'fig_size'    in kwargs else [15, 8]
    font_size   = kwargs['font_size']   if 'font_size'   in kwargs else 14
    font_weight = kwargs['font_weight'] if 'font_weight' in kwargs else 'semibold'
    path        = kwargs['path']        if 'path'        in kwargs else ''
    cmap        = kwargs['colormap']    if 'colormap'    in kwargs else spec['colormap']
    plot_range  = kwargs['plot_range']  if 'plot_range'  in kwargs else [lidar['attbsc1064_ip']['rg'][0], lidar['attbsc1064_ip']['rg'][0]['rg'][-1]]
    bsc         = lidar['attbsc1064']
    bsc_interp  = lidar['attbsc1064_ip']
    dpl         = lidar['depol']
    dpl_interp  = lidar['depol_ip']
    ts_list     = spec['ts']
    dt_list     = [h.ts_to_dt(ts) for ts in dpl['ts']]
    dt_begin    = dt_list[0]
    vlims_bsc   = np.array(bsc['var_lims'])
    vlims_dpl   = np.array(dpl['var_lims'])
    # plot it

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    cnt = 0
    for iT, ts in enumerate(ts_list):

        fig_name = path + f'limrad_{str(cnt).zfill(4)}_{dt_begin:%Y%m%d}_range_spectrogram.png'
        intervall = {'time': [ts], 'range': plot_range}
        spectrogram_slice = pyLARDA.Transformations.slice_container(spec, value=intervall)

        spectrogram_slice['colormap'] = 'jet'
        spectrogram_slice['var_lims'] = [-60, 20]
        spectrogram_slice['rg_unit']  = 'km'
        spectrogram_slice['rg']       = spectrogram_slice['rg']/1000.
        spectrogram_slice['colormap']  = cmap
        bsc['var_lims'] = [5.e-8, 5.e-3]
        dpl['var_lims'] = [-0.05, 0.35]
        dpl['var'][dpl['var'] > 0.3] = 0.3

        iT_lidar = h.argnearest(dpl['ts'], ts_list[iT])

        fig, (axspec, pcmesh) = pyLARDA.Transformations.plot_spectrogram(spectrogram_slice, fig_size=fig_size, v_lims=[-7, 7], grid='both', cbar=False)
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
    fig_size    = kwargs['fig_size']    if 'fig_size'    in kwargs else [12, 9]
    font_size   = kwargs['font_size']   if 'font_size'   in kwargs else 12
    font_weight = kwargs['font_weight'] if 'font_weight' in kwargs else 'bold'
    plot_range  = kwargs['plot_range']  if 'plot_range'  in kwargs else [bsc['rg'][0], bsc['rg'][-1]]

    iT             = fltr['iT']
    vT             = h.ts_to_dt(depol['ts'][iT])
    depol_filter   = fltr['depol_filter']
    idx_scl_end    = fltr['idx_scl_end']
    idx_scl_start  = fltr['idx_scl_start']
    d_bsc_depol    = fltr['d_bsc_depol']
    thresh_fac_fcn = fltr['thresh_fac_fcn']
    factor_thresh  = thresh_fac_fcn[0]
    idx_bsc_max    = fltr['idx_bsc_max']
    min_bsc_thresh = fltr['min_bsc_thresh']
    bsc_diff       = fltr['bsc_diff']
    n_bins_250m    = fltr['n_bins_250m']

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
    #ax0_right.tick_params(axis='both', which='minor', width=2, length=3)

    return fig, ax

def print_elapsed_time(t0, string='time = '):
    print(f'{string}{datetime.timedelta(seconds=int(time.time() - t0))} [hour:min:sec]')

def save_figure(fig, **kwargs):
    dotsperinch = kwargs['dpi']  if 'dpi'  in kwargs else 200
    name        = kwargs['name'] if 'name' in kwargs else 'no-name.png'
    fig.savefig(name, dpi=dotsperinch)
    print(f'Save figure :: {name}')

def Histogram(data, **kwargs):
    from copy import copy
    var_info = kwargs['var_info'] if 'var_info' in kwargs else sys.exit(-1)
    n_bins   = kwargs['n_bins']   if 'n_bins'   in kwargs else 256
    n_Dbins  = kwargs['n_Dbins']  if 'n_Dbins'  in kwargs else 256
    kind     = kwargs['kind']     if 'kind'     in kwargs else ''
    y_val    = kwargs['y_val']    if 'y_val'    in kwargs else np.linspace(-9, 9, 256)
    x_lim    = kwargs['x_lim']    if 'x_lim'    in kwargs else [0, 1]
    title    = kwargs['title']    if 'title'    in kwargs else 'Feature/Target space viewer'

    var = data.copy()

    i_moments = 0
    n_variables = 0
    font_size = 15
    font_weight = 'bold'
    list_moments = []
    var_lims = {}
    var[var <= 0.0] = 1.e-6

    if 'Ze_lims' in var_info:
#        if 'Ze_converter' in var_info and var_info['Ze_converter'] == 'lin2z':
#            var_lims.update({'Ze': 10*np.log10(var_info['Ze_lims'])})
#            var[:, 0] = 10*np.log10(var[:, i_moments])
#        else:
        var_lims.update({'Ze': var_info['Ze_lims']})
        print(f'min/max      Ze = {var[:, i_moments].min():.4f}/{var[:, i_moments].max():.4f}')
        print(f'boundaries   Ze = {var_lims["Ze"][0]:.4f}/{var_lims["Ze"][1]:.4f}')
        i_moments += 1
        n_variables += 1
        list_moments.append('Ze')

    if 'VEL_lims' in var_info:
        var_lims.update({'VEL': var_info['VEL_lims']})
        print(f'min/max     VEL = {var[:, i_moments].min():.4f}/{var[:, i_moments].max():.4f}')
        print(f'boundaries  VEL = {var_lims["VEL"][0]:.4f}/{var_lims["VEL"][1]:.4f}')
        i_moments += 1
        n_variables += 1
        list_moments.append('VEL')

    if 'sw_lims' in var_info:
        var_lims.update({'sw': var_info['sw_lims']})
        print(f'min/max      sw = {var[:, i_moments].min():.4f}/{var[:, i_moments].max():.4f}')
        print(f'boundaries   sw = {var_lims["sw"][0]:.4f}/{var_lims["sw"][1]:.4f}')
        i_moments += 1
        n_variables += 1
        list_moments.append('sw')

    if 'spec_lims' in var_info:
#        if 'spec_converter' in var_info and var_info['spec_converter'] == 'lin2z':
#            var_lims.update({'spec': 10*np.log10(var_info['spec_lims'])})
#            var[:, i_moments:] = 10*np.log10(var[:, i_moments:])
#        else:

        var_lims.update({'spec': var_info['spec_lims']})
        print(f'min/max    spec = {var[:, i_moments:].min():.4f}/{var[:, i_moments:].max():.4f}')
        print(f'boundaries spec = {0:.4f}/{1:.4f}')
        n_variables += 1

    if 'bsc_lims' in var_info:
#        if 'bsc_converter' in var_info and var_info['bsc_converter'] == 'log':
#            var_lims.update({'bsc': np.log10(var_info['bsc_lims'])})
#            var[:, 0] = np.log10(var[:, 0])
#        else:
        var_lims.update({'bsc': var_info['bsc_lims']})
        print(f'min/max     bsc = {var[:, 0].min():.4f}/{var[:, 0].max():.4f}')
        print(f'boundaries  bsc = {var_lims["bsc"][0]:.4f}/{var_lims["bsc"][1]:.4f}')
        n_variables += 1

    if 'dpl_lims' in var_info:
#        if 'dpl_converter' in var_info and var_info['dpl_converter'] == 'ldr2cdr':
#            var_lims.update({'dpl': np.log10(-2.0*var_info['dpl_lims']/(var_info['dpl_lims']-1.0))/(np.log10(2)+np.log10(5))})
#            var[:, 1] = np.log10(-2.*var[:, 1]/(var[:, 1]-1.0))/(np.log10(2)+np.log10(5))
#        else:
        var_lims.update({'dpl': var_info['dpl_lims']})
        print(f'min/max     dpl = {var[:, 1].min():.4f}/{var[:, 1].max():.4f}')
        print(f'boundaries  dpl = {var_lims["dpl"][0]:.4f}/{var_lims["dpl"][1]:.4f}')
        n_variables += 1

    fig_size = kwargs['fig_size'] if 'fig_size' in kwargs else [8, n_variables*5]
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
        #ax[i].set_ylim([0, 20])
        ax[i].set_yscale('log')
        ax[i].set_ylabel(f'FoO of {list_moments[i]}', fontsize=font_size, fontweight=font_weight)
        ax[i].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax[i].tick_params(axis='both', which='both', right=False, top=True)
        ax[i].tick_params(axis='both', which='major', labelsize=14, width=3, length=5.5)
        ax[i].tick_params(axis='both', which='minor', width=2, length=3)
        ax[i].grid(b=True, which='major', color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax[i].grid(b=True, which='minor', color='gray', linestyle=':', linewidth=0.25, alpha=0.5)
        #n, bins, patches

    if kind == 'trainingset':
        H_spec = np.zeros((n_bins, n_Dbins))
        ivar = 'spec'
        for i in range(n_Dbins):
            H_spec[:, i], _ = np.histogram(var[:, i_moments+i],
                                           bins=np.linspace(var_lims[ivar][0], var_lims[ivar][1], n_bins+1),
                                           density=False)

        import matplotlib.colors as colors
        i = n_variables-1
        # create figure containing the frequency of occurrence of reflectivity over height and the sensitivity limit
        cmap = copy(plt.get_cmap('viridis'))
        cmap.set_under('white', 1.0)

        pcol = ax[i].pcolormesh(np.linspace(x_lim[0], x_lim[1], n_bins), y_val, H_spec.T,
                                cmap=cmap, label='histogram', norm=colors.LogNorm(vmin=1, vmax=H_spec.max()))

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
                                               bins=np.linspace(var_lims[ivar][0], var_lims[ivar][1], n_bins+1),
                                               density=False)

        import matplotlib.colors as colors
        i = n_variables-1
        # create figure containing the frequency of occurrence of reflectivity over height and the sensitivity limit
        cmap = copy(plt.get_cmap('viridis'))
        cmap.set_under('white', 1.0)

        pcol = ax[i].pcolormesh(np.linspace(x_lim[0], x_lim[1], n_bins), y_val, H_spec.T,
                                cmap=cmap, label='histogram', norm=colors.LogNorm(vmin=1, vmax=H_spec.max()))

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


