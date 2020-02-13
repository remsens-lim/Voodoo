########################################################################################################################
#
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
import datetime
import sys

sys.path.append('../larda/')
sys.path.append('.')

import pyLARDA

import logging
import matplotlib.pyplot as plt

import numpy as np
import time
import pyLARDA.helpers as h
import pyLARDA.SpectraProcessing as sp
import voodoo.libVoodoo.Loader as Loader
import voodoo.libVoodoo.Plot as Plot

from PIL import Image
from numpy import pi
from colorsys import hls_to_rgb


def plot_as_colormesh(image, axes, **pcolormeshkwargs):
    raveled_pixel_shape = (image.shape[0]*image.shape[1], image.shape[2])
    color_tuple = image.transpose((1,0,2)).reshape(raveled_pixel_shape)

    if color_tuple.dtype == np.uint8:
        color_tuple = color_tuple / 255.

    index = np.tile(np.arange(image.shape[0]), (image.shape[1],1))
    quad = axes.pcolormesh(index, color=color_tuple, linewidth=0, **pcolormeshkwargs)
    quad.set_array(None)

def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c

def get_concat_h(im1, im2, *args):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    if len(args) > 0:
        dst.paste(args[0], (im1.width+args[0].width, 0))

    return dst

if __name__ == '__main__':

    spec_settings = {'rm_precip_ghost': False,   # reduces the domain (Nyquist velocitdy) by ± 2.5 [m/s], when signal > 0 [dBZ] within 200m above antenna
                     'do_despeckle3d': False,   # 3D convolution (5x5x5 window), removes single non-zero values, very slow!
                     'despeckle': False,         # 2D convolution (5x5 window), removes single non-zero values, very slow!
                     'NF': 6.0,                  # estimating noise in spectra noise_threshold = mean(noise) + noise_factor * std(noise)
                     'main_peak': True,         #
                     'filter_ghost_C1': False,   #
                     }

    PATH = '/home/sdig/code/larda3/voodoo/plots/spectra_png/'

    start_time = time.time()

    log = logging.getLogger('pyLARDA')
    log.setLevel(logging.CRITICAL)
    log.addHandler(logging.StreamHandler())

    # Load LARDA
    larda = pyLARDA.LARDA().connect('lacros_dacapo_gpu')

    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    # gather argument
    if 'date' in kwargs:
        date = str(kwargs['date'])
        begin_dt = datetime.datetime.strptime(date + ' 00:00:10', '%Y%m%d %H:%M:%S')
        end_dt = datetime.datetime.strptime(date + ' 23:59:50', '%Y%m%d %H:%M:%S')
    else:
        # date = '2019050'
        #begin_dt = datetime.datetime(2019, 1, 10, 12, 15)
        #end_dt = datetime.datetime(2019, 1, 10, 12, 57)
        begin_dt = datetime.datetime(2019, 8, 1, 6, 44,)
        end_dt = datetime.datetime(2019, 8, 1, 7, 20)

    TIME_SPAN_ = [begin_dt, end_dt]
    RANGE_ = [0, 12000]
    NF = float(kwargs['NF']) if 'NF' in kwargs else 6.0

    limrad94_Zspec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_, **spec_settings)
    limrad94_mom = sp.spectra2moments(limrad94_Zspec, larda.connectors['LIMRAD94'].system_info['params'], **spec_settings)
    limrad94_pol = sp.spectral_polarimetric_products(limrad94_Zspec, larda.connectors['LIMRAD94'].system_info['params'], **spec_settings)

    for iC in range(len(limrad94_Zspec)):
        limrad94_Zspec[iC].update(
            {varname: h.put_in_container(limrad94_pol[iC][varname], limrad94_Zspec[iC]['VHSpec'], name=varname) for varname in limrad94_pol[iC].keys()})

    # loading cloudnet temperature data
    T = larda.read("CLOUDNET_LIMRAD", "T", [begin_dt, end_dt], [0, 'max'])

    def toC(datalist):
        return datalist[0]['var'] - 273.15, datalist[0]['mask']

    T = pyLARDA.Transformations.combine(toC, [T], {'var_unit': "C"})
    contour = {'data': T, 'levels': np.arange(-35, 1, 10)}

    ########################################################################################################FONT=CYBERMEDIUM
    #
    #   ___  _    ____ ___ ___ _ _  _ ____    ____ ____ ___  ____ ____    _  _ ____ _  _ ____ _  _ ___ ____
    #   |__] |    |  |  |   |  | |\ | | __    |__/ |__| |  \ |__| |__/    |\/| |  | |\/| |___ |\ |  |  [__
    #   |    |___ |__|  |   |  | | \| |__]    |  \ |  | |__/ |  | |  \    |  | |__| |  | |___ | \|  |  ___]
    #
    plot_remsen_ql = False
    plot_radar_moments = False
    fig_size = [9, 6]
    # create folder for subfolders if it doesn't exist already
    h.change_dir(f'{PATH}/{begin_dt:%Y%m%d-%H%M}-{end_dt:%H%M}/')

    if plot_radar_moments:


        ZE = limrad94_mom['Ze']
        ZE['var_unit'] = 'dBZ'
        ZE['colormap'] = 'jet'
        ZE['var_lims'] = [-50, 20]
        fig, _ = pyLARDA.Transformations.plot_timeheight(ZE, fig_size=fig_size, range_interval=RANGE_,
                                                         z_converter='lin2z', contour=contour, rg_converter=True)
        fig_name = 'limrad_Ze' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)
        print('fig saved:', fig_name)

        VEL = limrad94_mom['VEL']
        VEL['var_lims'] = [-4, 2]
        fig, _ = pyLARDA.Transformations.plot_timeheight(VEL, fig_size=fig_size, range_interval=RANGE_,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_VEL' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        sw = limrad94_mom['sw']
        sw['var_lims'] = [0, 0.5]
        fig, _ = pyLARDA.Transformations.plot_timeheight(sw, fig_size=fig_size, range_interval=RANGE_,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_sw' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        skew = limrad94_mom['skew']
        skew['var_lims'] = [-0.5, 1]
        fig, _ = pyLARDA.Transformations.plot_timeheight(skew, fig_size=fig_size, range_interval=RANGE_,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_skew' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        skew_smoothed = skew
        skew_smoothed['var_lims'] = [-0.5, 1]

        import scipy as sp
        import scipy.ndimage

        sigma_y = 1.0
        sigma_x = 1.0
        sigma = [sigma_y, sigma_x]

        skew_smoothed['var'] = sp.ndimage.filters.gaussian_filter(skew['var'].copy(),
                                                                           sigma=1, mode='nearest')

        fig, _ = pyLARDA.Transformations.plot_timeheight(skew_smoothed, fig_size=fig_size,
                                                         range_interval=RANGE_,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_skew_smoothed' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        kurt = limrad94_mom['kurt']
        kurt['var_lims'] = [1, 6]
        fig, _ = pyLARDA.Transformations.plot_timeheight(kurt, fig_size=fig_size, range_interval=RANGE_,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_kurt' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        #
        ##    VEL = limrad94_mom['VEL']
        ##    VEL['var_lims'] = [-4, 2]
        ##    fig, _ = pyLARDA.Transformations.plot_timeheight(VEL, fig_size=[16, 3], range_interval=[0, 12000])
        ##    fig_name = 'limrad_VEL' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:o%H%M%S_}' + '0-12000' + '_filter.png'
        ##    fig.savefig(fig_name, dpi=250)
        #
        #    import scipy.ndimage as ndimage
        #
        #    ZE['var'] = ndimage.gaussian_filter(ZE['var'], sigma=1.0, order=0)
        #    fig, _ = pyLARDA.Transformations.plot_timeheight(ZE, fig_size=[16, 3], range_interval=[0, 12000],
        #                                                     #z_converter='lin2z',
        #                                                     title='filtered, smothed')
        #    fig_name = 'limrad_skew' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + '0-12000' + '_filter-smothed.png'
        #    fig.savefig(fig_name, dpi=250)
        #
        #    sw = limrad94_mom['sw']
        #    fig, _ = pyLARDA.Transformations.plot_timeheight(sw, fig_size=[16, 3], range_interval=[0, 12000])
        #    fig_name = 'limrad_sw' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S_}' + '0-12000' + '_filter..png'
        #    fig.savefig(fig_name, dpi=250)


    sensitivity_limit = larda.read("LIMRAD94", "SLv", TIME_SPAN_, RANGE_)
    RPG_spectra_interp = Loader.equalize_rpg_radar_chirps(limrad94_Zspec, 'VHSpec')
    #RPG_sldr_spectra = Loader.equalize_rpg_radar_chirps(limrad94_Zspec, 'ZDR')
    #RPG_spectra_interp['var'] = Loader.replace_fill_value(RPG_spectra_interp['var'], sensitivity_limit['var'])
    #RPG_spectra_interp['mask'][:, :] = False

    FONTW = 'normal'
    FONTS = 12
    FIGS  = [6, 6]

    for its, ts in enumerate(RPG_spectra_interp['ts']):
        fig_name = f'limrad_{str(its).zfill(4)}_{begin_dt:%Y%m%d}'

        intervall = {'time': [ts], 'range': RANGE_}

        spectrogram_LIMRAD = pyLARDA.Transformations.slice_container(RPG_spectra_interp, value=intervall)
        spectrogram_LIMRAD['colormap'] = 'cloudnet_jet'
        spectrogram_LIMRAD['var_lims'] = [-60, 20]
        spectrogram_LIMRAD['var_unit'] = 'dBZ'
        spectrogram_LIMRAD['rg_unit'] = 'km'
        spectrogram_LIMRAD['rg'] = spectrogram_LIMRAD['rg'] / 1000.


        fig_sp, (axspec, pcmesh) = pyLARDA.Transformations.plot_spectrogram(spectrogram_LIMRAD,
                                                                            z_converter='lin2z',
                                                                            fig_size=FIGS, v_lims=[-4, 4], grid='both', cbar=True)
        # additional spectrogram settings
        axspec.patch.set_facecolor('#E0E0E0')
        axspec.patch.set_alpha(0.7)
        axspec.set_ylim(np.array(RANGE_) / 1000.)
        axspec.grid(b=True, which='major', color='white', linestyle='--')
        axspec.grid(b=True, which='minor', color='white', linestyle=':')
        axspec.grid(linestyle=':', color='white')
        axspec.set_ylabel('Height [km]', fontsize=FONTS, fontweight=FONTW)
        axspec.tick_params(axis='y', which='both', right=True, top=True)
        plt.tight_layout()
        Plot.save_figure(fig_sp, name=f'rgspec-{fig_name}-a.png', dpi=200)

        from scipy import signal
        var_dBZ = h.lin2z(spectrogram_LIMRAD['var'])

        f, t, FFT = signal.stft(var_dBZ, fs=1.0, window='parzen', nperseg=256, noverlap=None, nfft=None,
                                detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=1)

        #_, FFT = signal.istft(var_dBZ, fs=1.0, window='parzen', nperseg=256, noverlap=None, nfft=None, time_axis=0, freq_axis=1)

        Amp = abs(FFT)
        Amp[:, :4, 1] = 0.0

        Phase = np.angle(FFT)
        kwargs = {'vmin': 1.e-2, 'vmax': 1.e0,
                  # 'z_converter': 'lin2z',
                  'colormap': 'viridis'}

        Amp_norm = (Amp[:, :, 1] - Amp[:, :, 1].min()) / (Amp[:, :, 1].max() - Amp[:, :, 1].min())
        fig_fft, _ = Plot.rangespectrogramFFT(f, spectrogram_LIMRAD['rg'], Amp_norm, **kwargs)
        plt.tight_layout()
        Plot.save_figure(fig_fft, name=f'rgspec-{fig_name}-b.png', dpi=200)

#        Phase_norm = (Phase[:, :, 1] - Phase[:, :, 1].min()) / (Phase[:, :, 1].max() - Phase[:, :, 1].min())
#        fig_fft, _ = Plot.rangespectrogramFFT(f, spectrogram_LIMRAD['rg'], Phase_norm, **kwargs)
#        plt.tight_layout()
#        Plot.save_figure(fig_fft, name=f'rgspec-{fig_name}-c.png', dpi=200)

        im1 = Image.open(f'rgspec-{fig_name}-a.png')
        im2 = Image.open(f'rgspec-{fig_name}-b.png')
        #im3 = Image.open(f'rgspec-{fig_name}-c.png')
        get_concat_h(im1, im2).save(f'rgfft-{fig_name}.png')

        print(its)
        dummy = 5
    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))
