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

import numpy as np
import time
import pyLARDA.helpers as h
import pyLARDA.SpectraProcessing as sp

if __name__ == '__main__':

    spec_settings = {'rm_precip_ghost': False,   # reduces the domain (Nyquist velocitdy) by ± 2.5 [m/s], when signal > 0 [dBZ] within 200m above antenna
                     'do_despeckle3d': False,   # 3D convolution (5x5x5 window), removes single non-zero values, very slow!
                     'despeckle': False,         # 2D convolution (5x5 window), removes single non-zero values, very slow!
                     'NF': 6.0,                  # estimating noise in spectra noise_threshold = mean(noise) + noise_factor * std(noise)
                     'main_peak': True,         #
                     'filter_ghost_C1': False,   #
                     }

    PATH = '/home/sdig/code/larda3/scripts_Willi/new_s2m/'

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
        begin_dt = datetime.datetime(2019, 1, 10, 10, 15)
        end_dt = datetime.datetime(2019, 1, 10, 11, 57)
        begin_dt = datetime.datetime(2019, 4, 28, 3, 1)
        end_dt = datetime.datetime(2019, 4, 28, 4, 15)

    TIME_SPAN_ = [begin_dt, end_dt]
    NF = float(kwargs['NF']) if 'NF' in kwargs else 6.0

    limrad94_Zspec = sp.load_spectra_rpgfmcw94(larda, TIME_SPAN_, **spec_settings)
    limrad94_mom = sp.spectra2moments(limrad94_Zspec, larda.connectors['LIMRAD94'].system_info['params'], **spec_settings)

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
    plot_radar_moments = True
    plot_range = [0, 12000]
    fig_size = [9, 6]

    if plot_radar_moments:
        # create folder for subfolders if it doesn't exist already
        h.change_dir(PATH)

        ZE = limrad94_mom['Ze']
        ZE['var_unit'] = 'dBZ'
        ZE['colormap'] = 'jet'
        ZE['var_lims'] = [-50, 20]
        fig, _ = pyLARDA.Transformations.plot_timeheight(ZE, fig_size=fig_size, range_interval=plot_range,
                                                         z_converter='lin2z', contour=contour, rg_converter=True)
        fig_name = 'limrad_Ze' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)
        print('fig saved:', fig_name)
        sys.exit(99)

        VEL = limrad94_mom['VEL']
        VEL['var_lims'] = [-4, 2]
        fig, _ = pyLARDA.Transformations.plot_timeheight(VEL, fig_size=fig_size, range_interval=plot_range,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_VEL' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        sw = limrad94_mom['sw']
        sw['var_lims'] = [0, 0.5]
        fig, _ = pyLARDA.Transformations.plot_timeheight(sw, fig_size=fig_size, range_interval=plot_range,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_sw' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        skew = limrad94_mom['skew']
        skew['var_lims'] = [-0.5, 1]
        fig, _ = pyLARDA.Transformations.plot_timeheight(skew, fig_size=fig_size, range_interval=plot_range,
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
                                                         range_interval=plot_range,
                                                         # contour=contour,
                                                         rg_converter=True)
        fig_name = 'limrad_skew_smoothed' + f'{begin_dt:%Y%m%d_%H%M%S_}{end_dt:%H%M%S}.png'
        fig.tight_layout()
        fig.savefig(fig_name, dpi=300, transparent=False)

        kurt = limrad94_mom['kurt']
        kurt['var_lims'] = [1, 6]
        fig, _ = pyLARDA.Transformations.plot_timeheight(kurt, fig_size=fig_size, range_interval=plot_range,
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

    print('total elapsed time = {:.3f} sec.'.format(time.time() - start_time))
