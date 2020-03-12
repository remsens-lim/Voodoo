"""
This module contains routines for loading and preprocessing cloud radar and lidar data.

"""

import logging
import time
import toml
from scipy import signal
import numpy as np
from tqdm.auto import tqdm

from larda.pyLARDA.SpectraProcessing import spectra2moments, load_spectra_rpgfmcw94
import pyLARDA.Transformations as tr
import pyLARDA.helpers as h

import libVoodoo.Plot as Plot

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__author__ = "Willi Schimmel"
__copyright__ = "Copyright 2019, The Voodoo Project"
__credits__ = ["Willi Schimmel", "Teresa Vogl", "Martin Radenz"]
__license__ = "MIT"
__version__ = "0.2.0"
__maintainer__ = "Willi Schimmel"
__email__ = "willi.schimmel@uni-leipzig.de"
__status__ = "Prototype"


def load_case_list(path, case_name):
    # gather command line arguments
    config_case_studies = toml.load(path)
    return config_case_studies['case'][case_name]


def scaling(strat='none'):
    def ident(x):
        return x

    def norm(x, min_, max_):
        x[x < min_] = min_
        x[x > max_] = max_
        return (x - min_) / max(1.e-15, max_ - min_)

    if strat == 'normalize':
        return norm
    else:
        return ident


def ldr2cdr(ldr):
    ldr = np.array(ldr)
    return np.log10(-2.0 * ldr / (ldr - 1.0)) / (np.log10(2) + np.log10(5))


def cdr2ldr(cdr):
    cdr = np.array(cdr)
    return np.power(10.0, cdr) / (2 + np.power(10.0, cdr))


def replace_fill_value(data, newfill):
    """
    Replaces the fill value of an spectrum container by their time and range specific mean noise level.
    Args:
        data (numpy.array) : 3D spectrum array (time, range, velocity)
        newfill (numpy.array) : 2D new fill values for 3rd dimension (velocity)

    Return:
        var (numpy.array) : spectrum with mean noise
    """

    n_ts, n_rg, _ = data.shape
    var = data.copy()
    masked = np.all(data <= 0.0, axis=2)

    for iT in range(n_ts):
        for iR in range(n_rg):
            if masked[iT, iR]:
                var[iT, iR, :] = newfill[iT, iR]
            else:
                var[iT, iR, var[iT, iR, :] <= 0.0] = newfill[iT, iR]
    return var


def load_training_mask(classes, status):
    # create mask
    valid_samples = np.full(status['var'].shape, False)
    valid_samples[status['var'] == 1] = True  # add good radar radar & lidar
    # valid_samples[status['var'] == 2]  = True   # add good radar only
    valid_samples[classes['var'] == 5] = True  # add mixed-phase class pixel
    # valid_samples[classes['var'] == 6] = True   # add melting layer class pixel
    # valid_samples[classes['var'] == 7] = True   # add melting layer + SCL class pixel
    valid_samples[classes['var'] == 1] = True  # add cloud droplets only class

    # at last, remove lidar only pixel caused by adding cloud droplets only class
    valid_samples[status['var'] == 3] = False
    return ~valid_samples

def load_data(spectra, classes, **feature_info):
    """
     For orientation

    " status
    \nValue 0: Clear sky.
    \nValue 1: Good radar and lidar echos.
    \nValue 2: Good radar echo only.
    \nValue 3: Radar echo, corrected for liquid attenuation.
    \nValue 4: Lidar echo only.
    \nValue 5: Radar echo, uncorrected for liquid attenuation.
    \nValue 6: Radar ground clutter.
    \nValue 7: Lidar clear-air molecular scattering.";

    " classes
    \nValue 0: Clear sky.
    \nValue 1: Cloud liquid droplets only.
    \nValue 2: Drizzle or rain.
    \nValue 3: Drizzle or rain coexisting with cloud liquid droplets.
    \nValue 4: Ice particles.
    \nValue 5: Ice coexisting with supercooled liquid droplets.
    \nValue 6: Melting ice particles.
    \nValue 7: Melting ice particles coexisting with cloud liquid droplets.
    \nValue 8: Aerosol particles, no cloud or precipitation.
    \nValue 9: Insects, no cloud or precipitation.
    \nValue 10: Aerosol coexisting with insects, no cloud or precipitation.";
    """
    t0 = time.time()

    if len(spectra['var'].shape) == 3:
        (n_time, n_range, n_Dbins), n_chan = spectra['var'].shape, 1
        spectra_3d = spectra['var'].reshape((n_time, n_range, n_Dbins, n_chan)).astype('float32')
        spectra_mask = spectra['mask'].reshape((n_time, n_range, n_Dbins, n_chan))
    elif len(spectra['var'].shape) == 4:
        n_time, n_range, n_Dbins, n_chan = spectra['var'].shape
        spectra_3d = spectra['var'].astype('float32')
        spectra_mask = spectra['mask']
    else:
        raise ValueError('Spectra has wrong dimension!', spectra['var'].shape)

    masked = np.all(np.any(spectra_mask, axis=3), axis=2)
    spec_params = feature_info['VSpec']
    cwt_params = feature_info['cwt']

    assert len(cwt_params['scales']) == 3, 'The list of scaling parameters 3 values!'
    n_cwt_scales = int(cwt_params['scales'][2])
    scales = np.linspace(cwt_params['scales'][0], cwt_params['scales'][1], n_cwt_scales)

    quick_check = feature_info['quick_check'] if 'quick_check' in feature_info else False
    if quick_check:
        ZE = np.sum(np.mean(spectra_3d, axis=3), axis=2)
        ZE = h.put_in_container(ZE, classes)  # , **kwargs)
        ZE['dimlabel'] = ['time', 'range']
        ZE['name'] = 'pseudoZe'
        ZE['joints'] = ''
        ZE['rg_unit'] = ''
        ZE['colormap'] = 'jet'
        # ZE['paraminfo'] = dict(ZE['paraminfo'][0])
        ZE['system'] = 'LIMRAD94'
        # ZE['var_lims'] = [ZE['var'].min(), ZE['var'].max()]
        ZE['var_lims'] = [-60, 20]
        ZE['var_unit'] = 'dBZ'
        ZE['mask'] = masked

        fig, _ = tr.plot_timeheight(ZE, var_converter='lin2z', title='bla inside wavelett')  # , **plot_settings)
        Plot.save_figure(fig, name=f'limrad_pseudoZe.png', dpi=200)

    spectra_lims = np.array(spec_params['var_lims'])
    # convert to logarithmic units
    if 'var_converter' in spec_params and 'lin2z' in spec_params['var_converter']:
        spectra_3d = h.get_converter_array('lin2z')[0](spectra_3d)
        spectra_lims = h.get_converter_array('lin2z')[0](spec_params['var_lims'])

    class_names = {
        '0': 'Clear sky.',
        '1': 'Cloud liquid droplets only.',
        '2': 'Drizzle or rain.',
        '3': 'Drizzle or rain coexisting with cloud liquid droplets.',
        '4': 'Ice particles.',
        '5': 'Ice coexisting with supercooled liquid droplets.',
        '6': 'Melting ice particles.',
        '7': 'Melting ice particles coexisting with cloud liquid droplets.',
        '8': 'Insects or ground clutter',
    }
    classes_var = classes['var'].astype(np.int8)

    # load scaling functions
    spectra_scaler = scaling(strat='normalize')
    cwt_scaler = scaling(strat='normalize')
    cnt = 0
    feature_list = []
    target_labels = []
    print('\nFeature Extraction......')
    for iT in tqdm(range(n_time)):
        for iR in range(n_range):
            if masked[iT, iR]: continue  # skip masked values

            spc_matrix = np.zeros((n_Dbins, n_chan))
            cwt_matrix = np.zeros((n_Dbins, n_cwt_scales, n_chan))
            for iCh in range(n_chan):
                spc_matrix[:, iCh] = spectra_scaler(spectra_3d[iT, iR, :, iCh], spectra_lims[0], spectra_lims[1])
                cwtmatr = signal.cwt(spc_matrix[:, iCh], signal.ricker, scales)
                cwt_matrix[:, :, iCh] = cwt_scaler(cwtmatr, cwt_params['var_lims'][0], cwt_params['var_lims'][1]).T

            feature_list.append(cwt_matrix)

            # one hot encoding
            if classes['var'][iT, iR] in [8, 9, 10]:
                target_labels.append([0, 0, 0, 0, 0, 0, 0, 0, 1])  # Insects or ground clutter.
            elif classes['var'][iT, iR] == 7:
                target_labels.append([0, 0, 0, 0, 0, 0, 0, 1, 0])  # Melting ice particles coexisting with cloud liquid droplets.
            elif classes['var'][iT, iR] == 6:
                target_labels.append([0, 0, 0, 0, 0, 0, 1, 0, 0])  # Melting ice particles.
            elif classes['var'][iT, iR] == 5:
                target_labels.append([0, 0, 0, 0, 0, 1, 0, 0, 0])  # Ice coexisting with supercooled liquid droplets.
            elif classes['var'][iT, iR] == 4:
                target_labels.append([0, 0, 0, 0, 1, 0, 0, 0, 0])  # Ice particles.
            elif classes['var'][iT, iR] == 3:
                target_labels.append([0, 0, 0, 1, 0, 0, 0, 0, 0])  # Drizzle or rain coexisting with cloud liquid droplets.
            elif classes['var'][iT, iR] == 2:
                target_labels.append([0, 0, 1, 0, 0, 0, 0, 0, 0])  # Drizzle or rain.
            elif classes['var'][iT, iR] == 1:
                target_labels.append([0, 1, 0, 0, 0, 0, 0, 0, 0])  # Cloud liquid droplets only
            else:
                target_labels.append([1, 0, 0, 0, 0, 0, 0, 0, 0])  # Clear-sky
            cnt += 1

            if 'plot' in cwt_params and cwt_params['plot']:
                velocity_lims = [-6, 6]
                for iCh in range(n_chan):
                    # show spectra, normalized spectra and wavlet transformation
                    fig, ax = Plot.spectra_wavelettransform(
                        spectra['vel'], spc_matrix[:, iCh], cwt_matrix[:, :, iCh].T,
                        ts=spectra['ts'][iT],
                        rg=spectra['rg'][iR],
                        colormap='cloudnet_jet',
                        x_lims=velocity_lims,
                        v_lims=cwt_params['var_lims'],
                        scales=scales,
                        hydroclass=class_names[str(classes_var[iT, iR])],
                        fig_size=[7, 4]
                    )
                    # fig, (top_ax, bottom_left_ax, bottom_right_ax) = Plot.spectra_wavelettransform2(vel_list, spcij_scaled, cwt_params['scales'])
                    Plot.save_figure(fig, name=f'limrad_cwt_{str(cnt).zfill(4)}_iT-iR-iCh_{str(iT).zfill(4)}-{str(iR).zfill(4)}-{iCh}.png', dpi=300)

    Plot.print_elapsed_time(t0, 'Added continuous wavelet transformation to features (single-core), elapsed time = ')

    FEATURES = np.array(feature_list, dtype=np.float16)
    LABELS = np.array(target_labels, dtype=np.float16)

    print(f'min/max value in features = {np.min(FEATURES)},  maximum = {np.max(FEATURES)}')
    print(f'min/max value in targets  = {np.min(LABELS)},  maximum = {np.max(LABELS)}')

    return FEATURES, LABELS, masked


def load_radar_data(larda, begin_dt, end_dt, **kwargs):
    """ This routine loads the radar spectra from an RPG cloud radar and caluclates the radar moments.

    Args:
        - larda (larda object) : the class for reading NetCDF files
        - begin_dt (datetime) : datetime object containing the start time
        - end_dt (datetime) : datetime object containing the end time

    **Kwargs:
        - rm_precip_ghost (bool) : removing speckle ghost echos which occur in all chirps caused by precipitation, default: False
        - rm_curtain_ghost (bool) : removing gcurtain host echos which occur in chirp==1, caused by strong signals between 2 and 4 km altitude, default: False
        - do_despeckle (bool) : removes isolated pixel in moments, using a 5x5 pixel window, where at least 80% of neighbouring pixel must be nonzeros,
        default: False
        - do_despeckle3d (bool) : removes isolated pixel i spectrum, using a 5x5x5 pixel window, whereat least 95% of neighbouring pixel must be nonzeros,
        default: 0.95
        - estimate_noise (bool) : remove the noise from the Doppler spectra, default: False
        - NF (float) : calculating the noise factor, mean noise level, left and right edge of a signal, if this value is larger than 0.0. The noise_threshold =
        mean_noise + NF * std(mean_noise), default: 6.0
        - main_peak (bool) : if True, calculate moments only for the main peak in the spectrum, default: True
        - fill_value (float) : non-signal values will be set to this value, default: -999.0

    Returns:
        - radar_data (dict) : containing radar Doppler spectra and radar moments, where non-signal pixels == fill_value, the pectra and Ze are
        stored in linear units [mm6 m-3]
    """

    rm_prcp_ghst = kwargs['rm_precip_ghost'] if 'rm_precip_ghost' in kwargs else False
    rm_crtn_ghst = kwargs['rm_curtain_ghost'] if 'rm_curtain_ghost' in kwargs else False
    dspckl = kwargs['do_despeckle'] if 'do_despeckle' in kwargs else False
    dspckl3d = kwargs['do_despeckle3d'] if 'do_despeckle3d' in kwargs else 95.
    est_noise = kwargs['estimate_noise'] if 'estimate_noise' in kwargs else False
    NF = kwargs['noise_factor'] if 'noise_factor' in kwargs else 6.0
    main_peak = kwargs['main_peak'] if 'main_peak' in kwargs else True
    fill_value = kwargs['fill_value'] if 'fill_value' in kwargs else -999.0

    t0 = time.time()

    time_span = [begin_dt, end_dt]

    radar_spectra = load_spectra_rpgfmcw94(
        larda,
        time_span,
        rm_precip_ghost=rm_prcp_ghst,
        do_despeckle3d=dspckl3d,
        estimate_noise=est_noise,
        noise_factor=NF)

    radar_moments = spectra2moments(
        radar_spectra,
        larda.connectors['LIMRAD94'].system_info['params'],
        despeckle=dspckl,
        main_peak=main_peak,
        filter_ghost_C1=rm_crtn_ghst
    )

    Plot.print_elapsed_time(t0, f'Reading spectra + moment calculation, elapsed time = ')
    return {'spectra': radar_spectra, 'moments': radar_moments}
