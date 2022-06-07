import datetime
import numpy
from numba import jit
from typing import List, Tuple, Dict
from matplotlib.colors import ListedColormap
import traceback
import sys
import numpy as np

# Voodoo cloud droplet likelyhood colorbar (viridis + grey below minimum value)
from matplotlib import cm
viridis = cm.get_cmap('viridis', 8)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[:1, :] = np.array([220/256, 220/256, 220/256, 1])
probability_cmap = ListedColormap(newcolors)


def dt_to_ts(dt: datetime.datetime) -> float:
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def get_unixtime(dt64: numpy.datetime64) -> int:
    return dt64.astype('datetime64[s]').astype('int')

def ts_to_dt(ts):
    """unix timestamp to dt"""
    return datetime.datetime.utcfromtimestamp(ts)

def decimalhour2unix(dt, time):
    return np.array([x*3600. + dt_to_ts(datetime.datetime(int(dt[:4]), int(dt[4:6]), int(dt[6:]), 0, 0, 0)) for x in time]).astype(int).astype( 'datetime64[s]')

def lin2z(array):
    """linear values to dB (for np.array or single number)"""
    return 10 * np.ma.log10(array)

def dh_to_ts(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh*3600))


def dh_to_dt(day_str, dh):
    """decimal hour to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    t0 = datetime.datetime.strptime(day_str, '%Y%m%d') - datetime.datetime(1970, 1, 1)
    return datetime.datetime.strptime(day_str, '%Y%m%d') + datetime.timedelta(seconds=float(dh*3600))

def z2lin(array):
    """dB to linear values (for np.array or single number)"""
    return 10 ** (array / 10.)



def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis

    Args:
        array (np.array): sorted array with values, list will be converted to 1D array
        value: value to find
    Returns:
        index
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i = i + 1
    return i


def interpolate_to_256(rpg_data, rpg_header, polarization='TotSpec'):
    from scipy.interpolate import interp1d
    
    rng_offsets = rpg_header['RngOffs']
    nts, nrg, nvel = rpg_data['TotSpec'].shape
    
    spec_new = np.zeros((nts, nrg, 256))
    for ichirp in range(len(rng_offsets)-1):
        
        ia = rng_offsets[ichirp]
        ib = rng_offsets[ichirp+1]
        nvel = rpg_header['SpecN'][ichirp]
        spec = rpg_data[polarization][:, ia:ib, :]
        
        if nvel == 256:
            spec_new[:, ia:ib, :] = spec
        else:
            old = rpg_header['velocity_vectors'][ichirp]
            f = interp1d(old, spec, axis=2, bounds_error=False, fill_value=-999., kind='linear')
            spec_new[:, ia:ib, :] = f(np.linspace(old[np.argmin(old)], old[np.argmax(old)], 256))

    return spec_new


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return np.array(lst3)

def set_intersection(mask0, mask1):
    mask_flt = np.where(~mask0.astype(np.bool).flatten())
    mask1_flt = np.where(~mask1.astype(np.bool).flatten())
    maskX_flt = intersection(mask_flt[0], mask1_flt[0])
    len_flt = len(maskX_flt)
    idx_list = []
    cnt = 0
    for iter, idx in enumerate(mask_flt[0]):
        if cnt >= len_flt: break
        if idx == maskX_flt[cnt]:
            idx_list.append(iter)
            cnt += 1

    return np.array(idx_list, dtype=int)

def reshape(input, mask):
    input_reshaped = np.zeros(mask.shape)
    cnt = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]: continue
            input_reshaped[i, j] = input[cnt]
            cnt += 1

    return input_reshaped

@jit(nopython=True, fastmath=True)
def isKthBitSet(n, k):
    if n & (1 << (k - 1)):
        return True
    else:
        return False

def read_cmd_line_args(argv=None):
    """Command-line -> method call arg processing.

    - positional args:
            a b -> method('a', 'b')
    - intifying args:
            a 123 -> method('a', 123)
    - json loading args:
            a '["pi", 3.14, null]' -> method('a', ['pi', 3.14, None])
    - keyword args:
            a foo=bar -> method('a', foo='bar')
    - using more of the above
            1234 'extras=["r2"]'  -> method(1234, extras=["r2"])

    @param argv {list} Command line arg list. Defaults to `sys.argv`.
    @returns (<method-name>, <args>, <kwargs>)

    Reference: http://code.activestate.com/recipes/577122-transform-command-line-arguments-to-args-and-kwarg/
    """
    import json
    import sys
    if argv is None:
        argv = sys.argv

    method_name, arg_strs = argv[0], argv[1:]
    args = []
    kwargs = {}
    for s in arg_strs:
        if s.count('=') == 1:
            key, value = s.split('=', 1)
        else:
            key, value = None, s
        try:
            value = json.loads(value)
        except ValueError:
            pass
        if key:
            kwargs[key] = value
        else:
            args.append(value)
    return method_name, args, kwargs


def traceback_error(time_span):
    exc_type, exc_value, exc_tb = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_tb)
    print(ValueError(f'Something went wrong with this interval: {time_span}'))


def interpolate2d(data, mask_thres=0.1, **kwargs):
    """interpolate timeheight data container

    Args:
        mask_thres (float, optional): threshold for the interpolated mask
        **new_time (np.array): new time axis
        **new_range (np.array): new range axis
        **method (str): if not given, use scipy.interpolate.RectBivariateSpline
        valid method arguments:
            'linear' - scipy.interpolate.interp2d
            'nearest' - scipy.interpolate.NearestNDInterpolator
            'rectbivar' (default) - scipy.interpolate.RectBivariateSpline
    """
    import scipy.interpolate as SI

    var = data['var'].copy()
    # var = h.fill_with(data['var'], data['mask'], data['var'][~data['mask']].min())
    # logger.debug('var min {}log_number_of_classes'.format(data['var'][~data['mask']].min()))
    method = kwargs['method'] if 'method' in kwargs else 'rectbivar'
    args_to_pass = {}
    if method == 'rectbivar':
        kx, ky = 1, 1
        interp_var = SI.RectBivariateSpline(data['ts'], data['rg'], var, kx=kx, ky=ky)
        interp_mask = SI.RectBivariateSpline(data['ts'], data['rg'], data['mask'].astype(np.float), kx=kx, ky=ky)
        args_to_pass["grid"] = True
    elif method == 'linear1d':
        points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
        interp_var = SI.LinearNDInterpolator(points, var.flatten(), fill_value=-999.0)
        interp_mask = SI.LinearNDInterpolator(points, (data['mask'].flatten()).astype(np.float))
    elif method == 'linear':
        interp_var = SI.interp2d(data['ts'], data['rg'], np.transpose(var), fill_value=np.nan)
        interp_mask = SI.interp2d(data['ts'], data['rg'], np.transpose(data['mask']).astype(np.float))
    elif method == 'nearest':
        points = np.array(list(zip(np.repeat(data['ts'], len(data['rg'])), np.tile(data['rg'], len(data['ts'])))))
        interp_var = SI.NearestNDInterpolator(points, var.flatten())
        interp_mask = SI.NearestNDInterpolator(points, (data['mask'].flatten()).astype(np.float))
    else:
        raise ValueError('Unknown Interpolation Method', method)

    new_time = data['ts'] if not 'new_time' in kwargs else kwargs['new_time']
    new_range = data['rg'] if not 'new_range' in kwargs else kwargs['new_range']

    if method in ["nearest", "linear1d"]:
        new_points = np.array(list(zip(np.repeat(new_time, len(new_range)), np.tile(new_range, len(new_time)))))
        new_var = interp_var(new_points).reshape((len(new_time), len(new_range)))
        new_mask = interp_mask(new_points).reshape((len(new_time), len(new_range)))
    else:
        new_var = interp_var(new_time, new_range, **args_to_pass)
        new_mask = interp_mask(new_time, new_range, **args_to_pass)

    # print('new_mask', new_mask)
    new_mask[new_mask > mask_thres] = 1
    new_mask[new_mask < mask_thres] = 0
    # print('new_mask', new_mask)

    # print(new_var.shape, new_var)
    # deepcopy to keep data immutable
    interp_data = {**data}

    interp_data['ts'] = new_time
    interp_data['rg'] = new_range
    interp_data['var'] = new_var if method in ['nearest', "linear1d", 'rectbivar'] else np.transpose(new_var)
    interp_data['mask'] = new_mask if method in ['nearest', "linear1d", 'rectbivar'] else np.transpose(new_mask)
    print("interpolated shape: time {} range {} var {} mask {}".format(
        new_time.shape, new_range.shape, new_var.shape, new_mask.shape))

    return interp_data


def load_training_mask(classes, status):
    """
    classes
    0: Clear sky\n
    1: Cloud droplets only\n
    2: Drizzle or rain\n
    3: Drizzle/rain & cloud droplets\n
    4: Ice\n
    5: Ice & supercooled droplets\n
    6: Melting ice\n
    7: Melting ice & cloud droplets\n
    8: Aerosol\n
    9: Insects\n
    10: Aerosol & insects";

    status
    0: Clear sky.\n
    1: Good radar and lidar echos.\n
    2: Good radar echo only.\n
    3: Radar echo, corrected for liquid attenuation.
    4: Lidar echo only.\nValue
    5: Radar echo, uncorrected for liquid attenuation.\nValue
    6: Radar ground clutter.\nValue
    7: Lidar clear-air molecular scattering.";
    """
    # create mask
    valid_samples = np.full(status.shape, False)
    valid_samples[status == 1] = True  # add good radar radar & lidar
    valid_samples[classes == 1] = True  # add cloud droplets only class
    # valid_samples[classes == 2] = True  # add drizzle/rain
    valid_samples[classes == 3] = True  # add cloud droplets + drizzle/rain
    valid_samples[classes == 5] = True  # add mixed-phase class pixel
    # valid_samples[classes == 6] = True  # add melting layer
    valid_samples[classes == 7] = True  # add melting layer + SCL class pixel

    # at last, remove lidar only pixel caused by adding cloud droplets only class
    valid_samples[status == 4] = False

    return ~valid_samples

