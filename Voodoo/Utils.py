import datetime
from numba import jit
from typing import List, Tuple, Dict
from matplotlib.colors import ListedColormap
import numpy as np

# Voodoo cloud droplet likelyhood colorbar (viridis + grey below minimum value)
from matplotlib import cm
viridis = cm.get_cmap('viridis', 8)
newcolors = viridis(np.linspace(0, 1, 256))
newcolors[:1, :] = np.array([220/256, 220/256, 220/256, 1])
probability_cmap = ListedColormap(newcolors)


def dt_to_ts(dt):
    """datetime to unix timestamp"""
    # return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
    return (dt - datetime.datetime(1970, 1, 1)).total_seconds()

def get_unixtime(dt64):
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

