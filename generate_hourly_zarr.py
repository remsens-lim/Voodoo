#!/home/sdig/anaconda3/bin/python3
"""
Short description:
    Creating a *.zarr file containing input features and labels for the voodoo neural network.
"""

import logging
import sys, os
import time
from datetime import timedelta, datetime

from libVoodoo.Utils import traceback_error, read_cmd_line_args, change_dir
from libVoodoo.Loader import features_from_nc, logger

logger.setLevel(logging.WARNING)

########################################################################################################################
#
#   _______ _______ _____ __   _       _____   ______  _____   ______  ______ _______ _______
#   |  |  | |_____|   |   | \  |      |_____] |_____/ |     | |  ____ |_____/ |_____| |  |  |
#   |  |  | |     | __|__ |  \_|      |       |    \_ |_____| |_____| |    \_ |     | |  |  |
#
#
if __name__ == '__main__':

    #DATA_PATH = f'/media/sdig/LACROS/cloudnet/data/punta-arenas/calibrated/voodoo/hourly-cn133/'
    DATA_PATH = f'/media/sdig/leipzig/cloudnet/calibrated/voodoo/hourly-cn133/'
    larda_path = '/home/sdig/code/larda3/larda/'

    method_name, args, kwargs = read_cmd_line_args(sys.argv)

    system = 'limrad94'
    cloudnet = 'CLOUDNETpy94'
    chunk = float(kwargs["chunk"]) if 'chunk' in kwargs else 60.0
    site = str(kwargs['site']) if 'site' in kwargs else 'leipzig_gpu'  # 'lacros_dacapo_gpu'

    
    feature_settings = {
        'var_lims': [1.0e-5, 1.0e2],    # linear units mm6/m3
        'var_converter': 'lin2z',       # conversion to dBZ
        'scaling': 'normalize',         # between 0 and 1 using var_lims
        'channels': 6,                   # number of time steps in time spectrogram
        'n_stride': 1,
    }
    
    if 'dt_start' in kwargs:
        dt_begin = datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=chunk)
        TIME_SPAN_ = [dt_begin, dt_end]
    else:
        dt_begin = datetime.strptime('20201230-1400', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=60.0)
        TIME_SPAN_ = [dt_begin, dt_end]


    savednd = 'x'
    start_time = time.time()
    try:
        ds = features_from_nc(
            time_span=TIME_SPAN_,
            system=system,
            cloudnet=cloudnet,
            site=site,
            build_lists=True,
            larda_path=larda_path,
            feature_settings=feature_settings,
        )

        change_dir(DATA_PATH)
        ds.to_zarr(store=f'{dt_begin:%Y%m%d_%H%M}-{dt_end:%H%M}_{system}-{cloudnet}-ND.zarr', mode='w', compute=True)
        savednd = '√'



    except Exception:
        traceback_error(TIME_SPAN_)

    finally:
        logger.critical(
            f'DONE :: {TIME_SPAN_[0]:%A %d. %B %Y - %H:%M:%S} to {TIME_SPAN_[1]:%H:%M:%S} zarr files generated, elapsed time = '
            f'{timedelta(seconds=int(time.time() - start_time))} min feature/label-data [{savednd}]'
        )
