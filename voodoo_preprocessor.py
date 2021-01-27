#!/home/sdig/anaconda3/bin/python3
"""
Short description:
    Creating a *.zarr file containing input features and labels for the voodoo neural network.
"""

import logging
import sys
from datetime import timedelta, datetime

from libVoodoo.Utils import traceback_error, read_cmd_line_args
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

    _DEFAULT_CHANNELS = 12
    _DEFAULT_DOPPBINS = 256

    VOODOO_PATH = '/home/sdig/code/voodoo/'
    ANN_INI_FILE = 'HP_12chdp2.toml'

    DATA_PATH = f'{VOODOO_PATH}/data/{ANN_INI_FILE[:-5]}/'

    method_name, args, kwargs = read_cmd_line_args(sys.argv)

    t_train = float(kwargs["t_train"]) if 't_train' in kwargs else 60.0

    if 'dt_start' in kwargs:
        dt_begin = datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=t_train)
        TIME_SPAN_ = [dt_begin, dt_end]
    else:
        #debug case
        dt_begin = datetime.strptime('20190801-0500', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=30.0)
        TIME_SPAN_ = [dt_begin, dt_end]

    try:
        features, targets, multitargets, masked, classes, ts, rg = features_from_nc(
            time_span=TIME_SPAN_,
            voodoo_path=VOODOO_PATH,
            data_path=DATA_PATH,
            system=kwargs['radar'] if 'radar' in kwargs else 'limrad94',
            cloudnet=kwargs['cnet'] if 'cnet' in kwargs else 'CLOUDNETpy94',
            save=kwargs['save'] if 'save' in kwargs else True,
            n_channels=_DEFAULT_CHANNELS,
            ann_settings_file=ANN_INI_FILE,
            site=kwargs['site'] if 'site' in kwargs else 'lacros_dacapo_gpu',
            dual_polarization=True,
        )

    except Exception:
        traceback_error(TIME_SPAN_)
