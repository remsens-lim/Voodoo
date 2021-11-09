#!/home/sdig/anaconda3/bin/python3
"""
Short description:
    Creating a *.zarr file containing input features and labels for the voodoo neural network.
"""

import logging
import sys, os
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

    #ANN_INI_FILE = 'HP_12chdp2.toml'
    ANN_INI_FILE = 'VnetSettings-1.toml'

    VOODOO_PATH = os.getcwd()
    DATA_PATH = f'{VOODOO_PATH}/data/Vnet_6ch_noliqext/hourly/'

    method_name, args, kwargs = read_cmd_line_args(sys.argv)

    t_train = float(kwargs["t_train"]) if 't_train' in kwargs else 60.0
    system = kwargs['radar'] if 'radar' in kwargs else 'limrad94'
    cnet = kwargs['cnet'] if 'cnet' in kwargs else 'CLOUDNETpy94'
    save = kwargs['save'] if 'save' in kwargs else True
    site = kwargs['site'] if 'site' in kwargs else 'leipzig_gpu' # 'lacros_dacapo_gpu'  #
    dpol = kwargs['dpol'] if 'dpol' in kwargs else True
    n_ch = int(kwargs['n_ch']) if 'n_ch' in kwargs else 6


    if 'dt_start' in kwargs:
        dt_begin = datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=t_train)
        TIME_SPAN_ = [dt_begin, dt_end]
    else:
        dt_begin = datetime.strptime('20210801-0600', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=60.0)
        TIME_SPAN_ = [dt_begin, dt_end]

    try:
        _, _ = features_from_nc(
            time_span=TIME_SPAN_,
            voodoo_path=VOODOO_PATH,
            data_path=DATA_PATH,
            system=system,
            cloudnet=cnet,
            save=save,
            n_channels=n_ch,
            ann_settings_file=ANN_INI_FILE,
            site=site,
            dual_polarization=dpol,
            build_lists=False if site == 'lacros_dacapo_gpu' else True,
        )

    except Exception:
        traceback_error(TIME_SPAN_)
