"""
Short description:
    Creating a *.zarr file containing input features and labels for the voodoo neural network.
"""

import sys
from datetime import timedelta, datetime
from Voodoo.Loader import features_from_nc
from Voodoo.Utils import traceback_error, read_cmd_line_args



PARAMETER = PARAMETER_punta

if __name__ == '__main__':

    method_name, args, kwargs = read_cmd_line_args(sys.argv)

    PARAMETER.update(kwargs.items())

    if 'dt_start' in kwargs:
        dt_begin = datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=60.0)
        TIME_SPAN_ = [dt_begin, dt_end]
    else:
        dt_begin = datetime.strptime('20191201-0500', '%Y%m%d-%H%M')
        dt_end = dt_begin + timedelta(minutes=60.0)
        TIME_SPAN_ = [dt_begin, dt_end]

    try:
        _, _ = features_from_nc(time_span=TIME_SPAN_, build_lists=True, **PARAMETER)

    except Exception:
        traceback_error(TIME_SPAN_)
