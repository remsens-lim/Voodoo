#!/home/sdig/anaconda3/bin/python
import datetime, sys
from libVoodoo.Loader import preproc_ini, generate_multicase_trainingset
from libVoodoo.Utils import timerange
sys.path.append(preproc_ini['larda']['path'])
import pyLARDA.helpers as h

# python generate_timeframe_toml.py dt_start=20181213 dt_end=20181213 t_train=15.0 t_skip=15.0


if __name__ == '__main__':
    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)
    date_str = kwargs['date'] if 'date' in kwargs else '20190102'

    all_files = False
    if all_files:


        dt_begin = datetime.datetime.strptime(f'{date_str} 0000', '%Y%m%d %H%M')
        dt_end = datetime.datetime.strptime(f'{date_str} 2359', '%Y%m%d %H%M')

        dt_list = timerange(dt_begin, dt_end)
        for dt in dt_list:
            generate_multicase_trainingset(
                dt,
                float(kwargs['t_train']) if 't_train' in kwargs else 60.0,   # minutes,
                float(kwargs['t_skip']) if 't_skip' in kwargs else 60.0,   # minutes,
                '/home/sdig/code/Voodoo/tomls/',
            )
    else:

        for i in range(10):

            if 'dt_start' in kwargs and 'dt_end' in kwargs:
                hour = 0
                dt_begin = datetime.datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d %H%M')
                dt_end = datetime.datetime.strptime(f'{kwargs["dt_end"]}', '%Y%m%d %H%M')
            else:
                hour = i
                t_beg, t_end = 20181127, 20190927 # PA
                #t_beg, t_end = 20201216, 20211001 # LIM
                dt_begin = datetime.datetime.strptime(f'{t_beg} 0{hour}00', '%Y%m%d %H%M')
                dt_end = datetime.datetime.strptime(f'{t_end} 2359', '%Y%m%d %H%M')

            generate_multicase_trainingset(
                [dt_begin, dt_end],
                float(kwargs['t_train']) if 't_train' in kwargs else 60.0,   # minutes,
                float(kwargs['t_skip']) if 't_skip' in kwargs else 600.0,   # minutes,
                '/home/sdig/code/Voodoo/tomls/10folds_all/',
                ifold=hour
            )
    print('done\n')
