#!/home/sdig/anaconda3/bin/python
import datetime, sys
sys.path.append('../larda/')
sys.path.append('.')
import pyLARDA.helpers as h

def generate_multicase_trainingset(t_span, t_train, t_skip, path, cnt=0):
    """
        time_interval = ['20190318-0359', '20190318-1701']
        range_interval = [0, 10000]
        plot_dir = 'plots/20190318-01/'
    """
    training_span = datetime.timedelta(minutes=t_train)
    skip = datetime.timedelta(minutes=t_skip)
    h.change_dir(path)
    t_span_str = f'{t_span[0]:%Y%m%d}-{t_span[1]:%Y%m%d}'
    if t_span[1] == '0000': t_span[1] = '2359'

    filename = f'auto-trainingset-{t_span_str}-{cnt}.toml'

    with open(filename, 'w+') as f:
        cnt = 0
        t = t_span[0]
        f.write(f'\nt_train = {t_train:.2f}\n')
        f.write(f't_skip = {t_skip:.2f}\n')
        while t <= t_span[1]:
            f.write(f'\n[case.{t:%Y%m%d}-{cnt}]\n')
            f.write(f"    time_interval = ['{t:%Y%m%d-%H%M}', '{t+training_span:%Y%m%d-%H%M}']\n")
            f.write(f"    range_interval = [0, 12000]\n")
            f.write(f"    plot_dir = 'plots/{t:%Y%m%d}-{cnt}/'\n")
            t += skip
            cnt += 1

    print(f'Number of cases = {cnt}')
    print(f'Filename = {path}/{filename}')


# python generate_timeframe_toml.py dt_start=20181213 dt_end=20181213 t_train=15.0 t_skip=15.0


if __name__ == '__main__':
    # gather command line arguments
    method_name, args, kwargs = h._method_info_from_argv(sys.argv)

    all_files = False
    if all_files:

        def timerange(begin, end):
            result = []
            pivot = begin
            while pivot <= end:
                result.append([pivot, pivot + datetime.timedelta(hours=23, minutes=59)])
                pivot += datetime.timedelta(days=1)
            return result

        dt_begin = datetime.datetime.strptime(f'20181127 0000', '%Y%m%d %H%M')
        dt_end = datetime.datetime.strptime(f'20190928 2359', '%Y%m%d %H%M')

        dt_list = timerange(dt_begin, dt_end)
        for dt in dt_list:
            generate_multicase_trainingset(
                dt,
                float(kwargs['t_train']) if 't_train' in kwargs else 60.0,   # minutes,
                float(kwargs['t_skip']) if 't_skip' in kwargs else 60.0,   # minutes,
                '/home/sdig/code/larda3/voodoo/tomls/',
            )
    else:

        for i in range(24):

            if 'dt_start' in kwargs and 'dt_end' in kwargs:
                dt_begin = datetime.datetime.strptime(f'{kwargs["dt_start"]}', '%Y%m%d %H%M')
                dt_end = datetime.datetime.strptime(f'{kwargs["dt_end"]}', '%Y%m%d %H%M')
            else:
                hour = i
                dt_begin = datetime.datetime.strptime(f'20181127 0{hour}00', '%Y%m%d %H%M')
                dt_end = datetime.datetime.strptime(f'20190927 2359', '%Y%m%d %H%M')

            generate_multicase_trainingset(
                [dt_begin, dt_end],
                float(kwargs['t_train']) if 't_train' in kwargs else 60.0,   # minutes,
                float(kwargs['t_skip']) if 't_skip' in kwargs else 300.0,   # minutes,
                '/home/sdig/code/larda3/voodoo/tomls/5folds_all/',
                cnt=hour
            )
    print('done\n')
