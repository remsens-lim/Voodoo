import datetime, sys
sys.path.append('/home/sdig/code/larda3/larda/')
sys.path.append('.')
from larda.pyLARDA.helpers import change_dir, _method_info_from_argv

def generate_multicase_trainingset(t_span, t_train, t_skip, path):
    """
        time_interval = ['20190318-0359', '20190318-1701']
        range_interval = [0, 10000]
        plot_dir = 'plots/20190318-01/'
    """
    training_span = datetime.timedelta(minutes=t_train)
    skip = datetime.timedelta(minutes=t_skip)
    change_dir(f'{path}')
    t_span_str = f'{t_span[0]:%Y%m%d}-{t_span[1]:%Y%m%d}'

    with open(f'auto-trainingset-{t_span_str}.toml', 'w+') as f:
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



# gather command line arguments
method_name, args, kwargs = _method_info_from_argv(sys.argv)

if 'dt_start' in kwargs and 'dt_end' in kwargs:
    dt_begin = datetime.datetime.strptime(f'{kwargs["dt_start"]} 0000', '%Y%m%d %H%M')
    dt_end   = datetime.datetime.strptime(f'{kwargs["dt_end"]} 2359', '%Y%m%d %H%M')
    time_span = [dt_begin, dt_end]
else:
    raise ValueError('Wrong dt_begin or dt_end')

t_train = float(kwargs['t_train']) if 't_train' in kwargs else 15.0   # minutes
t_skip = float(kwargs['t_skip']) if 't_skip' in kwargs else 15.0   # minutes

generate_multicase_trainingset(time_span, t_train, t_skip, '/home/sdig/code/larda3/voodoo/tomls/')
print('done\n')
