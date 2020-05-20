import subprocess, re
import pyLARDA.helpers as h

from pprint import pprint

from jinja2 import Template
import sys
import traceback
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.addHandler(logging.StreamHandler())


def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def write_ann_config_file(**kwargs):
    """
    Creates at folder and saves a matplotlib figure.

    Args:
        fig (matplotlib figure): figure to save as png

    Keyword Args:
        dpi (int): dots per inch
        name (string): name of the png
        path (string): path where the png is stored

    Returns:    0

    """
    path = kwargs['path'] if 'path' in kwargs else ''
    name = kwargs['name'] if 'name' in kwargs else 'no-name.cfg'
    if len(path) > 0: h.change_dir(path)

    import json
    with open(f"{name}", 'wt', encoding='utf8') as out:
        json.dump(kwargs, out, sort_keys=True, indent=4, ensure_ascii=False)
    print(f'Saved ann configure file :: {name}')
    return 0

def read_ann_config_file(**kwargs):
    """
    Creates at folder and saves a matplotlib figure.

    Args:
        fig (matplotlib figure): figure to save as png

    Keyword Args:
        dpi (int): dots per inch
        name (string): name of the png
        path (string): path where the png is stored

    Returns:    0

    """
    path = kwargs['path'] if 'path' in kwargs else ''
    name = kwargs['name'] if 'name' in kwargs else 'no-name.cfg'
    if len(path) > 0: h.change_dir(path)

    import json
    with open(f"{name}", 'r', encoding='utf8') as json_file:
        data = json.load(json_file)
    print(f'Loaded ann configure file :: {name}')
    return data

def make_html_overview(template_loc, case_study_info, png_names):
    print('case_config', case_study_info)
    print('savenames', png_names.keys())

    with open(f'{template_loc}/static_template.html.jinja2') as file_:
        template = Template(file_.read())

        with open(case_study_info['plot_dir'] + 'overview.html', 'w') as f:
            f.write(
                template.render(
                    png_names=png_names,
                    case_study_info=case_study_info,
                )
            )


        """
        <!--
        {% for key, value in data.items() %}
            <li>{{ key }}: {{ value['file_history'] }}</li>
        {% endfor %}
        -->
        """

def get_explorer_link(campaign, time_interval, range_interval, params):
    s = "http://larda.tropos.de/larda3/explorer/{}?interval={}-{}%2C{}-{}&params={}".format(
        campaign, h.dt_to_ts(time_interval[0]), h.dt_to_ts(time_interval[1]),
        *range_interval, ",".join(params))
    return s

def traceback_error(time_span):
    exc_type, exc_value, exc_tb = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_tb)
    logger.error(ValueError(f'Something went wrong with this interval: {time_span}'))