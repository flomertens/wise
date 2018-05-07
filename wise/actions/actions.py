import os
import glob

from libwise import imgutils

import wise

CONFIG_FILE = 'wise_config'


def get_config(create_if_none=False):
    config = wise.AnalysisConfiguration()
    if os.path.exists(CONFIG_FILE):
        config.from_file(CONFIG_FILE)
    elif create_if_none:
        config.to_file(CONFIG_FILE)

    return config


def select_files(ctx, args):
    if imgutils.is_fits(args[0]) or imgutils.is_img(args[0]):
        ctx.select_files(args)
    else:
        with open(args[0]) as file:
            ctx.select_files([k.strip() for k in file.readlines()])


def load(name):
    config = get_config(False)
    ext = '.set.dat'

    if config.data.data_dir is None:
        data_dir = '.'
    else:
        data_dir = config.data.data_dir

    all_results_set = glob.glob(os.path.join(data_dir, '*', '*' + ext))
    all_results_dirs = map(os.path.dirname, all_results_set)
    all_results_names = map(os.path.basename, all_results_dirs)

    if name not in all_results_names:
        return None

    idx = all_results_names.index(name)

    config_file = os.path.join(all_results_dirs[idx], '%s.config' % name)
    config.from_file(config_file)

    ctx = wise.AnalysisContext(config)
    wise.tasks.load(ctx, name)

    return ctx
