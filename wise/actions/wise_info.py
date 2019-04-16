#! /usr/bin/env python

import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Give information on beam, pixel scales or velocity resolution

Usage: wise info FILES_OR_FILE_LIST

Additional options:
--velocity, -V: gives information on velocity resolution
'''


def main():
    sh.init(wise.get_version(), USAGE)

    velocity = sh.get_opt_bool('velocity', 'V')

    args = sh.get_args(min_nargs=1)

    config = actions.get_config(False)
    ctx = wise.AnalysisContext(config)
    actions.select_files(ctx, args)

    if velocity:
        wise.tasks.info_files_delta(ctx)
    else:
        wise.tasks.info_files(ctx)


if __name__ == '__main__':
    main()
