#! /usr/bin/env python

import re
import sys

import libwise
import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Run the Segmented wavelet decomposition

Usage: wise detect FILES_OR_FILE_LIST

Arguments can be either the files to process, or a text file listing the files
to process.
'''


def main():
    sh.init(wise.get_version(), USAGE)

    args = sh.get_args(min_nargs=1)

    config = actions.get_config(True)
    ctx = wise.AnalysisContext(config)
    actions.select_files(ctx, args)

    if len(ctx.files) == 0:
        sys.exit(0)

    wise.tasks.detection_all(ctx)

    str2vector = lambda s: [float(k) for k in re.findall("[-0-9.]+", s)]
    check = lambda s: len(wise.tasks._get_scales(str2vector(s), ctx.result.get_scales())) > 0

    txt = "View scales (available: %s) (press enter to leave):" % ctx.result.get_scales()

    while True:
        scales = sh.ask(txt, check_fct=check, default=[])
        if len(scales) == 0:
            break
        wise.tasks.view_wds(ctx, scales=str2vector(scales))

    save = sh.askbool("Save the result ?")

    if save:
        name = str(sh.ask("Name (default=result):", default='result'))
        wise.tasks.save(ctx, name)


if __name__ == '__main__':
    main()
