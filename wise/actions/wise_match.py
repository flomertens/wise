#! /usr/bin/env python

import sys

from libwise import nputils
import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Run the matching procedure

Usage: wise match FILES_OR_FILE_LIST

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

    wise.tasks.match_all(ctx)

    check = lambda s: nputils.is_str_number(s) and float(s) in ctx.result.get_scales()

    txt = "View scales (available: %s) (press enter to leave):" % ctx.result.get_scales()

    while True:
        scale = float(sh.ask(txt, check_fct=check, default=0))
        if scale == 0:
            break
        wise.tasks.view_displacements(ctx, scale)

    save = sh.askbool("Save the result ?")

    if save:
        name = str(sh.ask("Name (default=result):", default='result'))
        wise.tasks.save(ctx, name)


if __name__ == '__main__':
    main()
