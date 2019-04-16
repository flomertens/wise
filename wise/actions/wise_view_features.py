#! /usr/bin/env python

from libwise import nputils
import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Plot all features location on the reference image.

Usage: wise view_features NAME SCALES

NAME: the name of the saved result set to use.
SCALES: coma separated list of scales to plot.
'''


def main():
    sh.init(wise.get_version(), USAGE)

    args = sh.get_args(min_nargs=2)
    name = args[0]
    ctx = actions.load(name)

    if ctx is None:
        print "No results saved with name %s" % name
        sh.usage(True)

    scales = args[1]

    try:
        scales = nputils.str2floatlist(scales)
    except Exception:
        print "Error: invalid scales. Availables scales: %s" % ctx.result.get_scales()
        sh.usage(True)

    print "Plotting features from scales %s" % scales
    wise.tasks.view_all_features(ctx, scales)


if __name__ == '__main__':
    main()
