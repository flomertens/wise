#! /usr/bin/env python

from libwise import nputils
import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Plot all features on a distance from core vs epoch

Usage: wise plot_features NAME SCALES

NAME: the name of the saved result set to use.
SCALES: coma separated list of scales to plot.

Additional options:
--pa, -p: Additionally plot the features positional angle vs epoch
'''


def main():
    sh.init(wise.get_version(), USAGE)

    pa = sh.get_opt_bool('pa', 'p')

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
        print "Error: invalid scales. Available scales: %s" % ctx.result.get_scales()
        sh.usage(True)

    print "Plotting features from scales %s" % scales
    wise.tasks.plot_all_features(ctx, scales, pa=pa)


if __name__ == '__main__':
    main()
