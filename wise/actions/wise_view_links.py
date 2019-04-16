#! /usr/bin/env python

from libwise import nputils
import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Plot all components trajectories on the reference map

Usage: wise view_links NAME SCALES

NAME: the name of the saved result set to use.
SCALES: coma separated list of scales to plot.

Additional options:
--min-link-size=INT, -m INT: Filter out links with size < min_link_size (default=2)
'''


def main():
    sh.init(wise.get_version(), USAGE)

    min_link_size = sh.get_opt_value('min-link-size', 'm', default=2)
    sh.check(min_link_size, nputils.is_str_number, "min-link-size must be an integer")
    min_link_size = float(min_link_size)

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

    wise.tasks.view_links(ctx, scales=scales, min_link_size=min_link_size)


if __name__ == '__main__':
    main()
