#! /usr/bin/env python

from libwise import nputils
import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Plot separation from core with time

Usage: wise plot_sep_from_core NAME SCALES

NAME: the name of the saved result set to use.
SCALES: coma separated list of scales to plot.

Additional options:
--pa, -p: Additionally plot the features positional angle vs epoch
--fit, -f: fit each links with a linear fct
--num, -n: Annotate each links
--min-link-size=INT, -m INT: Filter out links with size < min_link_size (default=2)
'''


def main():
    sh.init(wise.get_version(), USAGE)

    pa = sh.get_opt_bool('pa', 'p')
    fit = sh.get_opt_bool('fit', 'f')
    num = sh.get_opt_bool('num', 'n')
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

    fit_fct = None
    if fit:
        fit_fct = nputils.LinearFct
    fit_result = wise.tasks.plot_separation_from_core(ctx, scales=scales, num=num,
                                                      min_link_size=min_link_size, fit_fct=fit_fct, pa=pa)

    if fit:
        for link, fit_fct in fit_result.items():
            print "Fit result for link %s: %.2f +- %.2f mas / year" % (link.get_id(), fit_fct.a, fit_fct.ea)


if __name__ == '__main__':
    main()
