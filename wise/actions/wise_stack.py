#! /usr/bin/env python

import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Stack images

Usage: wise stack FILES_OR_FILE_LIST [-o OUTPUT_FITS]

Additional options:
--output FILENAME, -o FILENAME: output file name (default=stack_img.fits)
--nsigma NSIGMA, -n NSIGMA: clip background below NSIGMA level (default=0)
--nsigma_connected, -c: Keep only the brightest isolated structure
'''


def main():
    sh.init(wise.get_version(), USAGE)

    nsigma = float(sh.get_opt_value('nsigma', 'n', default=0))
    nsigma_connected = sh.get_opt_bool('nsigma_connected', 'c')
    output_filename = sh.get_opt_value('output', 'o', default='stack_img.fits')

    args = sh.get_args(min_nargs=1)

    config = actions.get_config(False)
    ctx = wise.AnalysisContext(config)
    actions.select_files(ctx, args)

    stack_img = ctx.build_stack_image(preprocess=False, nsigma=nsigma, nsigma_connected=nsigma_connected)
    stack_img.save(output_filename)

    print "Stacked images save to %s" % output_filename


if __name__ == '__main__':
    main()
