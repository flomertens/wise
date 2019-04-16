#! /usr/bin/env python

from libwise import imgutils
import libwise.scriptshelper as sh

import wise
import actions

USAGE = '''Simple image viewer

Usage: wise view FILES

Additional options:
--no-crop, -n: do not crop images according to the data.roi_coords configuration
--no-align: do not align images according to the data.core_offset_filename file
--show-mask, -m: overplot the images with the mask, if it exist
--reg-file=FILE: -r FILE: overplot region, multiple option possible
'''


def main():
    sh.init(wise.get_version(), USAGE)

    preprocess = not sh.get_opt_bool('no-crop', 'n')
    align = not sh.get_opt_bool('no-align', None)
    show_mask = sh.get_opt_bool('show-mask', 'm')

    region_files = sh.get_opt_value('reg-file', 'r', multiple=True)
    regions = []
    try:
        regions = [imgutils.Region(file) for file in region_files]
    except Exception:
        print "Error: failed to read a region file"
        sh.usage(True)

    args = sh.get_args(min_nargs=1)

    config = actions.get_config(False)
    ctx = wise.AnalysisContext(config)
    actions.select_files(ctx, args)

    wise.tasks.view_all(ctx, preprocess=preprocess, show_regions=regions,
                        show_mask=show_mask, align=align)


if __name__ == '__main__':
    main()
