#! /usr/bin/env python

from libwise import imgutils, nputils
import libwise.scriptshelper as sh

import wise

USAGE = '''Build a list of files and output the listing in OUTPUT_FILE.

Usage: wise select_files FILES [-o OUTPUT_FILE]

Additional options:
--output FILENAME, -o FILENAME: output file name (default=files)
--start-date=START, -s START: filter files with date < START
--end-date=END -d END: filter files with date > END
--filter-date=DATE, -f DATE: filter files with date == DATE

All dates must be formated as: YYYY-MM-DD
'''


def get_date(date, option):
    if date is None:
        return None
    date = nputils.guess_date(date, ["%Y-%m-%d", "%Y_%m_%d"])
    if date is None:
        print "Error: invalid date format for the option %s" % option
        sh.usage(True)
    return date


def main():
    sh.init(wise.get_version(), USAGE)

    start_date = get_date(sh.get_opt_value('start-date', 's'), 'start-date')
    end_date = get_date(sh.get_opt_value('end-date', 'e'), 'end-date')
    filter_dates = [get_date(k, 'filter-date') for k in sh.get_opt_value('filter-date', 'f', multiple=True)]
    output_filename = sh.get_opt_value('output', 'o', default='files')

    args = sh.get_args(min_nargs=1)

    files = imgutils.fast_sorted_fits(args, start_date=start_date,
                                      end_date=end_date, filter_dates=filter_dates)

    print "Outputing %s files in '%s'" % (len(files), output_filename)

    with open(output_filename, 'w') as f:
        f.write("\n".join(files) + "\n")


if __name__ == '__main__':
    main()
