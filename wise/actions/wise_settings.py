#! /usr/bin/env python

import os
import re

from libwise import imgutils, uiutils
import libwise.scriptshelper as sh

import wise
import actions

import astropy.units as u

USAGE = '''Set and get WISE configuration.

Possible actions are:

wise settings set SECTION.OPTION=VALUE [SECTION.OPTION=VALUE]
wise settings get/show [SECTION[.OPTION]]
wise settings doc [SECTION[.OPTION]]
wise settings restore CONFIG_FILE

SECTION is one of data, finder or matcher
'''


def get_section(section_name, config):
    if section_name not in ['data', 'finder', 'matcher']:
        print "Error: SECTION should be one of data, finder or matcher\n"
        sh.usage(True)
    return getattr(config, section_name)


def check_option(section, option):
    if not section.has(option):
        print "Error: option %s of %s does not exist\n" % (option, section.get_title())
        sh.usage(True)


def delta_range_filter_handler(config):
    current_filter = config.matcher.delta_range_filter
    append = False
    print "Current delta range filter:", current_filter
    if current_filter is not None:
        append = sh.askbool("Do you want to add a new delta range filter to the existing one?")

    region = None
    if sh.askbool("Restrict delta range filter to a region?"):
        region_filename = uiutils.open_file()
        try:
            region = imgutils.Region(region_filename)
        except Exception:
            print "Warning: opening region file failed"
    if region is not None:
        print "Delta range filter for region:", region.get_name()

    str2vector = lambda s: [float(k) for k in re.findall("[-0-9.]+", s)]
    check_vector = lambda s: len(str2vector(s)) == 2

    unit = u.Unit(sh.ask("Velocity unit:"))
    direction = str2vector(sh.ask("Direction vector (default=[1,0]):", check_fct=check_vector, default="1,0"))
    vx = str2vector(sh.ask("Velocity range in X direction:", check_fct=check_vector))
    vy = str2vector(sh.ask("Velocity range in Y direction:", check_fct=check_vector))

    range_filter = wise.DeltaRangeFilter(vxrange=vx, vyrange=vy, unit=unit, pix_limit=4, x_dir=direction)
    if region is not None:
        range_filter = wise.DeltaRegionFilter(wise.RegionFilter(region), range_filter)

    if append:
        range_filter = current_filter & range_filter

    print "Setting delta_range_filter to:", range_filter
    config.matcher.delta_range_filter = range_filter


def main():
    sh.init(wise.get_version(), USAGE)

    args = sh.get_args(min_nargs=0)

    config = actions.get_config(True)

    if len(args) == 0 or args[0] in ['get', 'show']:
        if len(args) < 2:
            print config.values()
        elif '.' in args[1]:
            section_name, option = args[1].split('.', 2)
            section = get_section(section_name, config)
            check_option(section, option)
            print '%s: %s' % (args[1], section.get(option, encode=True))
        else:
            section = get_section(args[1], config)
            print section.values()

    elif args[0] == 'set':
        for arg in args[1:]:
            if arg == 'matcher.delta_range_filter':
                delta_range_filter_handler(config)
                continue
            try:
                full_option, value = arg.split('=', 2)
                section_name, option = full_option.split('.', 2)
            except Exception:
                print "Error: setting option should be of the form SECTION.OPTION=VALUE\n"
                sh.usage(True)
            section = get_section(section_name, config)
            check_option(section, option)
            print "Setting %s to %s" % (full_option, value)
            section.set(option, value, decode=True)
        if len(args[1:]) > 0:
            config.to_file(actions.CONFIG_FILE)
            print "Configuration saved"

    elif args[0] == 'doc':
        if len(args) == 1:
            print config.doc()
        elif '.' in args[1]:
            section_name, option = args[1].split('.', 2)
            section = get_section(section_name, config)
            check_option(section, option)
            print "Documentation of %s: %s" % (args[1], section.get_doc(option))
        else:
            section = get_section(args[1], config)
            print section.doc()

    elif args[0] == 'restore':
        if len(args) != 2 or not os.path.isfile(args[1]):
            print "Error: an existing CONFIG_FILE is necessary\n"
            sh.usage(True)
        try:
            config.from_file(args[1])
            config.to_file(actions.CONFIG_FILE)
        except Exception:
            print "Error: restoring configuration from %s failed\n" % args[1]
            raise
        print "Configuration restored from %s" % args[1]
    else:
        print "Error: Action not possible\n"
        sh.usage(True)


if __name__ == '__main__':
    main()
