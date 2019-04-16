#! /usr/bin/env python

import os
import wise
from libwise import imgutils
from libwise.app import PolyRegionEditor
import libwise.scriptshelper as sh

USAGE = '''View and create DS9 type region files

Usage: wise region IMG [REG_FILE]'''


def main():
    sh.init(wise.get_version(), USAGE)
    args = sh.get_args(min_nargs=1)

    img = imgutils.guess_and_open(args[0])
    editor = PolyRegionEditor.PolyRegionEditor(img, current_folder=os.path.dirname(img.file))

    if len(args) >= 2:
        poly_region = imgutils.PolyRegion.from_file(args[1], img.get_coordinate_system())
        editor.load_poly_region(poly_region)

    editor.start()


if __name__ == '__main__':
    main()
