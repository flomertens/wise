#! /usr/bin/env python

import libwise
from libwise import nputils
import libwise.scriptshelper as sh

import wise
from wise.actions import actions
from wise import wiseutils
from libwise import plotutils

import matplotlib.pyplot as plt

USAGE = '''Plot all components trajectories on the reference map

Usage: wise_view_links_all_scales NAME SCALES

NAME: the name of the saved result set to use.
SCALES: coma separated list of scales to plot.

Additional options:
--min-link-size=INT, -m INT: Filter out links with size < min_link_size (default=2)
'''


def view_links_all_scales(ctx, scales=None, feature_filter=None, min_link_size=2, map_cmap='YlGnBu_r',
                          vector_width=6, title=True, save_filename=None, **kwargs):
    '''Plot all components trajectories on a map

    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    scales : int or list, optional
        Filter the results to scale or list of scales (in pixel)
    feature_filter : :class:`wise.features.FeatureFilter`, optional
        Filter the results
    min_link_size : int, optional
        Filter out links with size < min_link_size
    map_cmap : str, optional
        Colormap of the background map
    vector_width : int, optional
        Width of the displacement vectors
    title : bool, optional
    save_filename : str, optional
        If not None, the resulting maps will be saved in the project data dir
        with this filename.
    **kwargs :
        Additional arguments to pass to :func:`libwise.plot_links_map`


    .. _tags: task_matching
    '''
    if not ctx.result.has_match_result():
        print "No result found. Run match_all() first."
        return

    scales = wise.tasks._get_scales(scales, ctx.result.get_scales())

    if len(scales) == 0:
        print "No result found for this scales. Available scales:", ctx.result.get_scales()
        return

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)

    ms_match_results, link_builders = ctx.get_match_result()

    fig = plotutils.BaseCustomFigure()
    ax = fig.subplots()
    window = plotutils.BaseFigureWindow(figure=fig, extended_toolbar=False)

    colors = plotutils.ColorSelector()

    scale_str = ', '.join([str(projection.get_sky(scale)) for scale in scales])

    for scale in scales:
        link_builder = link_builders.get_scale(scale)
        links = link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size)

        wiseutils.plot_links_map(ax, ref_img, projection, links, color_style=colors.get(),
                                 map_cmap=map_cmap, vector_width=vector_width,
                                 link_id_label=False, **kwargs)

    if isinstance(title, bool) and title is True:
        ax.set_title('Displacement map at scale(s) %s' % scale_str)
    if isinstance(title, str):
        ax.set_title(title)

    # Save plot
    # ax.figure.savefig(os.path.join(ctx.get_data_dir(), save_filename))

    window.start()


def main():
    sh.init(libwise.get_version(), USAGE)

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

    view_links_all_scales(ctx, scales=scales, min_link_size=min_link_size)


if __name__ == '__main__':
    main()
