import logging
import datetime

import numpy as np

import wds
import matcher
import features as wfeatures

from libwise import plotutils, nputils, imgutils

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axisartist.grid_finder import MaxNLocator

from scipy.optimize import curve_fit
from scipy.ndimage import measurements

import astropy.units as u
import astropy.constants as const

unit_c = u.core.Unit("c", const.c, doc="light speed")
p2i = imgutils.p2i
logger = logging.getLogger(__name__)


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


def imshow_segmented_image(ax, segmented_image, projection=None, title=True, beam=True, num=False,
                           bg=None, mode='com', **kwargs):
    """Display the segmented image on the axes.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    segmented_image : :class:`wise.wds.SegmentedImages`
    projection : :class:`libwise.imgutils.Projection`, optional
    title : bool, optional
        Whatever to optionally display a title. Default is True.
    beam : bool, optional
        Whatever to optionally display the beam of the image. Default is True.
    num : bool, optional
        Whatever to optionally annotate the segments with there ids. Default is False.
    bg : :class:`libwise.imgutils.Image`, optional
        Diplay bg as background image instead of the on from segmented_image.
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    **kwargs :
        Additional arguments to be passed to :func:`libwise.plotutils.imshow_image`.

    
    .. _tags: plt_detection
    """
    if bg is None:
        bg = segmented_image.get_img()

    prj = plotutils.imshow_image(ax, bg, projection=projection, title=False, beam=beam, **kwargs)

    if title is True:
        title = "Segmented image"
        if isinstance(segmented_image, wds.AbstractScale):
            title += " at scale %s" % prj.get_sky(segmented_image.get_scale())
        if len(segmented_image.get_img().get_title()) > 0:
            title += "\n%s" % segmented_image.get_img().get_title()
        ax.set_title(title)

    plot_segments_contour(ax, segmented_image)

    plot_features(ax, segmented_image, num=num, c=plotutils.orange, mode=mode)


def plot_segments_contour(ax, segmented_image, **kwargs):
    """Display the segments contour of a segmented_image on a map.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`.
    segmented_image : :class:`wise.wds.SegmentedImages`.
    **kwargs :
        Additional arguments to be passed to :func:`libwise.plotutils.plot_mask`.

    
    .. _tags: plt_detection
    """
    for segment in segmented_image:
        [x0, y0, x1, y1] = segment.get_cropped_index()
        croped = segment.get_mask()[max(x0 - 2, 0):x1 + 3, max(y0 - 2, 0):y1 + 3]
        x0, y0 = [x0 - 2, y0 - 2]
        x1, y1 = [x1 + 2, y1 + 2]
        plotutils.plot_mask(ax, imgutils.Mask(croped), extent=[y0, y1, x0, x1], **kwargs)


def plot_features(ax, features, mode='com', color_fct=None, num=False, num_offset=[3, -3], **kwargs):
    """Plots the segments location and optionally the segments ids on a map.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    features : :class:`wise.features.FeaturesGroup`
        Any FeaturesGroup objects that store Features.
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    color_fct : a function, optional
        If set, it should be a function that take a feature as argument and return a color.
    num : bool, optional
        Whatever to optionally annotate the segments with there ids.
    num_offset : list, optional
        Offset in pixel for the id.
    **kwargs : 
        Additional arguments to be passed to :func:`libwise.plotutils.plot_coords`

    
    .. _tags: plt_detection
    """
    assert isinstance(features, wfeatures.FeaturesGroup)

    if color_fct is not None:
        kargs['c'] = [color_fct(f) for f in features]

    if features.size() > 0:
        plotutils.plot_coords(ax, p2i(features.get_coords(mode=mode)), **kwargs)

    if num and isinstance(features, wds.SegmentedImages):
        for segment in features.get_features():
            y, x = segment.get_center_of_mass()
            ax.text(x - num_offset[0], y - num_offset[1], "%d" % segment.get_segmentid(), size='small')


def add_features_tooltip(stack, ax, features, projection=None, epoch=False, tol=4):
    for feature in features:
        coord = feature.get_coord()
        if projection is not None:
            x, y = projection.p2s(p2i(coord))
        else:
            y, x = coord
        text = 'Feature x:%s y:%s i:%s' % (x, y, feature.get_intensity())

        if epoch and isinstance(feature, wfeatures.ImageFeature):
            text += ' e:%s' % feature.get_epoch().strftime("%Y-%m-%d")

        stack.add_tooltip(ax, coord, text, tol=tol)


def plot_inner_features(ax, segments, **kwargs):
    for segment in segments:
        if segment.has_inner_features():
            inner_features = segment.get_inner_features()
            if inner_features.size() > 1:
                plot_features(ax, inner_features, **kwargs)


def plot_delta_info_1dim(ax, x, deltax, input_delta=None, axis=0):
    # Deprecated
    ax.plot(x, deltax, ls='', marker='+', c=plotutils.blue, label="Matched delta")
    if input_delta is None:
        return [], [], 0, 0, 0

    expectedx = input_delta.get_fct(axis)(x)
    sigx = max(input_delta.get_sig(axis), 1)

    chi2 = (((deltax - expectedx) / sigx) ** 2).sum()
    dof = float(len(x) - 2 - 1)
    rms = (deltax - expectedx).std()

    xerror, up, down = input_delta.error_range(x.min(), x.max(), 1)

    ax.plot(xerror, input_delta.get_fct(axis)(xerror), c=plotutils.green, alpha=0.5)
    # ax.plot(xerror, np.gradient(input_delta.get_fct(axis)(xerror)), c=plotutils.red)
    plotutils.plot_error_span(ax, xerror, up[axis], down[axis], alpha=0.5)

    return expectedx, sigx, chi2, dof, rms


def plot_delta_info(stack, delta_info, input_delta=None, plot_error=True):
    # Deprecated
    def do_plot(fig):
        nrows = 2

        ax = fig.subplots(nrows=nrows, sharex=True)

        group = delta_info.get_features(wfeatures.DeltaInformation.DELTA_MATCH)

        if group.size() == 0:
            return

        x = np.array([k.get_coord()[1] for k in group])
        delta = delta_info.get_deltas(group)
        deltay, deltax = delta.T

        expectedx, sigx, chi2x, dof, rmsx = plot_delta_info_1dim(ax[0], x, deltax, input_delta, axis=0)
        expectedy, sigy, chi2y, dof, rmsy = plot_delta_info_1dim(ax[1], x, deltay, input_delta, axis=1)

        if input_delta is not None:
            residual = nputils.l2norm(np.array([deltax, deltay]).T - np.array([expectedx, expectedy]).T)

            print "Chi2 X:", chi2x / dof, "RMS:", rmsx
            print "Chi2 Y:", chi2y / dof, "RMS:", rmsy
            print "Full RMS:", residual.std()

        ax[0].set_ylabel("$\Delta_x (px)$")

        ax[1].set_xlabel("$X (px)$")
        ax[1].set_ylabel("$\Delta_y (px)$")

        ax[1].set_xlim(min(x) - 5, max(x) + 5)

    stack.add_replayable_figure("Delta Mesure", do_plot)


def plot_displacement_vector(ax, delta_info, mode='com', color_fct=None, 
                             flag=wfeatures.DeltaInformation.DELTA_MATCH, **kwargs):
    """Display displacements vectors represented as arrows.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    delta_info : :class:`wise.features.DeltaInformation`
        An object containing the displacements to display.
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    color_fct : TYPE, optional
        If set, it should be a function that take a feature as argument and return a color.
    flag : Attribute, optional
        Default is DeltaInformation.DELTA_MATCH.
    **kwargs: 
        Additional arguments to be passed to :func:`matplotlib.pyltot.Arrow`.

    
    .. _tags: plt_matching
    """
    features = delta_info.get_features(flag=flag)
    for i, feature in enumerate(features):
        coord = feature.get_coord(mode=mode)
        delta = delta_info.get_delta(feature)
        if color_fct is not None:
            kwargs['fc'] = color_fct(feature)
        plotutils.checkargs(kwargs, 'zorder', 3)
        plotutils.checkargs(kwargs, 'width', 3.5)
        patch = plt.Arrow(coord[1], coord[0], delta[1], delta[0], **kwargs)
        ax.add_patch(patch)


def plot_velocity_vector(ax, delta_info, projection, ang_vel_unit, pix_per_unit, 
                         mode='com', color_fct=None, 
                         flag=wfeatures.DeltaInformation.DELTA_MATCH, **kwargs):
    features = delta_info.get_features(flag=flag)
    for i, feature in enumerate(features):
        coord = feature.get_coord(mode=mode)
        delta = delta_info.get_delta(feature)
        ang_vel = delta_info.get_full_delta(feature).get_angular_velocity(projection)
        ang_vel_pix = ang_vel.to(ang_vel_unit).value * pix_per_unit
        ang_vel_vect_pix = ang_vel_pix * delta / nputils.l2norm(delta)
        if color_fct is not None:
            kwargs['fc'] = color_fct(feature)
        plotutils.checkargs(kwargs, 'zorder', 3)
        plotutils.checkargs(kwargs, 'width', 3.5)
        patch = plt.Arrow(coord[1], coord[0], ang_vel_vect_pix[1], ang_vel_vect_pix[0], **kwargs)
        ax.add_patch(patch)


def plot_displacements(ax, features1, features2, delta_info, num=False, projection=None, mode='com',
                bg=None, beam=True, cmap=None, **kwargs):
    """Display displacements of features on a map.
    
    If bg is not set and features1 and features2 are both SegmentedImage, a two color map,
    one color for the segments of each SegmentedImage, will be used as bg.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    features1 : :class:`wise.features.FeaturesGroup`
        The features of the first epoch.
    features2 : :class:`wise.features.FeaturesGroup`
        The features of the second epoch.
    delta_info : :class:`wise.features.DeltaInformation`
        An object containing the displacements information.
    num : bool, optional
        Whatever to optionally annotate the segments with there ids.
    projection : :class:`libwise.imgutils.Projection`, optional
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    bg : :class:`libwise.imgutils.Image`, optional
        The image to be used as background map.
    beam : bool, optional
        Whatever to optionally display the beam of the image. Default is True.
    cmap : :class:`matplotlib.cm.ColorMap` or str, optional
        A color map for the background map.
    **kwargs: 
        Additional arguments to be passed to :func:`plot_displacement_vector`

    
    .. _tags: plt_matching
    """
    alpha = 0.8

    if isinstance(features1, wds.SegmentedImages):
        projection = plotutils.get_projection(features1.get_img(), projection)
        if bg is None:
            data = ((features1.get_labels() > 1) * 2 + (features2.get_labels() > 1) * 4).astype(np.int8)
            bg = imgutils.Image.from_image(features1.get_img(), data)
            alpha = 0.5
    else:
        assert bg is not None
        assert projection is not None
        contour = False

    cmap = plotutils.get_cmap(cmap)
    plotutils.imshow_image(ax, bg, projection=projection, beam=beam, title=False,
                           alpha=0.8, cmap=cmap)

    if num:
        plot_segmentid(ax, features1)

    plot_features(ax, features1, mode=mode, c=plotutils.blue, alpha=0.8)
    plot_features(ax, features2, mode=mode, c=plotutils.orange, alpha=0.8)

    plot_displacement_vector(ax, delta_info, mode=mode, fc='k', ec='k', **kwargs)

    plot_segments_contour(ax, features1, colors=plotutils.dblue)
    plot_segments_contour(ax, features2, colors=plotutils.red)


def plot_features_dfc(ax, projection, features, mode='com', **kwargs):
    """Plot features distance from core vs epoch.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    projection : :class:`libwise.imgutils.Projection`
    features : :class:`wise.features.FeaturesGroup`
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    **kwargs : 
        Additional arguments to be passed to :func:`matplotlib.pyplot.plot`.

    
    .. _tags: plt_detection
    """
    dfcs, epochs = zip(*[(projection.dfc(p2i(k.get_coord(mode=mode))), k.get_epoch()) for k in features])
    if isinstance(features, wds.AbstractScale):
        scale = features.get_scale()
        ax.plot(epochs, dfcs, ls='', marker='o', markersize=1.5 * scale, zorder=-scale, alpha=0.8, **kwargs)
    else:
        ax.plot(epochs, dfcs, ls='', marker='o', alpha=0.8, **kwargs)


def plot_features_pa(ax, projection, features, mode='com', **kwargs):
    """Plot features PA vs epoch.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    projection : :class:`libwise.imgutils.Projection`
    features : :class:`wise.features.FeaturesGroup`
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    **kwargs : 
        Additional arguments to be passed to :func:`matplotlib.pyplot.plot`.

    
    .. _tags: plt_detection
    """
    pas, epochs = zip(*[(projection.pa(p2i(k.get_coord(mode=mode))), k.get_epoch()) for k in features])
    if isinstance(features, wds.AbstractScale):
        scale = features.get_scale()
        ax.plot(epochs, pas, ls='', marker='o', markersize=1.5 * scale, zorder=-scale, alpha=0.8, **kwargs)
    else:
        ax.plot(epochs, pas, ls='', marker='o', alpha=0.8, **kwargs)


def plot_ms_set_map(ax, img, ms_set, projection, mode='com', color_style='date', colorbar_setting=None, 
                    map_cmap='jet', **kwargs):
    """Display all features of ms_set on a map.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    img : Image
        An image to be used as background map.
    ms_set : :class:`wise.wds.MultiScaleImageSet`
        An object containing all the features to be displayed.
    projection : :class:`libwise.imgutils.Projection`
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    color_style : str, optional
        'scale': display one color per scale. 'date': color correspond to the epoch.
    colorbar_setting : :class:`libwise.ColorbarSetting, optional
        Settings for the color bar if color_style is 'date'.
    map_cmap : :class:`matplotlib.cm.ColorMap` or str, optional
    **kwargs : 
        Additional arguments to be passed to :func:`libwise.plotutils.imshow_image`.

    
    .. _tags: plt_detection
    """
    if colorbar_setting is None and color_style == 'date':
        colorbar_setting = plotutils.ColorbarSetting(ticks_locator=mdates.AutoDateLocator(),
                                                     ticks_formatter=mdates.DateFormatter('%m/%y'))
    
    epochs = ms_set.get_epochs()
    intensities = [k.get_intensity() for k in ms_set.features_iter()]
    int_norm = plotutils.Normalize(min(intensities), max(intensities))
    marker_select = plotutils.MarkerSelector()

    epochs_map = plotutils.build_epochs_mappable(epochs, colorbar_setting.get_cmap())

    if img is not None:
        plotutils.imshow_image(ax, img, projection=projection, title=False, cmap=plotutils.get_cmap(map_cmap), **kwargs)

    color_fct = None
    if color_style == 'date':
        color_fct = lambda f: epochs_map.to_rgba(f.get_epoch())
        colorbar_setting.add_colorbar(epochs_map, ax)
    elif color_style is 'scale':
        pass

    for ms_segments in ms_set:
        for segments in ms_segments:
            # if segments.get_scale() != 2:
            #     continue
            s = 10 * segments.get_scale()
            marker = marker_select.get(segments.get_scale())
            # s = 500 * int_norm(segments.get_intensities())
            # s = 200
            plot_features(ax, segments, mode=mode, color_fct=color_fct, s=s, alpha=0.7, marker=marker)


def plot_link_builder_dfc(ax, projection, link_builder, min_link_size=2, mode='com', num=False,
                          feature_filter=None, date_filter=None, fit_fct=None, 
                          fit_min_size=3, **kwargs):
    # Deprecated
    filter1 = wfeatures.DateFilter.from_filter_fct(date_filter)
    if feature_filter is not None:
        filter1 &= feature_filter
    links = link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size)

    plot_links_dfc(ax, projection, links, mode=mode, num=num, **kwargs)
    if fit_fct:
        plot_links_dfc_fit(ax, projection, links, fit_fct, fit_min_size=fit_min_size, mode=mode)


def plot_links_dfc(ax, projection, links, mode='com', num=False, num_bbox=None, **kwargs):
    """Plot features link on a distance from core vs epoch plot.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    projection : :class:`libwise.imgutils.Projection`
    links : a list of :class:`wise.matcher.FeaturesLink`
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    num : bool, optional
        Whatever to optionally annotate the links with there ids.
    num_bbox : dict, optional
    **kwargs : 
        Additional arguments to be passed to :func:`matplotlib.pyplot.plot`.

    
    .. _tags: plt_matching
    """
    dfc_fct_coord = lambda coord: nputils.l2norm(projection.p2s(p2i(coord)))
    dfc_fct = lambda feature: dfc_fct_coord(feature.get_coord(mode))

    for link in links:
        delta_info = link.get_delta_info()

        if num is True:
            if num_bbox is None:
                num_bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor=link.get_color(), lw=1.5)
            # ax.text(link.get_first_epoch(), dfc_fct(link.first()), "%s, %s" % (link.get_id(), link.size()))            
            ax.text(link.get_first_epoch(), dfc_fct(link.first()), link.get_id(), bbox=num_bbox, zorder=200, size='small')

        if 'c' not in kwargs:
            line_color = link.get_color()
        else:
            line_color = kwargs['c']
        # plotutils.checkargs(kwargs, 'lw', 2.5)

        for epoch, related in link.get_relations():
            link_feature = link.get(epoch)
            related_feature = related.get(epoch)
            ax.plot([epoch, epoch], [dfc_fct(link_feature), dfc_fct(related_feature)], color='k', marker='+')
        for feature1, feature2 in nputils.pairwise(link.get_features()):
            epoch1 = feature1.get_epoch()
            epoch2 = feature2.get_epoch()
            delta = delta_info.get_delta(feature1)
            dfc1 = dfc_fct(feature1)
            dfc2 = dfc_fct(feature2)
            # dfc2 = dfc_fct_coord(feature1.get_coord(mode) + delta)
            ax.plot([epoch1, epoch2], [dfc1, dfc2], color=line_color, **kwargs)
            ax.plot([epoch1], dfc1, ls='', marker='+', color='k')
        # ax.plot([epoch2], dfc2, ls='', marker='+', color='k')


def plot_links_dfc_fit(ax, projection, links, fit_fct, fit_min_size=3, mode='com', **kwargs):
    """Plot features link fit on a distance from core vs epoch plot.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    projection : :class:`libwise.imgutils.Projection`
    links : a list of :class:`wise.matcher.FeaturesLink`
    fit_fct : :class:`libwise.nputils.AbstractFct`
        The function that will be used for the fit.
    fit_min_size : int, optional
        Fit only links with size >= fit_min_size.
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    **kwargs : 
        Additional arguments to be passed to :func:`matplotlib.pyplot.plot`.

    
    .. _tags: plt_matching
    """
    results = dict()
    for link in links:
        if fit_fct is not None and link.size() >= fit_min_size:
            fct, epochs, dfcs = link.fit_dfc(fit_fct, projection, coord_mode=mode)
            results[link] = fct
            v = fct.a
            v_error = fct.error(epochs, dfcs)
            label = 'v = %.3f +- %.3f %s' % (v, v_error, (projection.unit / u.year).to_string())
            # datetime_epochs = [nputils.mjd_to_datetime(mjd) for mjd in epochs]
            ax.plot(link.get_epochs(), fct(epochs), label=label, c=link.get_color(), **kwargs)

    return results


def plot_links_pa(ax, projection, links, mode='com', **kwargs):
    """Plot features link on a PA vs epoch plot.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    projection : :class:`libwise.imgutils.Projection`
    links : a list of :class:`wise.matcher.FeaturesLink`
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    **kwargs : 
        Additional arguments to be passed to :func:`matplotlib.pyplot.plot`

    
    .. _tags: plt_matching
    """
    for link in links:
        pas = [np.degrees(projection.pa(p2i(k.get_coord(mode=mode)))) for k in link.get_features()]
        epochs = link.get_epochs()

        ax.plot(epochs, pas, marker='+', color=link.get_color(), **kwargs)


def plot_links_snr(ax, projection, links, mode='com', **kwargs):
    """Plot features link on a SNR vs epoch plot.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    projection : :class:`libwise.imgutils.Projection`
    links : a list of :class:`wise.matcher.FeaturesLink`
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    **kwargs : 
        Additional arguments to be passed to :func:`matplotlib.pyplot.plot`.

    
    .. _tags: plt_matching
    """
    for link in links:
        # snrs = [k.get_segmented_image().get_feature_snr(k) for k in link.get_features()]
        # snrs = [k.get_intensity() for k in link.get_features()]
        snrs = [k.get_snr() for k in link.get_features()]
        epochs = link.get_epochs()

        ax.plot(epochs, snrs, marker='+', color=link.get_color(), **kwargs)


def plot_velocity_map(ax, stack_img, projection, link_builder, min_link_size=2, title=False,
                      feature_filter=None, color_style='link', mode='com', colorbar_setting=None,
                      dfc_axis=False, dfc_axis_teta=None, map_cmap='jet', vector_width=4, 
                      dfc_axis_pos=(1, 3), link_id_label=False, **kwargs):
    #DEPRECATED
    #   dfc_axis_teta in radian
    #   dfc_axis_pos is (nth_coord, value)
    links = link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size)

    plot_links_map(ax, stack_img, projection, links, color_style=color_style, 
                               mode=mode, colorbar_setting=colorbar_setting, map_cmap=map_cmap,
                               vector_width=vector_width, link_id_label=link_id_label, 
                               num_bbox=None, **kwargs)

    if dfc_axis:
        axis = plotutils.add_rotated_axis(ax, projection, dfc_axis_teta, axis_pos=dfc_axis_pos)
        axis.label.set_text("$z_{obs} (mas)$")

    scale = projection.get_sky(link_builder.get_scale())
    if isinstance(title, bool) and title is True:
        ax.set_title(ax.get_title() + '\nVelocity map at scale %s.' % scale)
    if isinstance(title, str):
        ax.set_title(title)


def plot_links_map(ax, img, projection, links, color_style='link', mode='com', colorbar_setting=None,
                     map_cmap='jet', vector_width=4, link_id_label=False, num_bbox=None, 
                     ang_vel_arrows=False, ang_vel_unit=u.mas / u.year, pix_per_ang_vel_unit=1, **kwargs):
    """Display features links on a map.
    
    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
    img : Image
        An image to be used as background map.
    projection : :class:`libwise.imgutils.Projection`
    links : a list of :class:`wise.matcher.FeaturesLink`
    color_style : str, optional
        'link': one color per link. 
        'date': map the features epochs to a color.
        any color string: use one color for each displacements vectors.
        function: a function that take a feature as argument and return a color.
    mode : str, optional
        Coord mode for the location of the features: 'lm', 'com' or 'cos'.
    colorbar_setting : ColorBar, optional
        Settings for the color bar if color_style is 'date'.
    map_cmap : :class:`matplotlib.cm.ColorMap` or str, optional
    vector_width : int, optional
        Width of the displacements vector arrows. Default is 4.
    link_id_label : bool, optional
        Annotate the links with there ids.
    num_bbox : dict, optional
    **kwargs: 
        Additional arguments to be passed to :func:`libwise.plotutils.imshow_image`.

    
    .. _tags: plt_matching
    """
    color_fct = None
    if color_style == 'date':
        all_epochs = matcher.get_all_epochs(links)
        epochs_map = plotutils.build_epochs_mappable(all_epochs, colorbar_setting.get_cmap())
        if colorbar_setting is None:
            colorbar_setting = plotutils.ColorbarSetting(ticks_locator=mdates.AutoDateLocator(),
                                                         ticks_formatter=mdates.DateFormatter('%m/%y'))
        colorbar_setting.add_colorbar(epochs_map, ax)
        color_fct = lambda f: epochs_map.to_rgba(f.get_epoch())
    elif color_style is not 'link' and isinstance(color_style, str):
        color_fct = lambda f: color_style
    elif nputils.is_callable(color_style):
        color_fct = color_style

    plotutils.imshow_image(ax, img, projection=projection, title=False, cmap=plotutils.get_cmap(map_cmap), **kwargs)

    for link in links:
        delta_info = link.get_delta_info(measured_delta=False)

        if ang_vel_arrows:
            plot_velocity_vector(ax, delta_info, projection, ang_vel_unit, pix_per_ang_vel_unit, 
                                 color_fct=color_fct, mode=mode, lw=0.5,
                                 fc=link.get_color(), ec='k', alpha=0.9, zorder=link.size(), width=vector_width)
        else:
            plot_displacement_vector(ax, delta_info, color_fct=color_fct, mode=mode, lw=0.5,
                                 fc=link.get_color(), ec='k', alpha=0.9, zorder=link.size(), width=vector_width)
        # y, x = link.get_features()[int(np.random.normal(s / 2, s / 4))].get_coord()
        if link_id_label:
            y, x = link.last().get_coord()
            if num_bbox is None:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor=link.get_color(), lw=1.5)
            ax.text(x, y, link.get_id(), bbox=num_bbox, zorder=200, size='small')


class DifmapComponent(object):
    ''' Format is: Flux (Jy) Radius (mas) Theta (deg) Major (mas) Axial ratio Phi (deg) T Freq (Hz):

    Flux     -  The integrated flux in the component (Jy).
    Radius   -  The radial distance of the component center from the
                center of the map (milli-arcsec).
    Theta    -  The position angle of the center of the component (degrees
                North -> East) wrt an imaginary line drawn vertically
                through the map center.

    The following components may be omitted for delta components, except
    when you want to specify spectral indexes, in which case they should
    be specified as zero.

    Major    -  The FWHM major axis of the elliptically stretched
                component (milli-arcsec).
    Ratio    -  The ratio of the minor axis to the major axis (0 -> 1).
    Phi      -  The Position angle of the major axis (degrees North -> East).
    T        -  The type of component. Recognised types are:

                 0 - Delta function.
                 1 - Gaussian.
                 2 - Uniformly bright disk.
                 3 - Optically thin sphere.
                 4 - Ring.
                 5 - Un-supported component type - kept for compatibility.
                 6 - Sunyaev-Zel'dovich.

    The following two parameters are used to optionally give components
    spectral indexes.

     Freq    -  The frequency for which the specified Flux value is defined,
                specified in Hz.
     SpecInd -  The spectral index. '''
    def __init__(self, flux, radius, theta, major=0, ratio=0, phi=0, t=0, freq=0, specind=0):
        self.flux = flux
        self.radius = radius
        self.theta = theta
        self.major = major
        self.ratio = ratio
        self.phi = phi
        self.t = t
        self.freq = freq
        self.specind = specind

    def get_coord(self):
        ''' Return coord as index: yx'''
        return nputils.coord_rtheta_to_xy(self.radius, np.radians(self.theta))[::-1]

    @staticmethod
    def new_from_segment(self, segment):
        coord_sys = segment.get_segmented_image().get_img().get_coord_sys()
        y, x = coord_sys.index2coord(segment.get_coord())
        r, theta = nputils.coord_xy_to_rtheta(x, y)


class DifmapModel(list):
    ''' See DifmapComponent for information on format '''

    CHAR_COMMENTS = "!"

    def __init__(self):
        pass

    @staticmethod
    def new_from_file(self, filename):
        model = DifmapModel()
        data = np.loadtxt(filename, dtype=object, comments=DifmapModel.CHAR_COMMENTS)

        for line in data:
            cpt = DifmapComponent(*line)
            model.append(cpt)

        return model

    def save(self, filename):
        l = []
        for cpt in self:
            data = [cpt.flux, cpt.radius, cpt.theta, cpt.major, cpt.ratio, cpt.phi, cpt.t, cpt.freq, cpt.specind]
            s = " ".join(data)
            l.append(s)

        np.savetxt(filename, l, ['%s', '%s', '%s', '%s'])


class CoreOffsetPositions(object):
    ''' Format is: epoch (iso:%Y-%m-%d), id (always 0), dist, pa (in degrees). All space separated'''

    def __init__(self):
        self.cores = dict()

    def get_all(self):
        return self.cores.items()

    def get(self, epoch):
        return self.cores.get(epoch, np.array([0, 0]))

    def align_img(self, img, projection):
        epoch = img.get_epoch()
        if epoch in self.cores:
            x, y = self.cores[epoch]
            img.shift([-x, -y], projection=projection)

    def set(self, epoch, xy_coord):
        self.cores[epoch] = xy_coord

    def save(self, filename):
        l = []
        for epoch, (x, y) in nputils.get_items_sorted_by_keys(self.cores):
            r, theta = nputils.coord_xy_to_rtheta(x, y)
            l.append([epoch.strftime("%Y-%m-%d"), 0, r, np.degrees(theta)])

        np.savetxt(filename, l, ['%s', '%s', '%s', '%s'])

    @staticmethod
    def new_from_file(file):
        core_desc = CoreOffsetPositions()
        data = np.loadtxt(file, dtype=object)

        for epoch, i, r, pa in data:
            epoch = datetime.datetime.strptime(epoch, "%Y-%m-%d")
            x, y = nputils.coord_rtheta_to_xy(float(r), np.radians(float(pa)))
            core_desc.set(epoch, [x, y])
        return core_desc


class SSPData(object):

    def __init__(self):
        import pandas as pd

        self.df = pd.DataFrame()

    def add_features_group(self, features, projection, coord_mode='com', 
                           scale=None, min_snr=2, max_snr=10):
        import pandas as pd

        coords = p2i(features.get_coords(mode=coord_mode))
        ra, dec = zip(*projection.p2s(coords))
        
        epochs = []
        intensities = []
        snrs = []
        sigma_pos = []
        idx = []
        for feature in features:
            epochs.append(feature.get_epoch())
            intensities.append(feature.get_intensity())
            snrs.append(feature.get_snr())
            sigma_pos.append(feature.get_coord_error(min_snr=min_snr, max_snr=max_snr))
            idx.append(feature.get_id())

        sigma_pos = np.array(sigma_pos)
        ra_error = np.abs(projection.mean_pixel_scale()) * sigma_pos[:, 1] #* projection.get_unit()
        dec_error = np.abs(projection.mean_pixel_scale()) * sigma_pos[:, 0] #* projection.get_unit()

        if isinstance(projection, imgutils.AbstractRelativeProjection):
            dfc = projection.dfc(coords)
            dfc_error = nputils.uarray_s((nputils.uarray(ra, ra_error) ** 2 + 
                                    nputils.uarray(dec, dec_error) ** 2) ** 0.5)
            pa = projection.pa(coords)
            pa_error = nputils.uarray_s(nputils.unumpy.arctan2(nputils.uarray(ra, ra_error), 
                                    nputils.uarray(dec, dec_error)))
        else:
            dfc = dfc_error = pa = pa_error = None

        df = pd.DataFrame({'features': list(features), 'ra': ra, 'dec': dec, 'epoch': epochs, 'snr': snrs,
                           'dfc': dfc, 'dfc_error': dfc_error, 'pa': pa, 'pa_error': pa_error, 'intensity': intensities,
                           'ra_error': ra_error, 'dec_error': dec_error,
                           }, index=idx)

        if scale is not None:
            df.loc[:, 'scale'] = scale

        self.df = self.df.append(df)

        return df

    def add_col_region(self, region_list):
        import pandas as pd

        region = [region_list.get_region(f) for f in self.df.features]
        self.df['region'] = pd.Series(region, index=self.df.index)

    def filter(self, feature_filter):
        assert isinstance(feature_filter, nputils.AbstractFilter)
        self.df = self.df[[feature_filter.filter(k) for k in self.df['features'].values]]

    @staticmethod
    def from_results(results, projection, scales=None, **kargs):
        new = SSPData()
        detection_result = results.get_detection_result()
        for ms_image in detection_result:
            for segments in ms_image:
                scale = segments.get_scale()
                if scales is None or scale in scales:
                    new.add_features_group(segments, projection, scale=scale, **kargs)

        return new


class VelocityData(SSPData):

    def __init__(self):
        SSPData.__init__(self)

    def add_delta_info(self, delta_info, match, projection, link_builder=None, 
                       coord_mode='com', scale=None, min_snr=2, max_snr=10):
        import pandas as pd
        
        features = delta_info.get_features(flag=wfeatures.DeltaInformation.DELTA_MATCH)
        cdf = self.add_features_group(features, projection, coord_mode=coord_mode, 
                                      scale=scale, min_snr=min_snr, max_snr=max_snr)
        features = features.features
        idx = cdf.index
        sigma_pos = np.array([feature.get_coord_error() for feature in features])

        deltas = delta_info.get_full_deltas(flag=wfeatures.DeltaInformation.DELTA_MATCH)

        if isinstance(features[0].get_epoch(), (float, int)):
            time = [delta.get_time() for delta in deltas] * u.second
        else:
            time = [delta.get_time().total_seconds() for delta in deltas] * u.second

        delta_pix = p2i(np.array([delta.get_delta() for delta in deltas]))
        coord1 = p2i(np.array([delta.get_feature().get_coord(mode=coord_mode) for delta in deltas]))
        coord2 = coord1 + delta_pix
        angular_sep, sep_pa = projection.angular_separation_pa(coord1, coord2)
        # angular_sep_error = projection.angular_separation(coord1, coord1 + np.array([np.sqrt(2) / 2] * 2))

        if angular_sep.unit.is_equivalent(u.deg):
            ra_error1, dec_error1 = cdf[['ra_error', 'dec_error']].as_matrix().T
            ra_error2, dec_error2 = ra_error1, dec_error1
            
            angular_sep_error_ra = np.sqrt(ra_error1 ** 2 + ra_error2 ** 2) * projection.get_unit()
            angular_sep_error_dec = np.sqrt(dec_error1 ** 2 + dec_error2 ** 2) * projection.get_unit()

            angular_velocity = (angular_sep / time).to(u.mas / u.year)

            angular_velocity_error_ra = (angular_sep_error_ra / time).to(u.mas / u.year)
            angular_velocity_error_dec = (angular_sep_error_dec / time).to(u.mas / u.year)
            angular_velocity_error = nputils.l2norm(np.array([angular_velocity_error_ra, 
                                                              angular_velocity_error_dec]).T)

        proper_sep = projection.proper_distance(coord1, coord2)

        if proper_sep.unit.is_equivalent(u.m):
            proper_velocity = (proper_sep / time).to(unit_c)

            angular_proper_ratio = proper_velocity / angular_velocity

            proper_velocity_error_ra = angular_velocity_error_ra * angular_proper_ratio
            proper_velocity_error_dec = angular_velocity_error_dec * angular_proper_ratio
            proper_velocity_error = nputils.l2norm(np.array([proper_velocity_error_ra, 
                                                             proper_velocity_error_dec]).T)

        if link_builder is not None:
            fetaure_id_mapping = link_builder.get_features_id_mapping()
            link_id = [fetaure_id_mapping.get(feature) for feature in features]
            self.df.loc[idx, 'link_id'] = pd.Series(link_id, index=idx)

        if match is not None:
            matching_features = [match.get_peer_of_one(feature) for feature in features]
            self.df.loc[idx, 'match'] = pd.Series(matching_features, index=idx)

        self.df.loc[idx, 'delta_ra'] = pd.Series(delta_pix.T[0], index=idx)
        self.df.loc[idx, 'delta_dec'] = pd.Series(delta_pix.T[1], index=idx)
        self.df.loc[idx, 'delta_time'] = pd.Series(time, index=idx)
        self.df.loc[idx, 'sep_pa'] = pd.Series(sep_pa, index=idx)
        self.df.loc[idx, 'angular_sep'] = pd.Series(angular_sep, index=idx)

        if angular_sep.unit.is_equivalent(u.deg):
            self.df.loc[idx, 'angular_velocity'] = pd.Series(angular_velocity, index=idx)
            self.df.loc[idx, 'angular_velocity_error'] = pd.Series(angular_velocity_error, index=idx)
            self.df.loc[idx, 'angular_velocity_error_ra'] = pd.Series(angular_velocity_error_ra, index=idx)
            self.df.loc[idx, 'angular_velocity_error_dec'] = pd.Series(angular_velocity_error_dec, index=idx)
        
        if proper_sep.unit.is_equivalent(u.m):
            self.df.loc[idx, 'proper_velocity'] = pd.Series(proper_velocity, index=idx)
            self.df.loc[idx, 'proper_velocity_error'] = pd.Series(proper_velocity_error, index=idx)
            self.df.loc[idx, 'proper_velocity_error_dec'] = pd.Series(proper_velocity_error_dec, index=idx)
            self.df.loc[idx, 'proper_velocity_error_ra'] = pd.Series(proper_velocity_error_ra, index=idx)

    @staticmethod
    def from_link_builder(link_builder, projection, **kargs):
        new = VelocityData()
        delta_info = link_builder.get_delta_info()
        new.add_delta_info(delta_info, None, projection, link_builder=link_builder, **kargs)

        return new

    @staticmethod
    def from_results(results, projection, scales=None, **kargs):
        new = VelocityData()
        ms_match_results, ms_link_builders = results.get_match_result()

        for link_builder, match_results in zip(ms_link_builders.get_all(), zip(*ms_match_results)):
            scale = match_results[0].get_scale()
            if scales is None or scale in scales:
                for match_result in match_results:
                    segments1, segments2, match, delta_info = match_result.get_all()
                    if delta_info.size(flag=wfeatures.DeltaInformation.DELTA_MATCH) > 0:
                        new.add_delta_info(delta_info, match, projection, 
                                            link_builder=link_builder, scale=scale, **kargs)

        return new


def align_image_on_brightest(img, bg, scale):
    finder_conf = wds.FinderConfiguration()
    finder_conf.set("min_scale", scale)
    finder_conf.set("max_scale", scale + 1)

    group = wds.FeaturesFinder(img, bg, finder_conf).execute()[0]

    cmp_intensity = lambda x, y: cmp(x.get_intensity(), y.get_intensity())
    nucleus = group.sorted_list(cmp=cmp_intensity)[-1]

    return nucleus.get_coord()


def align_image_on_cores_com(img, bg):
    finder_conf = wds.FinderConfiguration()
    finder_conf.set("max_scale", 2)
    finder_conf.set("min_scale", 1)
    finder_conf.set("interscales", False)

    group = wds.FeaturesFinder(img, bg, finder_conf).execute()[0]

    cmp_intensity = lambda x, y: cmp(x.get_intensity(), y.get_intensity())
    brightest = group.sorted_list(cmp=cmp_intensity)[-1]

    cores = wfeatures.FeaturesGroup()
    # consider only features with total intensity > 10% of brightest one.
    for feature in group.get_features():
        if feature.get_total_intensity() > 0.1 * brightest.get_total_intensity():
            cores.add_feature(feature)

    img_cores = np.zeros_like(group.get_img().data)
    for core in cores:
        img_cores += core.get_segment_image()

    com = measurements.center_of_mass(img_cores)

    print "COM:", com
    return com


def align_image_on_southern_core(img, bg):
    cmp_y = lambda x, y: cmp(x.get_coord()[0], y.get_coord()[0])

    return align_image_on_core(img, bg, cmp_y)


def align_image_on_core(img, bg, cmp_cores, core_ratio=0.1, scale=1, reg_filter=None,
                        finder_conf=None):
    if finder_conf is None:
        finder_conf = wds.FinderConfiguration()
        finder_conf.set("min_scale", scale)
        finder_conf.set("max_scale", scale + 1)

    group = wds.FeaturesFinder(img, bg, finder_conf).execute()[0]

    cmp_intensity = lambda x, y: cmp(x.get_intensity(), y.get_intensity())
    brightest = group.sorted_list(cmp=cmp_intensity)[-1]

    cores = wfeatures.FeaturesGroup()
    # consider only features with total intensity > 10% of brightest one.
    for feature in group.get_features():
        if feature.get_intensity() > core_ratio * brightest.get_intensity():
            cores.add_feature(feature)

    core = cores.sorted_list(cmp=cmp_cores)[0]

    pix_ref = core.get_coord()

    return pix_ref


def align_image_on_center_two_sided_jet(img, bg, region_jet, region_counter, core_ratio=0.1, scale=1):
    finder_conf = wds.FinderConfiguration()
    finder_conf.set("min_scale", scale)
    finder_conf.set("max_scale", scale + 1)
    finder_conf.set("interscales", False)

    features = wds.FeaturesFinder(img, bg, finder_conf).execute()[0]
    max_intensity = max(features.get_intensities())

    intensity_filter = wfeatures.IntensityFilter(intensity_min=core_ratio * max_intensity)
    filter_jet = wfeatures.RegionFilter(region_jet) & intensity_filter
    filter_counter = wfeatures.RegionFilter(region_counter) & intensity_filter

    features_jet = features.get_filtered(filter_jet)
    features_counter = features.get_filtered(filter_counter)

    mean_jet_coord = np.mean(features_jet.get_coords(), axis=0)
    mean_counter_coord = np.mean(features_counter.get_coords(), axis=0)

    first_feature_jet = features_jet.find_at_coord(mean_counter_coord, tol=None)[0]
    first_feature_counter = features_counter.find_at_coord(mean_jet_coord, tol=None)[0]

    x, y, z = nputils.get_line_between_points(img.data, first_feature_counter.get_coord(),
                                              first_feature_jet.get_coord())
    print x, y, z

    iz, = nputils.coord_min(z, fit_gaussian=True, fit_gaussian_n=3)
    izf = np.floor(iz)
    izc = np.ceil(iz)
    print nputils.coord_min(z, fit_gaussian=False), iz

    coord_x = nputils.linear_fct([izf, x[izf]], [izc, x[izc]])(iz)
    coord_y = nputils.linear_fct([izf, y[izf]], [izc, y[izc]])(iz)

    coord = [coord_y, coord_x]
    print coord

    return coord


def aligne_image_on_feature_at_pos(img, bg, scale, yx_pos, tol=10):
    finder_conf = wds.FinderConfiguration()
    finder_conf.set("min_scale", scale)
    finder_conf.set("max_scale", scale + 1)
    finder_conf.set("interscales", False)

    features = wds.FeaturesFinder(img, bg, finder_conf).execute()[0]

    return features.find_at_coord(yx_pos, tol=tol)[0].get_coord()


def align_image_on_cores_cos(img, bg):
    finder_conf = wds.FinderConfiguration()
    finder_conf.set("max_scale", 2)
    finder_conf.set("min_scale", 1)
    finder_conf.set("interscales", False)

    group = wds.FeaturesFinder(img, bg, finder_conf).execute()[0]

    cmp_intensity = lambda x, y: cmp(x.get_intensity(), y.get_intensity())
    brightest = group.sorted_list(cmp=cmp_intensity)[-1]

    cores = wfeatures.FeaturesGroup()
    # consider only features with total intensity > 10% of brightest one.
    for feature in group.get_features():
        if feature.get_total_intensity() > 0.1 * brightest.get_total_intensity():
            cores.add_feature(feature)

    img_cores = np.zeros_like(group.get_img().data)
    for id in [k.get_segmentid() for k in cores]:
        img_cores[group.get_labels() == id] = 1

    cos = measurements.center_of_mass(img_cores)

    print "COS:", cos
    return cos


