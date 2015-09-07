import os
import glob
import logging
import datetime

import numpy as np
import pandas as pd

import wds
import matcher
import features as wfeatures

from libwise import plotutils, nputils, imgutils

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from matplotlib.ticker import ScalarFormatter

from scipy.optimize import curve_fit
from scipy.ndimage import measurements

import astropy.units as u
import astropy.constants as const

unit_c = u.core.Unit("c", const.c, doc="light speed")
p2i = imgutils.p2i
logger = logging.getLogger(__name__)


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def imshow_segmented_image(ax, segmented_image, projection=None, title=True, beam=True, num=False,
                           bg=None, mode='com', **kargs):
    if bg is None:
        bg = segmented_image.get_img()

    prj = plotutils.imshow_image(ax, bg, projection=projection, title=False, beam=beam, **kargs)

    if title is True:
        title = "Segmented image"
        if isinstance(segmented_image, wds.AbstractScale):
            title += " at scale %s" % prj.get_sky(segmented_image.get_scale())
        if len(segmented_image.get_img().get_title()) > 0:
            title += "\n%s" % segmented_image.get_img().get_title()
        ax.set_title(title)

    plot_segments_contour(ax, segmented_image)

    plot_segments(ax, segmented_image, num=num, c=plotutils.orange, mode=mode)


def plot_segments_contour(ax, segmented_image, **kargs):
    for segment in segmented_image:
        [x0, y0, x1, y1] = segment.get_cropped_index()
        croped = segment.get_mask()[max(x0 - 2, 0):x1 + 3, max(y0 - 2, 0):y1 + 3]
        x0, y0 = [x0 - 2, y0 - 2]
        x1, y1 = [x1 + 2, y1 + 2]
        plotutils.plot_mask(ax, imgutils.Mask(croped), extent=[y0, y1, x0, x1], **kargs)


def plot_segments(ax, segments, num=False, mode='com', **kargs):
    plot_features(ax, segments, mode=mode, **kargs)
    if num:
        plot_segmentid(ax, segments)


def plot_segmentid(ax, segments, offset=[3, -3]):
    for segment in segments.get_features():
        y, x = segment.get_center_of_mass()
        ax.text(x - offset[0], y - offset[1], "%d" % segment.get_segmentid(), size='small')


def plot_features(ax, features, mode='com', color_fct=None, tooltip_adder=None, **kargs):
    assert isinstance(features, wfeatures.FeaturesGroup)

    if color_fct is not None:
        kargs['c'] = [color_fct(f) for f in features]

    if features.size() > 0:
        plotutils.plot_coords(ax, p2i(features.get_coords(mode=mode)), **kargs)


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


def plot_inner_features(ax, segments, **kargs):
    for segment in segments:
        if segment.has_inner_features():
            inner_features = segment.get_inner_features()
            if inner_features.size() > 1:
                plot_features(ax, inner_features, **kargs)


def plot_delta_info_1dim(ax, x, deltax, input_delta=None, axis=0):
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


def plot_displacement_vector(ax, delta_info, mode='com', color_fct=None, flag=wfeatures.DeltaInformation.DELTA_MATCH, **kwargs):
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


def plot_segments_displacements(ax, bg, features1, features2, delta_info, num=False, projection=None, mode='com',
                           beam=True, contour=True, contour_width=1, cmap=None, alpha_map=0.8, **kwargs):

    plotutils.imshow_image(ax, bg, projection=projection, beam=beam, title=False,
                           alpha=alpha_map, cmap=cmap)

    if num:
        plot_segmentid(ax, features1)

    plot_features(ax, features1, mode=mode, c=plotutils.blue, alpha=0.8)
    plot_features(ax, features2, mode=mode, c=plotutils.orange, alpha=0.8)

    plot_displacement_vector(ax, delta_info, mode=mode, fc='k', ec='k', **kwargs)

    if contour:
        plot_segments_contour(ax, features1, colors=plotutils.dblue, linewidths=contour_width)
        plot_segments_contour(ax, features2, colors=plotutils.red, linewidths=contour_width)


def plot_displacements(ax, features1, features2, delta_info, num=False, projection=None, mode='com',
                bg=None, beam=True, contour=True, contour_width=1, cmap=None, **kwargs):
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

    plot_segments_displacements(ax, bg, features1, features2, delta_info, num=num, projection=projection, mode=mode,
                           beam=beam, contour=contour, contour_width=contour_width,
                           cmap=cmap, alpha_map=alpha, **kwargs)


def plot_features_dfc(ax, projection, features, mode='com', **kargs):
    dfcs, epochs = zip(*[(projection.dfc(p2i(k.get_coord(mode=mode))), k.get_epoch()) for k in features])
    if isinstance(features, wds.AbstractScale):
        scale = features.get_scale()
        ax.plot(epochs, dfcs, ls='', marker='o', markersize=1.5 * scale, zorder=-scale, alpha=0.8, **kargs)
    else:
        ax.plot(epochs, dfcs, ls='', marker='o', alpha=0.8, **kargs)


def plot_features_pa(ax, projection, features, mode='com', **kargs):
    pas, epochs = zip(*[(projection.pa(p2i(k.get_coord(mode=mode))), k.get_epoch()) for k in features])
    if isinstance(features, wds.AbstractScale):
        scale = features.get_scale()
        ax.plot(epochs, pas, ls='', marker='o', markersize=1.5 * scale, zorder=-scale, alpha=0.8, **kargs)
    else:
        ax.plot(epochs, pas, ls='', marker='o', alpha=0.8, **kargs)


def plot_ms_set_map(ax, img, ms_set, projection, mode='com', color_style='date', colorbar_setting=None, 
                    map_cmap='jet', **kargs):
    if colorbar_setting is None and color_style == 'date':
        colorbar_setting = plotutils.ColorbarSetting(ticks_locator=mdates.AutoDateLocator(),
                                                     ticks_formatter=mdates.DateFormatter('%m/%y'))
    
    epochs = ms_set.get_epochs()
    intensities = [k.get_intensity() for k in ms_set.features_iter()]
    int_norm = plotutils.Normalize(min(intensities), max(intensities))
    marker_select = plotutils.MarkerSelector()

    epochs_map = plotutils.build_epochs_mappable(epochs, colorbar_setting.get_cmap())

    if img is not None:
        plotutils.imshow_image(ax, img, projection=projection, title=False, cmap=cm.get_cmap(map_cmap), **kargs)

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
                          fit_min_size=3, **kargs):
    # Deprecated
    filter1 = wfeatures.DateFilter.from_filter_fct(date_filter)
    if feature_filter is not None:
        filter1 &= feature_filter
    links = link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size)

    plot_links_dfc(ax, projection, links, mode=mode, num=num, **kargs)
    if fit_fct:
        plot_links_dfc_fit(ax, projection, links, fit_fct, fit_min_size=fit_min_size, mode=mode)


def plot_links_dfc(ax, projection, links, mode='com', num=False, num_bbox=None, **kargs):
    dfc_fct_coord = lambda coord: nputils.l2norm(projection.p2s(p2i(coord)))
    dfc_fct = lambda feature: dfc_fct_coord(feature.get_coord(mode))

    for link in links:
        delta_info = link.get_delta_info()

        if num is True:
            if num_bbox is None:
                num_bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor=link.get_color(), lw=1.5)
            # ax.text(link.get_first_epoch(), dfc_fct(link.first()), "%s, %s" % (link.get_id(), link.size()))            
            ax.text(link.get_first_epoch(), dfc_fct(link.first()), link.get_id(), bbox=num_bbox, zorder=200, size='small')

        if 'c' not in kargs:
            line_color = link.get_color()
        else:
            line_color = kargs['c']
        # plotutils.checkargs(kargs, 'lw', 2.5)

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
            ax.plot([epoch1, epoch2], [dfc1, dfc2], color=line_color, **kargs)
            ax.plot([epoch1], dfc1, ls='', marker='+', color='k')
        # ax.plot([epoch2], dfc2, ls='', marker='+', color='k')


def plot_links_dfc_fit(ax, projection, links, fit_fct, fit_min_size=3, mode='com', **kargs):
    results = dict()
    for link in links:
        if fit_fct is not None and link.size() >= fit_min_size:
            fct, epochs, dfcs = link.fit_dfc(fit_fct, projection, coord_mode=mode)
            results[link] = fct
            v = fct.a
            v_error = fct.error(epochs, dfcs)
            label = 'v = %.3f +- %.3f %s' % (v, v_error, (projection.unit / u.year).to_string())
            # datetime_epochs = [nputils.mjd_to_datetime(mjd) for mjd in epochs]
            ax.plot(link.get_epochs(), fct(epochs), label=label, c=link.get_color(), **kargs)

    return results


def plot_links_pa(ax, projection, links, mode='com', **kargs):
    for link in links:
        pas = [np.degrees(projection.pa(p2i(k.get_coord(mode=mode)))) for k in link.get_features()]
        epochs = link.get_epochs()

        ax.plot(epochs, pas, marker='+', color=link.get_color(), **kargs)


def plot_links_snr(ax, projection, links, mode='com', **kargs):
    for link in links:
        # snrs = [k.get_segmented_image().get_feature_snr(k) for k in link.get_features()]
        # snrs = [k.get_intensity() for k in link.get_features()]
        snrs = [k.get_snr() for k in link.get_features()]
        epochs = link.get_epochs()

        ax.plot(epochs, snrs, marker='+', color=link.get_color(), **kargs)


class AbsFormatter(object):
    def __init__(self, useMathText=True):
        self._fmt = ScalarFormatter(useMathText=useMathText, useOffset=False)
        self._fmt.create_dummy_axis()

    def __call__(self, direction, factor, values):
        self._fmt.set_locs(values)
        return [self._fmt(abs(v)) for v in values]


def plot_velocity_map(ax, stack_img, projection, link_builder, min_link_size=2, title=False,
                      feature_filter=None, color_style='link', mode='com', colorbar_setting=None,
                      dfc_axis=False, dfc_axis_teta=None, map_cmap='jet', vector_width=4, 
                      dfc_axis_pos=(1, 3), link_id_label=False, **kwargs):
    ''' DEPRECATED
        dfc_axis_teta in radian
        dfc_axis_pos is (nth_coord, value) '''

    links = link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size)

    wiseutils.plot_links_map(ax, stack_img, projection, links, color_style=color_style, 
                               mode=mode, colorbar_setting=colorbar_setting, map_cmap=map_cmap,
                               vector_width=vector_width, link_id_label=link_id_label, 
                               num_bbox=None, **kargs)

    if dfc_axis:
        axis = plot_dfc_axis(ax, projection, dfc_axis_teta, dfc_axis_pos=dfc_axis_pos)
        axis.label.set_text("$z_{obs} (mas)$")

    scale = projection.get_sky(link_builder.get_scale())
    if isinstance(title, bool) and title is True:
        ax.set_title(ax.get_title() + '\nVelocity map at scale %s.' % scale)
    if isinstance(title, str):
        ax.set_title(title)


def plot_links_map(ax, img, projection, links, color_style='link', mode='com', colorbar_setting=None,
                     map_cmap='jet', vector_width=4, link_id_label=False, num_bbox=None, **kwargs):
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

    plotutils.imshow_image(ax, img, projection=projection, title=False, cmap=cm.get_cmap(map_cmap), **kwargs)

    for link in links:
        delta_info = link.get_delta_info(measured_delta=False)

        plot_displacement_vector(ax, delta_info, color_fct=color_fct, mode=mode, lw=0.5,
                             fc=link.get_color(), ec='k', alpha=0.9, zorder=link.size(), width=vector_width)
        # y, x = link.get_features()[int(np.random.normal(s / 2, s / 4))].get_coord()
        if link_id_label:
            y, x = link.last().get_coord()
            if num_bbox is None:
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor=link.get_color(), lw=1.5)
            ax.text(x, y, link.get_id(), bbox=num_bbox, zorder=200, size='small')


def plot_dfc_axis(ax, projection, dfc_axis_teta, dfc_axis_pos=(1, 3)):
    axis = projection.new_rotated_floating_axis([0, 0], dfc_axis_teta, dfc_axis_pos[0], 
        dfc_axis_pos[1], ax)
    axis.set_ticklabel_direction("+")
    axis.major_ticklabels.set_axis_direction("bottom")
    
    axis.set_axis_direction("bottom")
    axis.set_axislabel_direction("+")

    finder = axis._axis_artist_helper.grid_helper.grid_finder
    finder.update(grid_locator1=MaxNLocator(15, integer=True), tick_formatter1=AbsFormatter())

    return axis


class DataConfiguration(nputils.BaseConfiguration):

    def __init__(self):
        data = [
        ["data_dir", None, "Base data directory", nputils.validator_is(str)],
        ["fits_extension", 0, "Extension index", nputils.validator_is(int)],
        ["stack_image_filename", "full_stack_image.fits", "Stack Image filename", nputils.validator_is(str)],
        ["ref_image_filename", "reference_image", "Stack Image filename", nputils.validator_is(str)],
        ["mask_filename", "mask.fits", "Mask filename", nputils.validator_is(str)],
        ["mask_fct", None, "Mask generation fct", nputils.is_callable],
        ["bg_fct", None, "Background extraction fct", nputils.is_callable],
        ["core_offset_filename", "core.dat", "Core offset filename", nputils.validator_is(str)],
        ["core_offset_fct", None, "Core offset generation fct", nputils.is_callable],
        ["pre_bg_process_fct", None, "Initial processing before bg extraction", nputils.is_callable],
        ["pre_process_fct", None, "Pre detection processing", nputils.is_callable],
        ["post_process_fct", None, "Post detection processing", nputils.is_callable],
        ["crval", None, "CRVAL", nputils.validator_is(list)],
        ["crpix", None, "CRPIX", nputils.validator_is(list)],
        ["projection_unit", u.mas, "Unit used for the projection", nputils.validator_is(u.Unit)],
        ["projection_relative", True, "Use relative projection", nputils.validator_is(bool)],
        ["projection_center", "pix_ref", "Method used to get the center", nputils.validator_is(str)],
        ["object_distance", None, "Object distance", nputils.validator_is(u.Quantity)],
        ["object_z", 0, "Object z", nputils.validator_in_range(0, 5)],
        ]

        # nputils.BaseConfiguration.__init__(self, data, title="Finder configuration")
        super(DataConfiguration, self).__init__(data, title="Data configuration")


class AnalysisConfiguration(nputils.ConfigurationsContainer):

    def __init__(self):
        self.data = DataConfiguration()
        self.finder = wds.FinderConfiguration()
        self.matcher = matcher.MatcherConfiguration()
        nputils.ConfigurationsContainer.__init__(self, [self.data, self.finder, self.matcher])

    # def to_file(self, name, path):
    #     base = os.path.join(path, name)
    #     self.data.to_file(base + ".data.conf")
    #     print "Saved data configuration @ %s" % base + ".data.conf"
    #     self.finder.to_file(base + ".finder.conf")
    #     print "Saved finder configuration @ %s" % base + ".finder.conf"
    #     self.matcher.to_file(base + ".matcher.conf")
    #     print "Saved matcher configuration @ %s" % base + ".matcher.conf"


class AnalysisResult(object):

    def __init__(self, config):
        self.detection = wds.MultiScaleImageSet()
        self.ms_match_results = matcher.MultiScaleMatchResultSet()
        self.link_builder = matcher.MultiScaleFeaturesLinkBuilder()
        self.image_set = imgutils.ImageSet()
        self.config = config

    def has_detection_result(self):
        return len(self.detection) > 0

    def has_match_result(self):
        return len(self.ms_match_results) > 0

    def get_scales(self):
        return self.detection.get_scales()

    def add_detection_result(self, img, res):
        self.image_set.add_img(img)
        self.detection.append(res)

    def add_match_result(self, match_res):
        self.ms_match_results.append(match_res)
        self.link_builder.add_match_result(match_res)

    def get_match_result(self):
        if self.ms_match_results is None:
            raise Exception("No match result found.")
        return self.ms_match_results, self.link_builder

    def get_detection_result(self):
        if self.detection is None:
            raise Exception("No detection result found.")
        return self.detection


class AnalysisContext(object):

    def __init__(self, config=None):
        if config is None:
            config = AnalysisConfiguration()
        self.config = config
        self.result = AnalysisResult(self.config)
        self.files = []

        self.cache_mask_filter = None
        self.cache_core_offset = None

    def get_data_dir(self):
        path = self.config.data.data_dir
        if self.config.data.data_dir is None:
            raise Exception("A data directory need to be set")
        if not os.path.exists(path):
            print "Creating %s" % path
            os.makedirs(path)
        return path

    def get_projection(self, img):
        return img.get_projection(relative=self.config.data.projection_relative, 
                                  center=self.config.data.projection_center, 
                                  unit=self.config.data.projection_unit, 
                                  distance=self.config.data.object_distance, 
                                  z=self.config.data.object_z)

    def get_core_offset_filename(self):
        path = self.get_data_dir()
        return os.path.join(path, self.config.data.core_offset_filename)

    def get_mask_filename(self):
        path = self.get_data_dir()
        return os.path.join(path, self.config.data.mask_filename)

    def get_stack_image_filename(self):
        path = self.get_data_dir()
        return os.path.join(path, self.config.data.stack_image_filename)

    def get_ref_image(self, preprocess=True):
        ref_file =  os.path.join(self.get_data_dir(), self.config.data.ref_image_filename)
        if not os.path.isfile(ref_file):
            if len(self.files) == 0:
                raise Exception("No files selected")
            ref_file = self.files[0]

        img = imgutils.guess_and_open(ref_file, check_stack_img=True)

        if preprocess:
            self.pre_process(img)

        return img

    def set_ref_image(self, img):
        ref_file =  os.path.join(self.get_data_dir(), self.config.data.ref_image_filename)
        img.save(ref_file)

    def get_result(self):
        return self.result

    def get_match_result(self):
        return self.result.get_match_result()

    def get_detection_result(self):
        return self.result.get_detection_result()

    def get_mask(self):
        filename = self.get_mask_filename()
        if self.cache_mask_filter is None or self.cache_mask_filter[0] != filename \
                or self.cache_mask_filter[1] is None:
            self.cache_mask_filter = [filename, None]
            if os.path.isfile(filename):
                self.cache_mask_filter[1] = imgutils.FitsImage(filename)
        return self.cache_mask_filter[1]

    def align(self, img):
        filename = self.get_core_offset_filename()
        if self.cache_core_offset is None or self.cache_core_offset[0] != filename:
            self.cache_core_offset = [filename, None]
            if os.path.isfile(filename):
                core_offset_pos = CoreOffsetPositions.new_from_file(filename)
                self.cache_core_offset[1] = core_offset_pos
        if isinstance(self.cache_core_offset[1], CoreOffsetPositions):
            print "Aligning:", img.get_epoch()
            self.cache_core_offset[1].align_img(img, projection=self.get_projection(img))

    def build_stack_image(self, mode='full', detection_snr=7, preprocess=False):
        stack_mgr = imgutils.StackedImageBuilder()
        for file in self.files:
            img = self.open_file(file)
            if preprocess:
                self.pre_process(img)
            self.align(img)
            if mode == 'detection_count':
                bg = self.get_bg(img)
                img.data = (img.data > detection_snr * bg.std()).astype(np.float)
            stack_mgr.add(img)
        return stack_mgr.get()

    def get_stack_image(self, nsigma=0, nsigma_connected=False, preprocess=False):
        filename = self.get_stack_image_filename()
        if filename is None or not os.path.isfile(filename):
            raise Exception("A stack image need to be generated")

        stack_image = imgutils.StackedImage.from_file(filename)
        if nsigma > 0:
            bg = self.get_bg(stack_image)
        if preprocess:
            self.pre_process(stack_image)
        if nsigma > 0:
            stack_image.data[stack_image.data < nsigma * bg.std()] = 0
            if nsigma_connected:
                segments = wds.SegmentedImages(stack_image)
                segments.connected_structure()
                stack_image.data = segments.sorted_list()[-1].get_segment_image()

        return stack_image

    def open_file(self, file):
        img = imgutils.guess_and_open(file, fits_extension=self.config.data.fits_extension)
        if self.config.data.crval is not None:
            img.set_crval(self.config.data.crval)
        if self.config.data.crpix is not None:
            img.set_pix_ref(self.config.data.crpix)
        return img

    def get_bg(self, img):
        if self.config.data.bg_fct is None:
            raise Exception("Bg extraction method need to be set")
        return self.config.data.bg_fct(self, img)

    def pre_bg_process(self, img):
        if self.config.data.pre_bg_process_fct is not None:
            self.config.data.pre_bg_process_fct(self, img)

    def pre_process(self, img):
        if self.config.data.pre_process_fct is not None:
            self.config.data.pre_process_fct(self, img)

    def post_process(self, img, res):
        if self.config.data.post_process_fct is not None:
            self.config.data.post_process_fct(self, img, res)

    def save_core_offset_pos_file(self):
        if self.config.data.core_offset_fct is None:
            print "Warning: No core offset fct defined"
            return
        filename = self.get_core_offset_filename()
        core_offset_pos = CoreOffsetPositions()

        for file in self.files:
            img = self.open_file(file)
            core = self.config.data.core_offset_fct(self, img)
            core_offset_pos.set(img.get_epoch(), core)

        self.cache_core_offset = None
        core_offset_pos.save(filename)

    def save_mask_file(self, mask_fct):
        filename = self.get_mask_filename()
        if os.path.isfile(filename):
            os.remove(filename)
        mask = mask_fct(self)
        mask.data = mask.data.astype(bool).astype(float)

        self.cache_mask_filter = None
        mask.save_to_fits(filename)

    def save_stack_image(self, preprocess=False):
        stack = self.build_stack_image(preprocess=preprocess)
        stack.save_to_fits(self.get_stack_image_filename())

    def detection(self, img, config=None, filter=None):
        print "Start detection on: %s" % img
        if filter is not None:
            print "  with filter: %s" % filter
        self.pre_bg_process(img)
        bg = self.get_bg(img)
        detection_filter = wfeatures.MaskFilter(self.get_mask())
        if filter is not None:
            detection_filter = detection_filter & filter

        self.align(img)
        self.pre_process(img)

        if config is None:
            config = self.config.finder

        finder = wds.FeaturesFinder(img, bg, config=config, filter=detection_filter)

        res = finder.execute()

        self.post_process(img, res)

        return res

    def select_files(self, files, start_date=None, end_date=None, filter_dates=None, step=1):
        files = glob.glob(files)

        self.files = imgutils.fast_sorted_fits(files, start_date=start_date, 
                            end_date=end_date, filter_dates=filter_dates, step=step)

        print "Number of files selected:", len(self.files)

    def match(self, find_res1, find_res2):
        m = matcher.ImageMatcher(self.config.finder, self.config.matcher)

        return m.get_match(find_res1, find_res2)


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
        for epoch, (x, y) in self.cores.items():
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
        self.df = pd.DataFrame()

    def add_features_group(self, features, projection, coord_mode='com', scale=None):
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
            sigma_pos.append(feature.get_coord_error(min_snr=2, max_snr=10))
            idx.append(feature.get_id())

        sigma_pos = np.array(sigma_pos)
        ra_error = np.abs(projection.mean_pixel_scale()) * sigma_pos[:, 1] * projection.get_unit()
        dec_error = np.abs(projection.mean_pixel_scale()) * sigma_pos[:, 0] * projection.get_unit()


        if isinstance(projection, imgutils.AbstractRelativeProjection):
            dfc = projection.dfc(coords)
            pa = projection.pa(coords)
        else:
            dfc = pa = None

        df = pd.DataFrame({'features': list(features), 'ra': ra, 'dec': dec, 'epoch': epochs, 'snr': snrs,
                           'dfc': dfc, 'pa': pa, 'intensity': intensities,
                           'ra_error': ra_error, 'dec_error': dec_error,
                           }, index=idx)

        if scale is not None:
            df.loc[:, 'scale'] = scale

        self.df = self.df.append(df)

        return df

    def add_col_region(self, region_list):
        region = [region_list.get_region(f) for f in self.df.index]
        self.df['region'] = pd.Series(region, index=self.df.index)

    def filter(self, feature_filter):
        assert isinstance(feature_filter, nputils.AbstractFilter)
        self.df = self.df[[feature_filter.filter(k) for k in self.df.features.values]]

    @staticmethod
    def from_results(results, projection, scales=None, coord_mode='com'):
        new = SSPData()
        detection_result = results.get_detection_result()
        for ms_image in detection_result:
            for segments in ms_image:
                scale = segments.get_scale()
                if scales is None or scale in scales:
                    new.add_features_group(segments, projection, coord_mode=coord_mode, scale=scale)

        return new


class VelocityData(SSPData):

    def __init__(self):
        SSPData.__init__(self)

    def add_delta_info(self, delta_info, match, projection, link_builder=None, coord_mode='com', scale=None):
        features = delta_info.get_features(flag=wfeatures.DeltaInformation.DELTA_MATCH)
        cdf = self.add_features_group(features, projection, coord_mode=coord_mode, scale=scale)
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
    def from_link_builder(link_builder, projection, coord_mode='com'):
        new = VelocityData()
        delta_info = link_builder.get_delta_info()
        new.add_delta_info(delta_info, None, projection, link_builder=link_builder, 
                                           coord_mode=coord_mode)

        return new

    @staticmethod
    def from_results(results, projection, scales=None, coord_mode='com'):
        new = VelocityData()
        ms_match_results, ms_link_builders = results.get_match_result()

        for link_builder, match_results in zip(ms_link_builders.get_all(), zip(*ms_match_results)):
            scale = match_results[0].get_scale()
            if scales is None or scale in scales:
                for match_result in match_results:
                    segments1, segments2, match, delta_info = match_result.get_all()
                    if delta_info.size(flag=wfeatures.DeltaInformation.DELTA_MATCH) > 0:
                        new.add_delta_info(delta_info, match, projection, link_builder=link_builder, 
                                           coord_mode=coord_mode, scale=scale)

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


