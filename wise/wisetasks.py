import os
import re
import sys
import time
import glob
import inspect
import datetime

import numpy as np

import scc
import wds
import matcher
import wiseutils
import features as wfeatures

from libwise import plotutils, nputils, imgutils

import astropy.units as u
import astropy.constants as const

unit_c = u.core.Unit("c", const.c, doc="light speed")

'''
task documentation template:

Short Description.

More Description

*ctx*
    AnalysisContext object

*option1*


'''


def build_final_dfc(ctx, merge_file, final_sep):
    ''' Build a final seperation file from a merge file 

        Merge file shall be located in ctx.get_data_dir(). 
        One final component per line, described by a list of link id seperated by a ','.

        *merge_file*
            Merge file name

        *final_sep*
            Name of the final separation file 
        '''
    merge_file = os.path.join(ctx.get_data_dir(), merge_file)
    final_sep_file = os.path.join(ctx.get_data_dir(), final_sep)

    stack_img = ctx.get_stack_image()
    ctx.pre_process(stack_img)
    projection = ctx.get_projection(stack_img)

    merged_builder = matcher.MergedFeatureLinkBuilder(ctx.result.link_builder, merge_file)

    merged_builder.to_file(final_sep_file, projection)


def print_beam_info(ctx):
    ''' Print List of selected files with information on beam and pixel scales '''
    all_data = []
    for file in ctx.files:
        img = ctx.open_file(file)
        beam = img.get_beam()
        date = img.get_epoch().strftime('%Y-%m-%d')
        k = ctx.get_projection(img).mean_pixel_scale()
        b = [k * beam.bmin, k * beam.bmaj, beam.bpa]
        u = ctx.get_projection(img).get_unit()
        data = (os.path.basename(file), date, b[0] * u, b[1] * u, b[2],
                ctx.get_projection(img).mean_pixel_scale() * u, img.data.shape[0], img.data.shape[1])
        all_data.append(data)
        print "{0}: Date: {1}, Beam: {2:.3f}, {3:.3f}, {4:.2f}, Pixel: {5:.3f}, Shape: {6:d}x{7:d}".format(*data)

    bmin_mean = np.array([k[2].value for k in all_data]).mean()
    bmax_mean = np.array([k[3].value for k in all_data]).mean()
    angle_mean = np.array([k[4] for k in all_data]).mean()
    print "Mean beam: Bmin: %.3f, Bmaj: %.3f, Angle:%.2f" % (bmin_mean, bmax_mean, angle_mean)


def print_delta_info(ctx, delta_time_unit=u.day, angular_velocity_unit=(u.mas / u.year),
                     proper_velocity_unit=unit_c):
    ''' Print List of selected pair of files with information on velocity resolution '''
    all_delta_time = []
    all_velocity_c_px = []
    all_velocity_px = []
    for file1, file2 in nputils.pairwise(ctx.files):
        img1 = ctx.open_file(file1)
        img2 = ctx.open_file(file2)
        prj = ctx.get_projection(img1)
        date1 = img1.get_epoch().strftime('%Y-%m-%d')
        date2 = img2.get_epoch().strftime('%Y-%m-%d')
        delta_t = ((img2.get_epoch() - img1.get_epoch()).total_seconds() * u.second).to(delta_time_unit)
        all_delta_time.append(delta_t)

        velocity_c_px = prj.proper_velocity([0, 0], [0, 1], delta_t).to(proper_velocity_unit)
        velocity_px = prj.angular_velocity([0, 0], [0, 1], delta_t).to(angular_velocity_unit)
        all_velocity_c_px.append(velocity_c_px)
        all_velocity_px.append(velocity_px)

        data = (date1, date2, delta_t, velocity_px, velocity_c_px)

        print "{0} -> {1}: Delta time: {2}, Velocity resolution: {3:.3f}, {4:.3f}".format(*data)

    all_delta_time = nputils.quantities_to_array(all_delta_time)
    all_velocity_c_px = nputils.quantities_to_array(all_velocity_c_px)
    all_velocity_px = nputils.quantities_to_array(all_velocity_px)

    print "\nMean Delta time: %s +- %s" % (np.mean(all_delta_time), np.std(all_delta_time))
    print "Mean Velocity resolution: %s +- %s" % (np.mean(all_velocity_px), np.std(all_velocity_px))
    print "Mean Velocity resolution: %s +- %s" % (np.mean(all_velocity_c_px), np.std(all_velocity_c_px))


def detection_all(ctx, filter=None):
    ''' Run wds on all selected files '''
    ctx.result = wiseutils.AnalysisResult(ctx.config)
    for file in ctx.files:
        img = ctx.open_file(file)
        res = ctx.detection(img, filter=filter)
        ctx.result.add_detection_result(img, res)


def match_all(ctx, filter=None):
    ''' Run matching on all selected files '''
    ctx.result = wiseutils.AnalysisResult(ctx.config)
    prev_result = None
    for file in ctx.files:
        img = ctx.open_file(file)
        res = ctx.detection(img, filter=filter)
        ctx.result.add_detection_result(img, res)

        result_copy = res.copy()

        if prev_result is not None:
            match_res = ctx.match(prev_result, res)
            ctx.result.add_match_result(match_res)

        prev_result = result_copy


def save(ctx, name, coord_mode='com', measured_delta=True):
    ''' Save current result to files

        *name*
            Prefix name for the save files '''
    if ctx.result is None:
        print "Warning: no result to save"
        return

    stack_img = ctx.get_stack_image(preprocess=True)
    projection = ctx.get_projection(stack_img)

    path = os.path.join(ctx.get_data_dir(), name)
    if not os.path.exists(path):
        os.mkdir(path)
    ctx.result.detection.to_file(os.path.join(path, "%s.ms.dat" % name), projection, coord_mode=coord_mode)
    ctx.result.link_builder.to_file(os.path.join(path, name), projection, 
                               coord_mode=coord_mode, measured_delta=measured_delta)
    ctx.result.image_set.to_file(os.path.join(path, "%s.set.dat" % name), projection)
    ctx.result.config.to_file(os.path.join(path, "%s.conf" % name))


def load(ctx, name, projection=None, merge_with_previous=False, min_link_size=2):
    ''' Load result from files

        *name*
            Prefix name for the saved files '''

    if projection is None:
        stack_img = ctx.get_stack_image(preprocess=True)
        projection = ctx.get_projection(stack_img)
        
    path = os.path.join(ctx.get_data_dir(), name)
    if not os.path.isdir(path):
        raise Exception("No results found with name %s" % name)

    img_set_file = os.path.join(path, "%s.set.dat" % name)
    ms_detec_file = os.path.join(path, "%s.ms.dat" % name)
    link_builder_name = os.path.join(path, name)
    
    image_set = imgutils.ImageSet.from_file(img_set_file, projection)

    detection = matcher.MultiScaleImageSet.from_file(ms_detec_file, projection,
                                    image_set)

    link_builder = matcher.MultiScaleFeaturesLinkBuilder.from_file(link_builder_name, 
                                    projection, image_set, min_link_size=min_link_size)
    ms_match_results = link_builder.get_ms_match_results()

    if merge_with_previous and ctx.result is not None:
        ctx.result.image_set.merge(image_set)
        ctx.result.detection.merge(detection)
        ctx.result.link_builder.merge(link_builder)
        ctx.result.ms_match_results.merge(ms_match_results)
    else:
        ctx.result = wiseutils.AnalysisResult(ctx.config)
        ctx.result.image_set = image_set
        ctx.result.detection = detection
        ctx.result.link_builder = link_builder
        ctx.result.ms_match_results = ms_match_results


def view_all(ctx, preprocess=True, show_mask=True, show_regions=[], save_filename=None, **kargs):
    ''' Preview all selected files '''
    stack = plotutils.FigureStack()

    for file in ctx.files:
        img = ctx.open_file(file)
        # print "File:%s, Bg: %s, Noise:%s" % (file, ctx.get_bg(img).std(), img.header["NOISE"])
        
        def do_plot(fig):
            ax = fig.subplots()
            ctx.align(img)
            if preprocess:
                ctx.pre_process(img)
            plotutils.imshow_image(ax, img, projection=ctx.get_projection(img), **kargs)
            if not preprocess:
                bg_mask = imgutils.Image(np.zeros_like(img.data, dtype=np.int8))
                bg = ctx.get_bg(bg_mask)
                if bg is not None and isinstance(bg, np.ndarray):
                    bg.fill(1)
            if ctx.get_mask() is not None and show_mask is True:
                ax.contour(ctx.get_mask().data, [0.5])
            for region in show_regions:
                plotutils.plot_region(ax, region, ctx.get_projection(img), text=True)

        stack.add_replayable_figure(img.get_epoch(), do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.show()


def view_stack(ctx, preprocess=True, nsigma=0, nsigma_connected=False, show_mask=True, show_regions=[], 
                  intensity_colorbar=False, save_filename=None, **kargs):
    ''' Preview the stack image '''


    stack = plotutils.FigureStack()

    stack_img = ctx.get_stack_image(nsigma=nsigma, nsigma_connected=nsigma_connected)
    if preprocess:
        ctx.pre_process(stack_img)

    def do_plot(fig):
        ax = fig.subplots()

        plotutils.imshow_image(ax, stack_img, projection=ctx.get_projection(stack_img), 
                               intensity_colorbar=intensity_colorbar, **kargs)
        if ctx.get_mask() is not None and show_mask is True:
            ax.contour(ctx.get_mask().data, [0.5])
        for region in show_regions:
            plotutils.plot_region(ax, region, ctx.get_projection(stack_img), text=False, fill=True)

    stack.add_replayable_figure("Stack Image", do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.show()


def view_wds(ctx, title=True, num=True, scale=None, save_filename=None, **kargs):
    ''' Plot WDS decomposition '''
    stack_img = ctx.get_stack_image(preprocess=True)
    projection = ctx.get_projection(stack_img)

    detection_result = ctx.get_detection_result()

    if not detection_result.is_full_wds():
        print "This task require full WDS result. Please run detection again."
        return

    stack = plotutils.FigureStack()

    nax = 1

    for res in detection_result:
        epoch = res.get_epoch()
        if scale is not None:
            res = [res.get_scale(scale)]
        nax = len(res)

        def do_plot(fig):
            ax = fig.subplots(n=nax, reshape=False)
            for ax, segments in zip(ax, res):
                wiseutils.imshow_segmented_image(ax, segments, title=title, 
                    projection=projection, num=num, **kargs)

        stack.add_replayable_figure("WDS %s" % epoch, do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.show()


def view_all_features(ctx, scales, region_list=None, legend=True, feature_filter=None, 
                      save_filename=None, **img_kargs):
    stack_img = ctx.get_stack_image(preprocess=True, nsigma=3, nsigma_connected=True)
    projection = ctx.get_projection(stack_img)

    data = wiseutils.SSPData.from_results(ctx.get_result(), projection, scales)
    if feature_filter is not None:
        data.filter(feature_filter)
    if region_list is not None:
        data.add_col_region(region_list)

    stack = plotutils.FigureStack()

    def do_plot(fig):
        ax_all = fig.subplots()

        plotutils.imshow_image(ax_all, stack_img, projection=projection, **img_kargs)

        if region_list is not None:
            for region, gdata in data.df.groupby('region'):
                features = wfeatures.FeaturesGroup(gdata.index)

                wiseutils.plot_features(ax_all, features, mode='com', c=region.get_color(), label=region.get_name())
                plotutils.plot_region(ax_all, region, projection=projection, text=False, 
                                      color=region.get_color(), fill=True)
        else:
            features = wfeatures.FeaturesGroup(data.df.index)
            wiseutils.plot_features(ax_all, features, mode='com')

        if legend:
            ax_all.legend(loc='best')

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.add_replayable_figure("All features", do_plot)
    stack.show()


def plot_separation_from_core(ctx, scale, feature_filter=None, min_link_size=2, title=True, 
                            pa=False, snr=False, num=False, fit_fct=None, save_filename=None, **kargs):
    stack = plotutils.FigureStack()

    stack_img = ctx.get_stack_image(nsigma=3, preprocess=True, nsigma_connected=True)
    projection = ctx.get_projection(stack_img)

    ms_match_results, link_builders = ctx.get_match_result()

    for link_builder in link_builders.get_all()[:]:
        if scale is None or scale == link_builder.get_scale():
            links = list(link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size))

            def do_plot(fig):
                ax = fig.subplots(nrows=1 + int(pa or snr), reshape=False, sharex=True)

                if fit_fct:
                    plotutils.checkargs(kargs, 'ls', '--')
                wiseutils.plot_links_dfc(ax[0], projection, links, num=num, **kargs)
                if fit_fct:
                    wiseutils.plot_links_dfc_fit(ax[0], projection, links, fit_fct, lw=2)

                scale = projection.mean_pixel_scale() * link_builder.get_scale()

                if title:
                    ax[0].set_title("Distance from core at scale %.2f mas." % scale)
                ax[-1].set_xlabel("Epoch (years)")
                ax[0].set_ylabel("Separation from core (mas)")

                if pa:
                    wiseutils.plot_links_pa(ax[1], projection, links, **kargs)
                elif snr:
                    wiseutils.plot_links_snr(ax[1], projection, links, **kargs)
                    ax[1].set_ylabel("Wavelet Coef SNR")

            stack.add_replayable_figure("DFC scale %s" % scale, do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.show()


def plot_all_features(ctx, scales=None, pa=False, feature_filter=None, save_filename=None):
    stack_img = ctx.get_stack_image(preprocess=True, nsigma=3, nsigma_connected=True)
    projection = ctx.get_projection(stack_img)

    data = wiseutils.SSPData.from_results(ctx.get_result(), projection, scales)
    if feature_filter is not None:
        data.filter(feature_filter)

    stack = plotutils.FigureStack()

    def do_plot(fig):
        ax = fig.subplots(nrows=1 + int(pa), reshape=False, sharex=True)
        colors = plotutils.ColorSelector()

        for scale, gdata in data.df.groupby('scale'):
            color = colors.get()
            features = wds.DatedFeaturesGroupScale(scale, features=gdata.index)

            wiseutils.plot_features_dfc(ax[0], projection, features, c=color)

            if pa:
                wiseutils.plot_features_pa(ax[1], projection, features, c=color)

        ax[-1].set_xlabel("Epoch (years)")
        ax[0].set_ylabel("Distance from core (mas)")
        
        if pa:
            ax[1].set_ylabel("PA (degree)")

    stack.add_replayable_figure("All features", do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.show()


def view_displacements(ctx, scale, feature_filter=None, title=True, save_filename=None):
    ''' Plot match results '''
    stack = plotutils.FigureStack()

    stack_img = ctx.get_stack_image(nsigma=3, preprocess=True, nsigma_connected=True)
    projection = ctx.get_projection(stack_img)

    ms_match_results, link_builders = ctx.get_match_result()

    for ms_match_result in ms_match_results:
        epoch = ms_match_result.get_epoch()
        match_result = ms_match_result.get_scale(scale)

        segments1, segments2, match, delta_info = match_result.get_all(feature_filter=feature_filter)

        if segments1.size() == 0:
            continue

        def do_plot(figure):
            ax = figure.subplots()

            axtitle = 'Velocity vector at scale %.2f' % segments1.get_scale(projection=projection)

            if isinstance(segments1, wds.SegmentedImages):
                axtitle += '\n%s' % segments1.get_img().get_title()
                axtitle += '\n%s' % segments2.get_img().get_title()
                bg = segments1.get_img()
            else:
                axtitle += '\n%s vs %s' % (segments1.get_epoch(), segments2.get_epoch())
                bg = stack_img

            wiseutils.plot_displacements(ax, segments1, segments2, delta_info, 
                                 projection=projection, bg=bg)

            if title:
                ax.set_title(axtitle)

        stack.add_replayable_figure(epoch, do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.show()


def view_links(ctx, scale, feature_filter=None, min_link_size=2, map_cmap='YlGnBu_r',
               vector_width=6, title=True, color_style='link', save_filename=None, **kargs):
    ''' Plot stacked match results '''
    stack = plotutils.FigureStack()

    stack_img = ctx.get_stack_image(nsigma=3, preprocess=True, nsigma_connected=True)
    projection = ctx.get_projection(stack_img)

    ms_match_results, link_builders = ctx.get_match_result()

    for link_builder in link_builders.get_all()[:]:
        if scale is None or scale == link_builder.get_scale():
            def do_plot(fig):
                ax = fig.subplots()
                links = link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size)

                wiseutils.plot_links_map(ax, stack_img, projection, links, color_style=color_style, 
                                           map_cmap=map_cmap, vector_width=vector_width, 
                                           link_id_label=False, **kargs)

                scale = projection.mean_pixel_scale() * link_builder.get_scale()

                if isinstance(title, bool) and title is True:
                    ax.set_title(ax.get_title() + '\nVelocity map at scale %.2f mas.' % scale)
                if isinstance(title, str):
                    ax.set_title(title)

            stack.add_replayable_figure("Velocity field", do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))

    stack.show()


def preview_detection_stack(ctx, stack_detection_name, count_threshold=0, ms_set=None, 
                            date_filter=None, show_regions=[]):
    ''' Plot detection in stack'''
    stack = plotutils.FigureStack()

    stack_img, img_snr, img_count = load_detection_stack_image(ctx, stack_detection_name, preprocess=True)

    img_snr.data[img_count.data < count_threshold] = 0
    img_count.data[img_count.data < count_threshold] = 0

    prj = ctx.get_projection(stack_img)

    fig, ax = stack.add_subplots("Stack image")
    plotutils.imshow_image(ax, stack_img, projection=prj)
    for region in show_regions:
        plotutils.plot_region(ax, region, prj, text=True)

    fig, ax1 = stack.add_subplots("Stack detection SNR")
    plotutils.imshow_image(ax1, img_snr, projection=prj, cmap=plotutils.get_cmap('jet'))
    for region in show_regions:
        plotutils.plot_region(ax1, region, prj, text=True)

    fig, ax2 = stack.add_subplots("Stack detection count")
    plotutils.imshow_image(ax2, img_count, projection=prj, cmap=plotutils.get_cmap('jet'))
    for region in show_regions:
        plotutils.plot_region(ax2, region, prj, text=True)

    if ms_set is not None:
        colorbar_setting = plotutils.ColorbarSetting(cmap='jet', ticks_locator=mdates.AutoDateLocator(),
                                                     ticks_formatter=mdates.DateFormatter('%m/%y'))

        def feature_filter(feature):
            if img_snr.data[tuple(feature.get_coord())] == 0:
                return False
            if date_filter is not None and not date_filter(feature.get_epoch()):
                return False
            return True

        ms_set = wds.MultiScaleImageSet.from_file(os.path.join(ctx.get_data_dir(), j),
                                                  feature_filter=feature_filter)
        plot_ms_set_map(ax1, None, ms_set, prj, colorbar_setting=colorbar_setting)
        plot_ms_set_map(ax2, None, ms_set, prj, colorbar_setting=colorbar_setting)

        add_features_tooltip(stack, ax1, ms_set.features_iter(), projection=prj, epoch=True, tol=1)
        add_features_tooltip(stack, ax2, ms_set.features_iter(), projection=prj, epoch=True, tol=1)

    stack.show()


def create_poly_region(ctx, img, features=None):
    ''' Create a region file '''
    from libwise.app import PolyRegionEditor
    prj = ctx.get_projection(img)

    editor = PolyRegionEditor.PolyRegionEditor(img, prj=prj, current_folder=ctx.get_data_dir())

    if features is not None:
        wiseutils.plot_features(editor.ax, features, mode='com', c=plotutils.black, s=20)
    editor.start()


def list_saved_results(ctx):
    ''' List all saved results '''
    stack_img = ctx.get_stack_image(preprocess=True)
    projection = ctx.get_projection(stack_img)

    ext = '.set.dat'
    data = []
    header = ["Name", "Number of epochs", "First epoch", "Last epoch", "Kinematic", "Scales"]
    for file in glob.glob(os.path.join(ctx.get_data_dir(), '*', '*' + ext)):
        name = os.path.basename(os.path.dirname(file))
        if not os.path.basename(file) == name + ext:
            continue
        img_set = imgutils.ImageSet.from_file(file, projection)
        epochs = img_set.get_epochs()
        n = len(epochs)
        first, last = epochs[0], epochs[-1]
        link_builder_files = glob.glob(os.path.join(os.path.dirname(file), '*.ms.dfc.dat'))
        scales = [f.rsplit('_')[-1].split('.')[0] for f in link_builder_files]

        data.append([name, n, first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"), bool(len(scales)), ", ".join(scales)])

    print nputils.format_table(data, header)


def list_tasks():
    data = []
    header = ["Name", "Description"]
    for name, value in inspect.getmembers(sys.modules[__name__]):
        doc = inspect.getdoc(value)
        if inspect.isfunction(value) and doc is not None and not name.startswith('_'):
            data.append([name, doc.splitlines()[0]])

    print nputils.format_table(data, header)


def build_detection_stack_image(ctx, preprocess=True, smooth=False):
    stack_mgr = imgutils.StackedImageManager()
    stack_mgr_snr = imgutils.StackedImageManager()
    stack_mgr_count = imgutils.StackedImageManager()
    for file in ctx.files:
        img = ctx.open_file(file)
        # post processing and alignwill happens there
        results = ctx.detection(img)

        stack_mgr.add(img)

        stack_mgr_file_snr = imgutils.StackedImageManager()
        stack_mgr_file_count = imgutils.StackedImageManager()

        for segments in results:
            img = segments.get_img()
            # img.data = img.data / segments.get_rms_noise()
            # img.data[segments.get_labels() == 0] = 0
            stack_mgr_file_snr.add(img, action='mean')

            img.data = (segments.get_labels() > 0).astype(np.float)
            stack_mgr_file_count.add(img, action='max')

        stack_mgr_snr.add(stack_mgr_file_snr.get(), action='add')
        stack_mgr_count.add(stack_mgr_file_count.get(), action='add')

    stack_img = stack_mgr.get()
    img_snr = stack_mgr_snr.get()
    img_count = stack_mgr_count.get()

    if smooth:
        img_snr.data = nputils.smooth(img_snr.data, 2, mode='same')
        img_count.data = nputils.smooth(img_count.data, 2, mode='same')
    return stack_img, img_snr, img_count


def save_detection_stack_image(ctx, name, preprocess=True, smooth=False):
    stack_img, img_snr, img_count = build_detection_stack_image(ctx, preprocess=preprocess,
        smooth=smooth)

    img_snr.save_to_fits(os.path.join(ctx.get_data_dir(), name + '.stack.detection_snr.fits'))
    img_count.save_to_fits(os.path.join(ctx.get_data_dir(), name + '.stack.detection_count.fits'))
    stack_img.save_to_fits(os.path.join(ctx.get_data_dir(), name + '.stack.full.fits'))

    return stack_img, img_snr, img_count


def load_detection_stack_image(ctx, name, preprocess=True):
    stack_img = imgutils.StackedImage.from_file(os.path.join(ctx.get_data_dir(), name + '.stack.full.fits'))
    ctx.pre_process(stack_img)
    img_snr = imgutils.StackedImage.from_file(os.path.join(ctx.get_data_dir(), name + '.stack.detection_snr.fits'))
    ctx.pre_process(img_snr)
    img_count = imgutils.StackedImage.from_file(os.path.join(ctx.get_data_dir(), name + '.stack.detection_count.fits'))
    ctx.pre_process(img_count)

    return stack_img, img_snr, img_count


def stack_cross_correlation(ctx, config, debug=0, nwise=2, stack=None):
    scc_result = scc.StackCrossCorrelation(config, debug=debug, stack=stack)
    all_files = nputils.nwise(ctx.files, nwise)

    for pair in all_files:
        img1 = ctx.open_file(pair[0])
        img2 = ctx.open_file(pair[-1])

        prj = ctx.get_projection(img1)

        delta_t, velocity_pix, tol_pix = scc_result.get_velocity_resolution(prj, img1, img2)

        if not nputils.in_range(tol_pix, config.get("tol_pix_range")):
            print "-> Skip: Not in the allowed range of pixel velocity resolution:", tol_pix
            continue

        res1 = ctx.detection(img1, filter=config.get("filter1"))
        res2 = ctx.detection(img2, filter=config.get("filter2"))

        print [k.size() for k in res1]

        scc_result.process(prj, res1, res2)

    return scc_result


def bootstrap_scc(ctx, config, output_dir, n, append=False, seperate_scales=False):
    nwise = 2
    random_shift = config.get("img_rnd_shift")

    if config.get("shuffle") == config.get("rnd_pos_shift"):
        print "Configuration Error: either 'shuffle' or 'rnd_pos_shift' need to be set"
        return

    all_files = list(ctx.files)

    prj = ctx.get_projection(ctx.open_file(all_files[0]))

    all_res1 = dict()
    all_res2 = dict()
    all_epochs = []

    for file1 in ctx.files:
        img1 = ctx.open_file(file1)
        img1.data = nputils.shift2d(img1.data, np.random.uniform(-random_shift, random_shift, 2))

        img2 = ctx.open_file(file1)
        img2.data = nputils.shift2d(img2.data, np.random.uniform(-random_shift, random_shift, 2))

        res1 = ctx.detection(img1, filter=config.get("filter1"))
        res2 = ctx.detection(img2, filter=config.get("filter2"))

        # print [k.size() for k in res1]
        # print [k.size() for k in res2]

        all_res1[file1] = res1
        all_res2[file1] = res2
        all_epochs.append(img1.get_epoch())

    t = time.time()

    # all_segments2_img = dict()
    # for file, segments2 in all_res2.items():
    #     all_segments2_img[file] = [k.get_img().data.copy() for k in segments2]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    files = os.listdir(output_dir)
    if append and len(files) > 0:
        if seperate_scales and os.path.isdir(os.path.join(output_dir, files[0])):
            files = os.listdir(os.path.join(output_dir, files[0]))
        all_i = sorted([int(os.path.splitext(file)[0].split('_')[-1]) for file in files])
        if len(all_i) == 0:
            start = 0
        else:
            start = all_i[-1] + 1
    else:
        start = 0

    for i in range(n):
        eta = ""
        if i > 0:
            remaining = (np.round((time.time() - t) / float(i) * (n - i)))
            eta = " (ETA: %s)" % time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
        print "Run %s / %s%s" % (i + 1, n, eta)

        if config.get("shuffle"):
            # np.random.shuffle(all_files)
            shuffled = nputils.permutation_no_succesive(all_files)
            files_pair = nputils.nwise(shuffled, nwise)
        else:
            files_pair = nputils.nwise(all_files, nwise)
        epochs_pair = nputils.nwise(all_epochs, nwise)

        scc_result = scc.StackCrossCorrelation(config, verbose=False)

        for shuffled_pair, epoch_pair in zip(files_pair, epochs_pair):
            res1 = all_res1[shuffled_pair[0]]
            res2 = all_res2[shuffled_pair[-1]]

            # for segments2, segments2_img in zip(res2, all_segments2_img[shuffled_pair[-1]]):
            #     segments2.get_img().data = nputils.shift2d(segments2_img, 
            #                                     np.random.uniform(-random_shift, random_shift, 2))

            res1.epoch = epoch_pair[0]
            res2.epoch = epoch_pair[-1]

            delta_t, velocity_pix, tol_pix = scc_result.get_velocity_resolution(prj, res1, res2)

            if tol_pix <= 4 or tol_pix >= 20:
                # print "-> Skip"
                continue

            scc_result.process(prj, res1, res2)

        if seperate_scales:
            for scale, gncc_map in scc_result.get_mean_ncc_scales(smooth_len=1).items():
                save_dir = os.path.join(output_dir, "scale_%s" % scale)
                
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                imgutils.Image(gncc_map).save_to_fits(os.path.join(save_dir, "gncc_map_%s.fits" % (start + i)))
        else:
            gncc_map = scc_result.get_global_ncc(smooth_len=1)
            imgutils.Image(gncc_map).save_to_fits(os.path.join(output_dir, "gncc_map_%s.fits" % (start + i)))

    print "Done"


def test_load_3c120_config():
    import astropy.cosmology as cosmology

    cosmology.default_cosmology.set(cosmology.FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27))

    BASE_DIR = os.path.expanduser("~/data/3c120/mojave")

    ctx = wiseutils.AnalysisContext()

    # data configuration
    ctx.config.data.data_dir = os.path.join(BASE_DIR, "run001")
    ctx.config.data.object_z = 0.033

    def get_bg(ctx, img):
        return img.data[:, :800]

    def build_mask(ctx):
        stack = ctx.get_stack_image(nsigma=4)
        segments = wise.SegmentedImages(stack)
        segments.connected_structure()
        mask = segments.get_mask(segments.sorted_list()[-1])
        mask = imgutils.Image.from_image(stack, mask)
        return mask

    def pre_process(ctx, img):
        img.crop([3, -10], [-22, 3], projection=ctx.get_projection(img))
    #     img.crop([10, -20], [-40, 10], projection=ctx.get_projection(img))
    #     img.crop([2, -4], [-5, 2], projection=ctx.get_projection(img))

    def get_core(ctx, img):
        bg = ctx.get_bg(img)
        ctx.pre_process(img)
        cmp_y = lambda x, y: cmp(x.get_coord()[1], y.get_coord()[1])
        core = wiseutils.align_image_on_core(img, bg, cmp_y)
        core = ctx.get_projection(img).p2s(plotutils.p2i(core))

        return core
        
    ctx.config.data.bg_fct = get_bg
    ctx.config.data.mask_fct = build_mask
    ctx.config.data.pre_process_fct = pre_process
    ctx.config.data.core_offset_filename = "mojave_core.dat"
    ctx.config.data.core_offset_fct = get_core

    # finder configuration
    ctx.config.finder.min_scale = 1
    ctx.config.finder.max_scale = 4
    ctx.config.finder.alpha_threashold = 3
    ctx.config.finder.exclude_noise = False
    ctx.config.finder.ms_dec_klass = wds.WaveletMultiscaleDecomposition

    files = glob.glob(os.path.join(BASE_DIR, "icn/**.icn.fits"))
    start_date = end_date = filter_dates = None
    start_date = datetime.datetime(2002, 1, 1)
    end_date = datetime.datetime(2006, 1, 1)
    # filter_dates = [datetime.datetime(2000, 12, 30)]
    ctx.files = imgutils.fast_sorted_fits(files, start_date=start_date, end_date=end_date, filter_dates=filter_dates)

    return ctx

def test_3c120_plot_distance_from_core():
    ctx = test_load_3c120_config()

    load(ctx, "ms_sep5", min_link_size=5)

    start_date = datetime.datetime(2006, 1, 1)
    end_date = datetime.datetime(2012, 1, 1)
    date_filter = wfeatures.DateFilter(start_date=start_date, end_date=end_date)
    date_filter2 = wfeatures.DateFilter(start_date=None, end_date=start_date)
    dfc_filter = wfeatures.DfcFilter(0, 10, u.mas)
    dfc_filter2 = wfeatures.DfcFilter(5, 25, u.mas)

    feature_filter = (date_filter & dfc_filter) | (date_filter2 & dfc_filter2) 

    plot_separation_from_core(ctx, scale=None, feature_filter=None, pa=False, fit_fct=nputils.LinearFct)


def test_3c120_view_links():
    ctx = test_load_3c120_config()

    load(ctx, "ms_sep5", min_link_size=5)

    start_date = datetime.datetime(2006, 1, 1)
    end_date = datetime.datetime(2012, 1, 1)
    date_filter = wfeatures.DateFilter(start_date=start_date, end_date=end_date)
    date_filter2 = wfeatures.DateFilter(start_date=None, end_date=start_date)
    dfc_filter = wfeatures.DfcFilter(0, 10, u.mas)
    dfc_filter2 = wfeatures.DfcFilter(5, 25, u.mas)

    feature_filter = (date_filter & dfc_filter) | (date_filter2 & dfc_filter2) 

    view_links(ctx, scale=8, feature_filter=feature_filter)


def test_3c120_plot_all_features():
    ctx = test_load_3c120_config()

    start_date = datetime.datetime(2006, 1, 1)
    end_date = datetime.datetime(2012, 1, 1)
    date_filter = wfeatures.DateFilter(start_date=start_date, end_date=end_date)
    date_filter2 = wfeatures.DateFilter(start_date=None, end_date=start_date)
    dfc_filter = wfeatures.DfcFilter(0, 10, u.mas)
    dfc_filter2 = wfeatures.DfcFilter(5, 25, u.mas)

    feature_filter = (date_filter & dfc_filter) | (date_filter2 & dfc_filter2) 

    # load(ctx, "ms_sep5", min_link_size=5)
    # detection_all(ctx)
    load(ctx, "detection_test")

    plot_all_features(ctx, feature_filter=feature_filter, pa=True)


def test_3c120_view_wds():
    ctx = test_load_3c120_config()

    detection_all(ctx)

    view_wds(ctx, scale=8)


if __name__ == '__main__':
    # test_3c120_plot_distance_from_core()
    # test_3c120_plot_all_features()
    # list_saved_results(test_load_3c120_config())
    # test_3c120_view_links()
    test_3c120_view_wds()
