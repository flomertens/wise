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
import project
import wiseutils
import features as wfeatures

from libwise import plotutils, nputils, imgutils

import astropy.units as u
import astropy.constants as const

unit_c = u.core.Unit("c", const.c, doc="light speed")


def build_final_dfc(ctx, merge_file, final_sep):
    '''Build a final separation file from a merge file.
    
    Merge file shall be located in ctx.get_data_dir(). 
    One final component per line, described by a list of link id separated by a ','.

    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`

    merge_file : str
        Merge file name.
    final_sep : str
        Name of the final separation file.


    .. _tags: task_matching
    '''
    merge_file = os.path.join(ctx.get_data_dir(), merge_file)
    final_sep_file = os.path.join(ctx.get_data_dir(), final_sep)

    ref_img = ctx.get_ref_image()
    ctx.pre_process(ref_img)
    projection = ctx.get_projection(ref_img)

    merged_builder = matcher.MergedFeatureLinkBuilder(ctx.result.link_builder, merge_file)

    merged_builder.to_file(final_sep_file, projection)


def info_files(ctx):
    '''Print List of selected files with information on beam and pixel scales
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`


    .. _tags: task_general
    '''
    data_beam = []
    header = ["File", "Date", "Shape", "Pixel scale", "Beam"]
    data_table = []
    for file in ctx.files:
        img = ctx.open_file(file)
        beam = img.get_beam()
        date = img.get_epoch(str_format='%Y-%m-%d')
        k = ctx.get_projection(img).mean_pixel_scale()
        u = ctx.get_projection(img).get_unit()
        if isinstance(beam, imgutils.GaussianBeam):
            b = [k * beam.bmin, k * beam.bmaj, beam.bpa]
            # beam_str = "%.3f, %.3f, %.3f" % (b[0] * u, b[1] * u, b[2])
            beam_str = "{0:.3f}, {1:.3f}, {2:.2f}".format(b[0] * u, b[1] * u, b[2])
            data_beam.append([b[0], b[1], b[2]])
        else:
            beam_str = str(beam)
        shape_str = "%sx%s" % img.data.shape
        data_table.append([os.path.basename(file), date, shape_str, "{0:.3f}".format(k * u), beam_str])

    print nputils.format_table(data_table, header)
    print "Number of files: %s" % (len(ctx.files))
    if len(data_beam) > 0:
        print "Mean beam: Bmin: %.3f, Bmaj: %.3f, Angle:%.2f" % tuple(np.mean(data_beam, axis=0))


def set_stack_image_as_ref(ctx, nsigma=3, nsigma_connected=True):
    '''Set the reference image from a stacked images
        
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    nsigma : int, optional
        Clip data below nsigma time the background level.
    nsigma_connected : bool, optional
        Keep only the brightest isolated structure.


    .. _tags: task_conf_helper
    '''
    stack_img = ctx.build_stack_image(preprocess=False, nsigma=nsigma, 
                                      nsigma_connected=nsigma_connected)

    ctx.set_ref_image(stack_img)


def set_mask_from_stack_img(ctx, nsigma=3, nsigma_connected=True):
    """Set the mask image from a stacked images
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    nsigma : int, optional
        Clip data below nsigma time the background level.
    nsigma_connected : bool, optional
        Keep only the brightest isolated structure.


    .. _tags: task_conf_helper
    """
    def mask_fct(ctx):
        stack_img = ctx.build_stack_image(preprocess=False, nsigma=nsigma, 
                                          nsigma_connected=nsigma_connected)

        return stack_img

    ctx.save_mask_file(mask_fct)


def info_files_delta(ctx, delta_time_unit=u.day, angular_velocity_unit=u.mas / u.year,
                     proper_velocity_unit=unit_c):
    '''Print List of selected pair of files with information on velocity resolution 
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    delta_time_unit : :class:`astropy.units.Unit`, optional
    angular_velocity_unit : :class:`astropy.units.Unit`, optional
    proper_velocity_unit : :class:`astropy.units.Unit`, optional


    .. _tags: task_general
    '''
    all_delta_time = []
    all_velocity_c_px = []
    all_velocity_px = []
    data = []
    header = ["Date 1", "Date 2", "Delta (%s)" % delta_time_unit, 
            "Angular vel. res. (%s)" % angular_velocity_unit, 
            "Proper vel. res. (%s)" % proper_velocity_unit]
    has_distance = ctx.config.data.object_distance or ctx.config.data.object_z
    for file1, file2 in nputils.pairwise(ctx.files):
        img1 = ctx.open_file(file1)
        img2 = ctx.open_file(file2)
        prj = ctx.get_projection(img1)
        date1 = img1.get_epoch(str_format='%Y-%m-%d')
        date2 = img2.get_epoch(str_format='%Y-%m-%d')
        if not isinstance(date1, datetime.date):
            print "Images have no date information"
            return
        delta_t = ((img2.get_epoch() - img1.get_epoch()).total_seconds() * u.second).to(delta_time_unit)
        all_delta_time.append(delta_t)

        velocity_px = prj.angular_velocity([0, 0], [0, 1], delta_t).to(angular_velocity_unit)
        all_velocity_px.append(velocity_px)

        if has_distance:
            velocity_c_px = prj.proper_velocity([0, 0], [0, 1], delta_t).to(proper_velocity_unit)
            all_velocity_c_px.append(velocity_c_px)

        if has_distance:
            data.append([date1, date2, delta_t.value, velocity_px.value, velocity_c_px.value])
        else:
            data.append([date1, date2, delta_t.value, velocity_px.value, '-'])

        # print "{0} -> {1}: Delta time: {2}, Velocity resolution: {3:.3f}, {4:.3f}".format(*data)

    all_delta_time = nputils.quantities_to_array(all_delta_time)
    all_velocity_px = nputils.quantities_to_array(all_velocity_px)

    print nputils.format_table(data, header)

    print "Mean Delta time: %s +- %s" % (np.mean(all_delta_time), np.std(all_delta_time))
    print "Mean Velocity resolution: %s +- %s" % (np.mean(all_velocity_px), np.std(all_velocity_px))
    if has_distance:
        all_velocity_c_px = nputils.quantities_to_array(all_velocity_c_px)
        print "Mean Velocity resolution: %s +- %s" % (np.mean(all_velocity_c_px), np.std(all_velocity_c_px))


def detection_all(ctx, filter=None):
    '''Run the Segmented wavelet decomposition on all selected files
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    filter : :class:`wise.features.FeatureFilter`, optional


    .. _tags: task_detection
    '''
    ctx.result = project.AnalysisResult(ctx.config)
    for file in ctx.files:
        img = ctx.open_file(file)
        res = ctx.detection(img, filter=filter)
        ctx.result.add_detection_result(img, res)


def match_all(ctx, filter=None):
    '''Run matching on all selected files
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    filter : :class:`wise.features.FeatureFilter`, optional


    .. _tags: task_detection
    '''
    ctx.result = project.AnalysisResult(ctx.config)
    prev_result = None
    prev_img = None
    match_ratio_list = []
    for i, file in enumerate(ctx.files):
        img = ctx.open_file(file)
        res = ctx.detection(img, filter=filter)
        result_copy = res.copy()

        ctx.result.add_detection_result(img, result_copy)

        if prev_result is not None:
            print "Matching %s vs %s (%s / %s)" % (prev_img, img, i, len(ctx.files) - 1)
            match_res = ctx.match(prev_result, res)
            ctx.result.add_match_result(match_res)

            for match_res_scale in match_res:
                n_segments = match_res_scale.segments1.size()
                match_ratio = (match_res_scale.get_match().size()) / (float(n_segments))
                # print match_res_scale.get_match().size(), n_segments
                match_ratio_list.append(match_ratio)
            # print match_ratio_list

        prev_result = result_copy
        prev_img = img

    print "Match ratio stat:", nputils.stat(match_ratio_list)


def bootstrap_matching(ctx, n=100, filter=None, cb_post_match=None):
    all_res = dict()
    all_epochs = []
    all_imgs = dict()
    for i, file in enumerate(ctx.files):
        img = ctx.open_file(file)
        all_epochs.append(img.get_epoch())

        res = ctx.detection(img, filter=filter)
        all_res[img.get_epoch()] = res
        all_imgs[img.get_epoch()] = img

    t = time.time()
    all_match_ratio = []
    for i in range(n):
        eta = ""
        if i > 0:
            remaining = (np.round((time.time() - t) / float(i) * (n - i)))
            eta = " (ETA: %s)" % time.strftime("%H:%M:%S", time.localtime(time.time() + remaining))
        print "Run %s / %s%s" % (i + 1, n, eta)
        
        shuffled = nputils.permutation_no_succesive(all_epochs)
        match_ratio_list = []
        match_results = project.AnalysisResult(ctx.config)

        for epoch1, epoch2 in nputils.pairwise(shuffled):
            res1 = all_res[epoch1].copy()
            res2 = all_res[epoch2].copy()
            # print "Matching:", res1.get_epoch(), "vs", res2.get_epoch()

            res1.epoch = epoch1
            for segments in res1:
                segments.epoch = epoch1

            res2.epoch = epoch2
            for segments in res2:
                segments.epoch = epoch2

            full_match_res = ctx.match(res1, res2, verbose=False)
            match_results.add_match_result(full_match_res)
            match_results.add_detection_result(all_imgs[epoch1], res1)

            for match_res in full_match_res:
                n_segments = match_res.segments1.size()
                match_ratio = (match_res.get_match().size()) / (float(n_segments))
                match_ratio_list.append(match_ratio)

        all_match_ratio.append(np.mean(match_ratio_list))

        if cb_post_match is not None:
            cb_post_match(shuffled, match_results)

        # print "Match ratio stat:", nputils.stat(match_ratio_list)

    # print all_match_ratio

    print "Match ratio stat:", nputils.stat(all_match_ratio)


def save(ctx, name, coord_mode='com', measured_delta=True):
    '''Save current result to disk
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    name : str
        Prefix name for the save files
    coord_mode : str, optional
    measured_delta : bool, optional


    .. _tags: task_general
    '''
    if not ctx.result.has_detection_result():
        print "No result to save"
        return

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)

    path = os.path.join(ctx.get_data_dir(), name)
    if not os.path.exists(path):
        os.mkdir(path)
    ctx.result.detection.to_file(os.path.join(path, "%s.ms.dat" % name), projection, coord_mode=coord_mode)
    ctx.result.link_builder.to_file(os.path.join(path, name), projection, 
                               coord_mode=coord_mode, measured_delta=measured_delta)
    ctx.result.image_set.to_file(os.path.join(path, "%s.set.dat" % name), projection)
    ctx.result.config.to_file(os.path.join(path, "%s.conf" % name))


def load(ctx, name, projection=None, merge_with_previous=False, min_link_size=2):
    '''Load result from files
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    name : str
        Prefix name of the saved files
    projection : :class:`libwise.imgutils.Projection`, optional
        If not provided, default Projection will be used
    merge_with_previous : bool, optional
        If True, this result will be added to current result
    min_link_size : int, optional
        Filter out links with size < min_link_size
    

    .. _tags: task_general
    '''

    if projection is None:
        ref_img = ctx.get_ref_image()
        projection = ctx.get_projection(ref_img)
        
    path = os.path.join(ctx.get_data_dir(), name)
    if not os.path.isdir(path):
        print "No results saved with name %s" % name
        return

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
        ctx.result = project.AnalysisResult(ctx.config)
        ctx.result.image_set = image_set
        ctx.result.detection = detection
        ctx.result.link_builder = link_builder
        ctx.result.ms_match_results = ms_match_results


def view_all(ctx, preprocess=True, show_mask=True, show_regions=[], save_filename=None, 
            align=True, **kwargs):
    '''Preview all images
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    preprocess : bool, optional
        If True, run the pre_process fct on each images
    show_mask : bool, optional
        If True, show the saved mask in the map
    show_regions : list of :class:`libwise.imgutils.Region`, optional
        Plot the regions
    save_filename : str, optional
        If not None, the resulting maps will be saved in the project data dir
        with this filename.
    **kwargs :
        Arguments to pass to :func:`libwise.plotutils.imshow_image`


    .. _tags: task_general
    '''
    if len(ctx.files) == 0:
        print "No files selected"
        return

    stack = plotutils.FigureStack()

    def do_plot(fig, file):
        img = ctx.open_file(file)
        ax = fig.subplots()
        if align:
            ctx.align(img)
        if preprocess:
            ctx.pre_process(img)
        prj = ctx.get_projection(img)
        plotutils.imshow_image(ax, img, projection=prj, **kwargs)
        if not align:
            core_offset = ctx.get_core_offset()
            if core_offset is not None:
                x, y = core_offset.get(img.get_epoch())
                x, y = prj.s2p([x, y])
                # print core_offset, x, y
                ax.scatter(x, y, marker='*', s=40, c=plotutils.black)
        # if not preprocess:
        #     bg_mask = imgutils.Image(np.zeros_like(img.data, dtype=np.int8))
        #     bg = ctx.get_bg(bg_mask)
        #     if bg is not None and isinstance(bg, np.ndarray):
        #         bg.fill(1)
        if ctx.get_mask() is not None and show_mask is True:
            mask = ctx.get_mask()
            ctx.pre_process(mask)
            ax.contour(mask.data, [0.5])
        for region in show_regions:
            plotutils.plot_region(ax, region, prj, text=True)

    for file in ctx.files:
        if isinstance(file, basestring):
            name = os.path.basename(file)
        else:
            name = str(file)
        stack.add_replayable_figure(name, do_plot, file)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()


def view_stack(ctx, preprocess=True, nsigma=0, nsigma_connected=False, show_mask=True, show_regions=[], 
                  intensity_colorbar=False, save_filename=None, **kwargs):
    '''Preview the stack image
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    preprocess : bool, optional
        If True, run the pre_process fct on each images
    nsigma : int, optional
        Clip bg below nsigma level
    nsigma_connected : bool, optional
        If True, keep only the brightest connected structure
    show_mask : bool, optional
        If True, show the saved mask in the map
    show_regions : list of :class:`libwise.imgutils.Region`, optional
        Plot the regions
    intensity_colorbar : bool, optional
        Add an intensity colorbar
    save_filename : str, optional
        If not None, the resulting maps will be saved in the project data dir
        with this filename.
    **kwargs :
        Arguments to pass to :func:`libwise.plotutils.imshow_image`


    .. _tags: task_general
    '''

    stack = plotutils.FigureStack()

    stack_img = ctx.build_stack_image(preprocess=preprocess, nsigma=nsigma, 
                                      nsigma_connected=nsigma_connected)

    def do_plot(fig):
        ax = fig.subplots()

        plotutils.imshow_image(ax, stack_img, projection=ctx.get_projection(stack_img), 
                               intensity_colorbar=intensity_colorbar, **kwargs)
        if ctx.get_mask() is not None and show_mask is True:
            ax.contour(ctx.get_mask().data, [0.5])
        for region in show_regions:
            plotutils.plot_region(ax, region, ctx.get_projection(stack_img), text=False, fill=True)

    stack.add_replayable_figure("Stack Image", do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()


def view_wds(ctx, title=True, num=True, scales=None, save_filename=None, **kwargs):
    '''Plot WDS decomposition
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    title : bool, optional
        If True, add a title to the map
    num : bool, optional
        If True, annotate each segments
    scales : list or int, optional
        Plot only for scale or list of scales (in pixel)
    save_filename : TYPE, optional
        Description
    **kwargs :
        Description


    .. _tags: task_detection
    '''
    if not ctx.result.has_detection_result():
        print "No result found. Run detect_all() or match_all() first."
        return

    detection_result = ctx.get_detection_result()

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)
    scales = _get_scales(scales, ctx.result.get_scales())

    if not detection_result.is_full_wds():
        print "This task require full WDS result. Please run detection again."
        return

    if len(scales) == 0:
        print "No result found for this scales. Available scales:", ctx.result.get_scales()
        return

    stack = plotutils.FigureStack()

    def do_plot(fig, epoch):
        ms_result = detection_result.get_epoch(epoch)
        ax = fig.subplots(n=len(scales), reshape=False)
        for ax, scale in zip(ax, scales):
            segments = ms_result.get_scale(scale)
            wiseutils.imshow_segmented_image(ax, segments, title=title, 
                projection=projection, num=num, **kwargs)

    for epoch in detection_result.get_epochs():
        stack.add_replayable_figure("WDS %s" % epoch, do_plot, epoch)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()


def view_all_features(ctx, scales, region_list=None, legend=False, feature_filter=None, 
                      save_filename=None, **img_kargs):
    ''' Plot all features location
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    scales : int or list
        Filter the results to scale or list of scales (in pixel)
    region_list : list of :class:`libwise.imgutils.Region`, optional
        Plot the regions, and set the features color according to the region
    legend : bool, optional
        Add a legend 
    feature_filter : :class:`wise.features.FeatureFilter`, optional
        Filter the results
    save_filename : TYPE, optional
        If not None, the resulting maps will be saved in the project data dir
        with this filename.
    **img_kargs
        Additional arguments to pass to :func:`libwise.plotutils.imshow_image`


    .. _tags: task_detection
    '''
    if not ctx.result.has_detection_result():
        print "No result found. Run detect_all() or match_all() first."
        return

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)
    scales = _get_scales(scales, ctx.result.get_scales())

    if len(scales) == 0:
        print "No result found for this scales. Available scales:", ctx.result.get_scales()
        return

    data = wiseutils.SSPData.from_results(ctx.get_result(), projection, scales)
    if feature_filter is not None:
        data.filter(feature_filter)
    if region_list is not None:
        data.add_col_region(region_list)

    stack = plotutils.FigureStack()

    def do_plot(fig):
        ax_all = fig.subplots()

        plotutils.imshow_image(ax_all, ref_img, projection=projection, **img_kargs)

        if region_list is not None:
            for region, gdata in data.df.groupby('region'):
                features = wds.DatedFeaturesGroupScale(0, features=gdata.features.values)

                wiseutils.plot_features(ax_all, features, mode='com', c=region.get_color(), label=region.get_name())
                plotutils.plot_region(ax_all, region, projection=projection, text=False, 
                                      color=region.get_color(), fill=True)
        else:
            features = wds.DatedFeaturesGroupScale(0, features=data.df.features.values)
            wiseutils.plot_features(ax_all, features, mode='com')

        if legend:
            ax_all.legend(loc='best')

    stack.add_replayable_figure("All features", do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()


def plot_separation_from_core(ctx, scales=None, feature_filter=None, min_link_size=2, title=True, 
                            pa=False, snr=False, num=False, fit_fct=None, save_filename=None, **kwargs):
    """Plot separation from core with time
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    scales : int or list, optional
        Filter the results to scale or list of scales (in pixel)
    feature_filter : :class:`wise.features.FeatureFilter`, optional
        Filter the results
    min_link_size : int, optional
        Filter out links with size < min_link_size
    title : bool, optional
    pa : bool, optional
        Additionaly plot the PA vs epoch (only pa or snr)
    snr : bool, optional
        Additionaly plot the snr vs epoch (only pa or snr)
    num : bool, optional
        Annotate each links
    fit_fct : :class:`libwise.nputils.AbstractFct`, optional
        Fit each links with provided fct
    save_filename : str, optional
        If not None, the resulting maps will be saved in the project data dir
        with this filename.
    **kwargs :
        Additional arguments to pass to :func:`wise.wiseutils.plot_links_dfc`


    .. _tags: task_matching
    """
    if not ctx.result.has_match_result():
        print "No result found. Run match_all() first."
        return

    stack = plotutils.FigureStack()

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)
    scales = _get_scales(scales, ctx.result.get_scales())

    if len(scales) == 0:
        print "No result found for this scales. Available scales:", ctx.result.get_scales()
        return

    ms_match_results, link_builders = ctx.get_match_result()

    fit_results = dict()

    def do_plot(fig, scale):
        link_builder = link_builders.get_scale(scale)

        links = list(link_builder.get_links(feature_filter=feature_filter, 
                        min_link_size=min_link_size))
        ax = fig.subplots(nrows=1 + int(pa or snr), reshape=False, sharex=True)

        if fit_fct:
            plotutils.checkargs(kwargs, 'ls', '--')
        wiseutils.plot_links_dfc(ax[0], projection, links, num=num, **kwargs)
        if fit_fct:
            fit_result = wiseutils.plot_links_dfc_fit(ax[0], projection, links, fit_fct, lw=2)
            fit_results.update(fit_result)

        if title:
            ax[0].set_title("Separation from core at scale %s" % projection.get_sky(scale))

        ax[-1].set_xlabel("Epoch (years)")
        ax[0].set_ylabel("Distance from core (mas)")

        if pa:
            wiseutils.plot_links_pa(ax[1], projection, links, **kwargs)
            ax[-1].set_xlabel("Separation PA (deg)")
        elif snr:
            wiseutils.plot_links_snr(ax[1], projection, links, **kwargs)
            ax[1].set_ylabel("Wavelet Coef SNR")

    for scale in scales:
        stack.add_replayable_figure("DFC scale %s" % scale, do_plot, scale)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()

    if fit_fct:
        return fit_results


def plot_all_features(ctx, scales=None, pa=False, feature_filter=None, save_filename=None):
    """Plot all features distance from core with time
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    scales : int or list, optional
        Filter the results to scale or list of scales (in pixel)
    pa : bool, optional
        Additionaly plot the PA vs epoch
    feature_filter : :class:`wise.features.FeatureFilter`, optional
        Filter the results
    save_filename : str, optional
        If not None, the resulting maps will be saved in the project data dir
        with this filename.


    .. _tags: task_detection
    """
    if not ctx.result.has_detection_result():
        print "No result found. Run detect_all() or match_all() first."
        return

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)
    scales = _get_scales(scales, ctx.result.get_scales())

    if len(scales) == 0:
        print "No result found for this scales. Available scales:", ctx.result.get_scales()
        return

    data = wiseutils.SSPData.from_results(ctx.get_result(), projection, scales)
    if feature_filter is not None:
        data.filter(feature_filter)

    stack = plotutils.FigureStack()

    def do_plot(fig):
        ax = fig.subplots(nrows=1 + int(pa), reshape=False, sharex=True)
        colors = plotutils.ColorSelector()

        for scale, gdata in data.df.groupby('scale'):
            color = colors.get()
            features = wds.DatedFeaturesGroupScale(scale, features=gdata.features.values)

            wiseutils.plot_features_dfc(ax[0], projection, features, c=color)

            if pa:
                wiseutils.plot_features_pa(ax[1], projection, features, c=color)

        ax[-1].set_xlabel("Epoch (years)")
        ax[0].set_ylabel("Distance from core (mas)")
        
        if pa:
            ax[1].set_ylabel("PA (rad)")

    stack.add_replayable_figure("All features", do_plot)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()


def view_displacements(ctx, scale, feature_filter=None, title=True, save_filename=None):
    '''Plot individual match results at specified scale
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
        Description
    scales : int
        Scale in pixel
    feature_filter : :class:`wise.features.FeatureFilter`, optional
        Filter the results
    save_filename : str, optional
        If not None, the resulting maps will be saved in the project data dir
        with this filename.
    title : bool, optional


    .. _tags: task_matching
    '''
    if not ctx.result.has_match_result():
        print "No result found. Run match_all() first."
        return

    if scale not in ctx.result.get_scales():
        print "No result found for this scale. Available scales:", ctx.result.get_scales()
        return

    stack = plotutils.FigureStack()

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)

    ms_match_results, link_builders = ctx.get_match_result()
    all_epochs = ms_match_results.get_epochs()

    def do_plot(figure, epoch):
        ms_match_result = ms_match_results.get_epoch(epoch)
        match_result = ms_match_result.get_scale(scale)

        segments1, segments2, match, delta_info = match_result.get_all(feature_filter=feature_filter)
        
        ax = figure.subplots()

        axtitle = 'Displacements vector at scale %s' % projection.get_sky(segments1.get_scale())

        if isinstance(segments1, wds.SegmentedImages):
            if len(segments1.get_img().get_title()) > 0:
                axtitle += '\n%s' % segments1.get_img().get_title()
                axtitle += '\n%s' % segments2.get_img().get_title()
            bg = segments1.get_img()
        else:
            axtitle += '\n%s vs %s' % (segments1.get_epoch(), segments2.get_epoch())
            bg = ref_img

        wiseutils.plot_displacements(ax, segments1, segments2, delta_info, 
                             projection=projection, bg=bg)

        if title:
            ax.set_title(axtitle)

    for epoch in ms_match_results.get_epochs():
        stack.add_replayable_figure(epoch, do_plot, epoch)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()


def view_links(ctx, scales=None, feature_filter=None, min_link_size=2, map_cmap='YlGnBu_r',
               vector_width=6, title=True, color_style='link', save_filename=None, **kwargs):
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
    color_style : str, optional
        'link': one color per components
        'date': colors correspond to the epoch
        color_str: use color_str as color for all components
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

    stack = plotutils.FigureStack()
    scales = _get_scales(scales, ctx.result.get_scales())

    if len(scales) == 0:
        print "No result found for this scales. Available scales:", ctx.result.get_scales()
        return

    ref_img = ctx.get_ref_image()
    projection = ctx.get_projection(ref_img)

    ms_match_results, link_builders = ctx.get_match_result()

    def do_plot(fig, scale):
        ax = fig.subplots()

        link_builder = link_builders.get_scale(scale)
        links = link_builder.get_links(feature_filter=feature_filter, min_link_size=min_link_size)

        wiseutils.plot_links_map(ax, ref_img, projection, links, color_style=color_style, 
                                   map_cmap=map_cmap, vector_width=vector_width, 
                                   link_id_label=False, **kwargs)

        if isinstance(title, bool) and title is True:
            ax.set_title('Displacement map at scale %s' % projection.get_sky(scale))
        if isinstance(title, str):
            ax.set_title(title)

    for scale in scales:
        stack.add_replayable_figure("Displacement map scale %s" % scale, do_plot, scale)

    if save_filename is not None:
        stack.save_all(os.path.join(ctx.get_data_dir(), save_filename))
        stack.destroy()
    else:
        stack.show()


def get_velocities_data(ctx, scales=None, region_list=None, min_link_size=2, 
                        feature_filter=None, add_match_features=False,
                        **kargs):
    if not ctx.result.has_match_result():
        print "No result found. Run match_all() first."
        return

    scales = _get_scales(scales, ctx.result.get_scales())

    if len(scales) == 0:
        print "No result found for this scales. Available scales:", ctx.result.get_scales()
        return

    ref_img = ctx.get_ref_image()
    prj = ctx.get_projection(ref_img)

    data = wiseutils.VelocityData.from_results(ctx.get_result(), prj, scales=scales, **kargs)
    
    if feature_filter is not None:
        data.filter(feature_filter)

    data.df = data.df.groupby('link_id').filter(lambda x: len(x) > min_link_size)
    
    if add_match_features:
        features = wfeatures.FeaturesGroup()
        added_idx = [] 
        for idx, match in zip(data.df.index, data.df.match):
            if not match in data.df.features.values:
                features.add_feature(match)
                added_idx.append(idx)
        data.add_features_group(features, prj)
        data.df.loc[[f.get_id() for f in features], 'link_id'] = data.df.loc[added_idx, 'link_id'].values

    if region_list is not None:
        data.add_col_region(region_list)

    return data.df


def create_poly_region(ctx, img=None, features=None):
    '''Create a region file
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    img : :class:`libwise.imgutils.Image`, optional
        Image to use as background
    features : :class:`wise.features.FeaturesGroup`, optional
        Features to plot on top of the image


    .. _tags: task_general
    '''
    from libwise.app import PolyRegionEditor

    if img is None:
        img = ctx.get_ref_image()

    prj = ctx.get_projection(img)

    editor = PolyRegionEditor.PolyRegionEditor(img, prj=prj, current_folder=ctx.get_data_dir())

    if features is not None:
        wiseutils.plot_features(editor.ax, features, mode='com', c=plotutils.black, s=20)
    editor.start()


def list_saved_results(ctx):
    '''List all saved results
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`


    .. _tags: task_general
    '''
    ref_img = ctx.get_ref_image()
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
        if isinstance(epochs[0], datetime.datetime):
            epochs = [epoch.strftime("%Y-%m-%d") for epoch in epochs]
        n = len(epochs)
        first, last = epochs[0], epochs[-1]
        link_builder_files = glob.glob(os.path.join(os.path.dirname(file), '*.ms.dfc.dat'))
        scales = [f.rsplit('_')[-1].split('.')[0] for f in link_builder_files]

        data.append([name, n, first, last, bool(len(scales)), ", ".join(scales)])

    print nputils.format_table(data, header)


def list_tasks():
    """Lists all WISE tasks
    """
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


def stack_cross_correlation(ctx, config, debug=0, nwise=2, stack=None):
    """Perform a Stack Cross Correlation analysis
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    config : :class:`wise.scc.SCCConfiguration`
    debug : int, optional
    nwise : int, optional
    stack : :class:`libwise.plotutils.FigureStack`, optional
    
    Returns
    -------
    :class:`wise.scc.StackCrossCorrelation` : a StackCrossCorrelation containing the results of the analysis


    .. _tags: task_scc
    """
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
        print "-> Numbers of detected SSP: %s" % ", ".join([str(k.size()) for k in res1])
        res2 = ctx.detection(img2, filter=config.get("filter2"))

        scc_result.process(prj, res1, res2)

    return scc_result


def bootstrap_scc(ctx, config, output_dir, n, nwise = 2, append=False, 
                  verbose=False, seperate_scales=False):
    """Perform Stack Cross Correlation analysis n time and store results in output_dir
    
    Parameters
    ----------
    ctx : :class:`wise.project.AnalysisContext`
    config : :class:`wise.scc.SCCConfiguration`
    output_dir : str
    n : int
    append : bool, optional
        Append results
    seperate_scales : bool, optional


    .. _tags: task_scc
    """
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
        print "-> Numbers of detected SSP: %s" % ", ".join([str(k.size()) for k in res1])
        res2 = ctx.detection(img2, filter=config.get("filter2"))
        print "-> Numbers of detected SSP: %s" % ", ".join([str(k.size()) for k in res2])

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

        scc_result = scc.StackCrossCorrelation(config, verbose=verbose)

        for shuffled_pair, epoch_pair in zip(files_pair, epochs_pair):
            res1 = all_res1[shuffled_pair[0]]
            res2 = all_res2[shuffled_pair[-1]]

            # for segments2, segments2_img in zip(res2, all_segments2_img[shuffled_pair[-1]]):
            #     segments2.get_img().data = nputils.shift2d(segments2_img, 
            #                                     np.random.uniform(-random_shift, random_shift, 2))

            res1.epoch = epoch_pair[0]
            res2.epoch = epoch_pair[-1]

            delta_t, velocity_pix, tol_pix = scc_result.get_velocity_resolution(prj, res1, res2)

            if not nputils.in_range(tol_pix, config.get("tol_pix_range")):
                print "-> Skip: Not in the allowed range of pixel velocity resolution:", tol_pix
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


def _get_scales(scales, all_scales):
    if isinstance(scales, (list, set, np.ndarray)):
        return sorted(set(scales) & set(all_scales))
    if isinstance(scales, (int, float)):
        if not scales in all_scales:
            return []
        return [scales]
    return sorted(all_scales)


def _test_load_3c120_config():
    import astropy.cosmology as cosmology

    cosmology.default_cosmology.set(cosmology.FlatLambdaCDM(H0=71 * u.km / u.s / u.Mpc, Om0=0.27))

    BASE_DIR = os.path.expanduser("~/data/3c120/mojave")

    ctx = project.AnalysisContext()

    # data configuration
    ctx.config.data.data_dir = os.path.join(BASE_DIR, "run001")
    ctx.config.data.object_z = 0.033

    def get_bg(ctx, img):
        return img.data[:, :800]

    def build_mask(ctx):
        stack = ctx.get_ref_image()
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

def _test_3c120_plot_distance_from_core():
    ctx = _test_load_3c120_config()

    load(ctx, "ms_sep5", min_link_size=5)

    start_date = datetime.datetime(2006, 1, 1)
    end_date = datetime.datetime(2012, 1, 1)
    date_filter = wfeatures.DateFilter(start_date=start_date, end_date=end_date)
    date_filter2 = wfeatures.DateFilter(start_date=None, end_date=start_date)
    dfc_filter = wfeatures.DfcFilter(0, 10, u.mas)
    dfc_filter2 = wfeatures.DfcFilter(5, 25, u.mas)

    feature_filter = (date_filter & dfc_filter) | (date_filter2 & dfc_filter2) 

    plot_separation_from_core(ctx, scales=None, feature_filter=None, pa=False, fit_fct=nputils.LinearFct)


def _test_3c120_view_links():
    ctx = _test_load_3c120_config()

    load(ctx, "ms_sep5", min_link_size=5)

    start_date = datetime.datetime(2006, 1, 1)
    end_date = datetime.datetime(2012, 1, 1)
    date_filter = wfeatures.DateFilter(start_date=start_date, end_date=end_date)
    date_filter2 = wfeatures.DateFilter(start_date=None, end_date=start_date)
    dfc_filter = wfeatures.DfcFilter(0, 10, u.mas)
    dfc_filter2 = wfeatures.DfcFilter(5, 25, u.mas)

    feature_filter = (date_filter & dfc_filter) | (date_filter2 & dfc_filter2) 

    view_links(ctx, scales=8, feature_filter=feature_filter)


def _test_3c120_plot_all_features():
    ctx = _test_load_3c120_config()

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


def _test_3c120_view_wds():
    ctx = _test_load_3c120_config()

    detection_all(ctx)

    view_wds(ctx, scale=8)


test = [bootstrap_scc, stack_cross_correlation]


if __name__ == '__main__':
    # _test_3c120_plot_distance_from_core()
    # _test_3c120_plot_all_features()
    # list_saved_results(_test_load_3c120_config())
    # _test_3c120_view_links()
    _test_3c120_view_wds()
