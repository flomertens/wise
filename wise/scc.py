import datetime
from types import NoneType

import numpy as np
from scipy.ndimage.interpolation import rotate, zoom

from libwise import nputils, imgutils, plotutils

import astropy.units as u
from astropy.time import TimeDelta

p2i = plotutils.p2i


class SCCConfiguration(nputils.BaseConfiguration):

    def __init__(self):
        data = [
        ["unit", u.mas / u.year, "Velocity unit", nputils.validator_is(u.Unit)],
        ["bounds", [1, 1, 1, 1], "Velocity bounds", nputils.validator_list(4, (int, float))],
        ["filter1", None, "Filter for the first image", nputils.validator_is((NoneType, nputils.AbstractFilter))],
        ["filter2", None, "Filter for the second image", nputils.validator_is((NoneType, nputils.AbstractFilter))],
        ["tol_pix_range", [4, 25], "Allowed range of pixel velocity resolution", nputils.validator_list(2, int)],
        ["ncc_threshold", 0.6, "Threshold for the NCC", nputils.validator_is(bool)],
        ["factor", 10, "Zoom factor of the resulting map", nputils.validator_in_range(1, 20)],
        ["method", 'ncc_peaks_direct', "Method used to compute the SCC", lambda v: v in ['ncc', 'ncc_peaks', 'ncc_peaks_direct']],
        ["vector_direction", None, "Project the result on this direction", lambda v: v == 'position_angle' or nputils.is_callable(v) or isinstance(v, (list, np.ndarray, NoneType))],
        ["velocity_trans", None, "Do any transform on the velocity vector, pre projection", lambda v: nputils.is_callable(v)],
        ["rnd_pos_shift", False, "Randomly shift the segments position", nputils.validator_is((bool, NoneType))],
        ["rnd_pos_factor", 1.5, "Factor of the standart deviation of the shift", nputils.validator_in_range(0.1, 5)],
        ["img_rnd_shift", 0, "Randomly shift the images (pixel std)", nputils.validator_in_range(0, 10)],
        ["shuffle", False, "Suffle the list of images", nputils.validator_is(bool)],
        ]

        nputils.BaseConfiguration.__init__(self, data, title="Stack cross correlation configuration")


class StackCrossCorrelation(object):

    def __init__(self, config, debug=0, stack=None, verbose=True):
        self.unit = config.get("unit")
        self.verbose = verbose
        self.bounds = np.array(config.get("bounds"))
        self.factor = config.get("factor")
        self.debug = debug
        self.stack = stack
        self.mode = config.get("method")
        self.x_dir = config.get("vector_direction")
        self.agg_fct = np.mean
        self.rnd_pos_shift = config.get("rnd_pos_shift")
        self.rnd_pos_factor = config.get("rnd_pos_factor")
        self.ncc_threshold = config.get("ncc_threshold")
        self.vel_trans = config.get("velocity_trans")

        if self.stack is None:
            self.stack = plotutils.FigureStack()

        self.global_ncc_scales = dict()
        self.global_ncc_scales_n = dict()
        self.global_ncc_extent = np.array([-self.bounds[2], self.bounds[3], 
                                           -self.bounds[0], self.bounds[1]])

    def ncc_segment(self, segment1, segments2, tol):
        shape = [tol, tol]
        region1 = segment1.get_image_region()
        scale = segment1.get_segmented_image().get_scale()
        i1 = segments2.get_img().get_data()
        i1 = nputils.zoom(i1, region1.get_center(), [2 * tol, 2 * tol])
        i2 = region1.get_region()
        mask = (i2 > 0)

        if min(region1.get_region().shape) <= 3:
            return None

        if self.mode == 'ncc':
            ncc = nputils.norm_xcorr2(i1, i2, mode='same')
            ncc[ncc < self.ncc_threshold] = 0
            # ncc = nputils.weighted_norm_xcorr2(i1, i2, (i2 > 0), mode='same')
            data = nputils.resize(ncc, shape)
        elif self.mode == 'ncc_peaks':
            data = np.zeros(shape)
            ncc = nputils.norm_xcorr2(i1, i2, mode='same')
            ncc = nputils.resize(ncc, shape)
            peaks = nputils.find_peaks(ncc, 4, self.ncc_threshold, fit_gaussian=False)
            for peak in peaks[:]:
                data += imgutils.gaussian(shape, width=8, center=peak) * ncc[tuple(peak)]
        return data

    def __get_delta_time(self, time):
        if isinstance(time, datetime.timedelta):
            time = time.total_seconds() * u.second
        if isinstance(time, TimeDelta):
            time = time.to(u.second)
        return time.decompose()

    def get_velocity_resolution(self, prj, img1, img2):
        delta_t = np.abs(img2.get_epoch() - img1.get_epoch())

        if delta_t == 0:
            velocity_pix = 1
        else:
            if self.unit.is_equivalent(u.m / u.s):
                velocity_pix = prj.proper_velocity([0, 0], [0, 1], delta_t).to(self.unit).value
            else:
                velocity_pix = prj.angular_velocity([0, 0], [0, 1], delta_t).to(self.unit).value

        max_bounds = max(self.bounds)
        tol_pix = np.ceil(max_bounds / velocity_pix)

        return delta_t, velocity_pix, tol_pix

    def process(self, prj, res1, res2):
        delta_t, velocity_pix, tol_pix = self.get_velocity_resolution(prj, res1, res2)
        max_bounds = max(self.bounds)

        if self.verbose:
            print "Processing:", res1.get_epoch(), res2.get_epoch(), delta_t, velocity_pix, tol_pix

        if self.debug >= 2:
            fig, ax_check0 = self.stack.add_subplots("Segments check 1", ncols=1)
            fig, ax_check1 = self.stack.add_subplots("Segments check 2", ncols=1)

        epoch_all_mean_ncc = []

        for segments1, segments2 in zip(res1, res2):
            if self.debug >= 2:
                imshow_segmented_image(ax_check0, segments1, alpha=1, projection=prj)
                imshow_segmented_image(ax_check1, segments2, alpha=1, projection=prj)

            scale = segments1.get_scale()
            # print "Scale %s: %s" % (scale, len(segments1))

            all_ncc = []
            all_ncc_shape = None
            for segment1 in segments1:
                x_dir = self.x_dir
                if x_dir == 'position_angle':
                    x_dir = prj.p2s(p2i(segment1.get_coord()))
                elif nputils.is_callable(x_dir):
                    x_dir = self.x_dir(prj.p2s(p2i(segment1.get_coord())))

                if self.mode == 'ncc_peaks_direct':
                    region1 = segment1.get_image_region()
                    
                    if min(region1.get_region().shape) <= 3:
                        continue

                    if self.rnd_pos_shift:
                        shift = np.random.normal(0, self.rnd_pos_factor * segment1.get_coord_error(min_snr=3))
                        region1.set_shift(shift)

                    i1 = segments2.get_img().get_data()
                    # i1 = nputils.shift2d(i1, np.array([4 / velocity_pix] * 2))
                    i1 = nputils.zoom(i1, region1.get_center(), [2 * tol_pix, 2 * tol_pix])
                    i2 = region1.get_region()
                    shape = [self.factor * (self.bounds[0] + self.bounds[1]), self.factor * (self.bounds[2] + self.bounds[3])]
                    data = np.zeros(shape)

                    # print i1.shape, i2.shape
                    ncc = nputils.norm_xcorr2(i1, i2, mode='same')
                    peaks = nputils.find_peaks(ncc, 4, self.ncc_threshold, fit_gaussian=False)
                    if len(peaks) > 0:
                        delta_pix = p2i(peaks - np.array(ncc.shape) / 2)

                        delta = prj.pixel_scales() * np.array(delta_pix)  * prj.unit
                        v = delta / self.__get_delta_time(delta_t)

                        if self.vel_trans:
                            v = self.vel_trans(prj.p2s(p2i(segment1.get_coord())), v.T).T

                        if x_dir is not None:
                            vx, vy = nputils.vector_projection(v.T, x_dir) * v.unit
                        else:
                            vx, vy = v.T

                        vx = vx.to(self.unit).value
                        vy = vy.to(self.unit).value
                        d = np.array([vx, vy])

                        if len(vx) > 0:
                            ix = self.factor * (self.bounds[0] + vx)
                            iy = self.factor * (self.bounds[2] + vy)

                            widths = np.array([[4 * self.factor * velocity_pix] * 2] * len(peaks))
                            # widths = np.array([[1] * 2] * len(peaks))
                            centers = np.array([iy, ix]).T
                            heights = ncc[tuple(np.array([tuple(k) for k in peaks]).T)]
                            # print widths, centers, heights
                            data = imgutils.multiple_gaussian(shape, heights, widths, centers)

                    ncc = data
                else:
                    ncc = self.ncc_segment(segment1, segments2, 2 * tol_pix)
                    if ncc is not None:
                        # print ncc.shape, nputils.coord_max(ncc)
                        ncc = zoom(ncc, velocity_pix * self.factor, order=3)
                        # zoom will shift the center
                        ncc = nputils.shift2d(ncc, [- (velocity_pix * self.factor) / 2.] * 2)
                        # print ncc.shape, nputils.coord_max(ncc), velocity_pix * self.factor
                        ncc = nputils.zoom(ncc, np.array(ncc.shape) / 2, [2 * max_bounds * self.factor, 2 * max_bounds * self.factor])
                        # print ncc.shape, nputils.coord_max(ncc)
                        # ncc = ncc[self.factor * (max_bounds - self.bounds[0]):self.factor * (max_bounds + self.bounds[1]),
                        #           self.factor * (max_bounds - self.bounds[2]):self.factor * (max_bounds + self.bounds[3])]
                        # print ncc.shape

                        if x_dir is not None:
                            angle_rad = - np.arctan2(x_dir[1], x_dir[0])
                            ncc = rotate(ncc, - angle_rad / (2 * np.pi) * 360, reshape=False, order=2)

                if ncc is not None:
                    if self.debug >= 3:
                        fig, (ax0, ax1, ax2) = self.stack.add_subplots("Segment %s" % segment1.get_segmentid(), ncols=3)
                        r1 = segment1.get_image_region()
                        ax0.imshow(r1.get_region())
                        i2 = nputils.zoom(segments2.get_img().get_data(), r1.get_center(), [2 * tol_pix, 2 * tol_pix])
                        ax1.imshow(i2, norm=plotutils.LogNorm(), 
                                   extent=(-tol_pix, tol_pix, -tol_pix, tol_pix))
                        plotutils.img_axis(ax1)

                        ax2.imshow(ncc, extent=self.global_ncc_extent)
                        plotutils.img_axis(ax2)
                    
                    all_ncc.append(ncc)

                    if all_ncc_shape is None:
                        all_ncc_shape = ncc.shape

            if len(all_ncc) > 0:
                if scale not in self.global_ncc_scales:
                    self.global_ncc_scales[scale] = []
                    self.global_ncc_scales_n[scale] = 0
                mean_ncc = self.agg_fct(np.array(all_ncc), axis=0)
                epoch_all_mean_ncc.append(mean_ncc)

                if self.debug >= 1:
                    fig, ax = self.stack.add_subplots("Ncc epoch %s vs %s scale %s" % (res1.get_epoch(), res2.get_epoch(), scale))
                    ax.imshow(mean_ncc, extent=self.global_ncc_extent)
                    plotutils.img_axis(ax)

                self.global_ncc_scales[scale].append(mean_ncc)
                self.global_ncc_scales_n[scale] += len(all_ncc)

        if self.debug >= 0.5 and len(epoch_all_mean_ncc) > 0:
            epoch_mean_ncc = self.agg_fct(np.array(epoch_all_mean_ncc), axis=0)
            fig, ax = self.stack.add_subplots("Ncc epoch %s vs %s" % (res1.get_epoch(), res2.get_epoch()))
            ax.imshow(epoch_mean_ncc, extent=self.global_ncc_extent)
            plotutils.img_axis(ax)

    def get_result(self):
        return nputils.get_items_sorted_by_keys(self.global_ncc_scales)

    def get_n(self):
        return np.sum(self.global_ncc_scales_n.values())

    def get_mean_ncc_scales(self, smooth_len=3):
        mean_global_ncc_scales = dict()
        for scale, ncc in self.get_result():
            mean_global_ncc_scales[scale] = nputils.smooth(self.agg_fct(ncc, axis=0), smooth_len, mode='same')
        return mean_global_ncc_scales

    def get_global_ncc(self, smooth_len=3):
        global_ncc = self.agg_fct(self.get_mean_ncc_scales(smooth_len=smooth_len).values(), axis=0)
        return global_ncc

    def plot_result(self):
        global_ncc = self.get_global_ncc(smooth_len=5)
        if isinstance(global_ncc, np.ndarray):
            mean_global_ncc_scales = self.get_mean_ncc_scales(smooth_len=5)

            fig, ax0 = self.stack.add_subplots("Global NCC", ncols=1)
            
            peaks = nputils.find_peaks(global_ncc, 3, global_ncc.mean() + 2 * global_ncc.std(), fit_gaussian=True)
            peaks = sorted(peaks, key=lambda p: global_ncc[tuple(p)], reverse=True)
            if len(peaks) < 10:
                for peak in peaks:
                    p = (np.array(peak) / float(self.factor) - np.array([self.bounds[0], self.bounds[2]]))
                    print "Peak at %s (norm:%s, intensity:%s, pix:%s)" % (p[::-1], np.linalg.norm(p), global_ncc[tuple(peak)], peak)
            
            ax0.imshow(global_ncc, extent=self.global_ncc_extent)
            plotutils.img_axis(ax0)

            for scale, ncc in nputils.get_items_sorted_by_keys(mean_global_ncc_scales):
                if ncc.ndim != 2: 
                    continue
                fig, ax = self.stack.add_subplots("Global NCC scale %s" % scale, ncols=1)
                ax.imshow(ncc, extent=self.global_ncc_extent)
                plotutils.img_axis(ax)

    def show(self):
        self.stack.show()
